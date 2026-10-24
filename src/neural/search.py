"""Exact dense, learned-zone and category-routed retrieval over four stores."""
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

import numpy as np

from src.neural.common import connect, digest_file
from src.neural.learning.zone_weights.zones import ZONES
from src.neural.model import Encoder
from src.neural.scoring import aggregate_documents, weighted_zone_scores

MODES = ('dense', 'zones', 'routed-centroid', 'routed-nb')


class SearchEngine:
    def __init__(self, root, device='cuda', corpus_root=None, require_learning=True):
        self.root = Path(root)
        self.summary = json.loads((self.root/'zones/summary.json').read_text())
        source = json.loads((self.root/'extraction/summary.json').read_text())
        if source['fingerprint'] != self.summary['source_fingerprint']:
            raise RuntimeError('Zone index is stale; run --phase all')
        if corpus_root is not None and digest_file(Path(corpus_root)/'metadata/corpus_manifest.jsonl') != source['dataset_manifest_sha256']:
            raise RuntimeError('Dataset changed; run --phase all')
        self.encoder = Encoder(self.root/'model', device)
        self.torch, self.device = self.encoder.torch, device
        self.document_db = connect(self.root/'extraction/corpus.sqlite', True)
        self.documents = [dict(r) for r in self.document_db.execute('SELECT doc_index,doc_id,title,url,technology,category FROM documents ORDER BY doc_index')]
        self.ids = np.array([d['doc_id'] for d in self.documents])
        if [d['doc_index'] for d in self.documents] != list(range(len(self.documents))):
            raise RuntimeError('Document indexes must be contiguous')
        self.stores, self.matrices, self.mappings, self.resident = {}, {}, {}, {}
        total_bytes = 0
        for zone in ZONES:
            directory = self.root/f'zones/{zone}'
            checkpoint = json.loads((directory/'embeddings/checkpoint.json').read_text())
            summary = json.loads((directory/'embeddings/summary.json').read_text())
            passages = json.loads((directory/'passages/summary.json').read_text())
            if not checkpoint['complete'] or summary['input_fingerprint'] != passages['fingerprint'] or passages['input_fingerprint'] != source['fingerprint']:
                raise RuntimeError(f'{zone}: incomplete/stale embeddings')
            self.stores[zone] = connect(directory/'passages/passages.sqlite', True)
            self.matrices[zone] = np.load(directory/'embeddings/vectors.npy', mmap_mode='r')
            self.mappings[zone] = np.load(directory/'embeddings/doc_indices.npy', mmap_mode='r')
            if self.matrices[zone].shape != (summary['passages'], self.encoder.dimension) or self.mappings[zone].shape != (summary['passages'],):
                raise RuntimeError(f'{zone}: vector/mapping shape mismatch')
            total_bytes += self.matrices[zone].nbytes
        resident = device == 'cuda' and total_bytes < self.torch.cuda.mem_get_info()[0] * .55
        for zone in ZONES:
            self.resident[zone] = self.torch.as_tensor(np.array(self.matrices[zone]), device=device) if resident else None
        self.weights = None
        weights_path = self.root/'weights/model.json'
        if weights_path.exists():
            weights = json.loads(weights_path.read_text())
            if weights['index_fingerprint'] == self.summary['fingerprint']:
                self.weights = np.array(weights['weights'], dtype=np.float32)
            elif require_learning:
                raise RuntimeError('Zone weights belong to an older index')
        self.classifier = None
        if (self.root/'classification/model.json').exists():
            from src.neural.learning.classification.train import Classifiers
            try:
                self.classifier = Classifiers(self.root)
            except RuntimeError:
                if require_learning: raise
        self.routing = json.loads((self.root/'routing/model.json').read_text()) if (self.root/'routing/model.json').exists() else None
        if self.routing and self.routing['index_fingerprint'] != self.summary['fingerprint']:
            self.routing = None
        if require_learning and self.routing and self.routing.get('weights_sha256') != digest_file(weights_path):
            raise RuntimeError('Routing was tuned for different weights; run --phase routing')
        if require_learning and self.routing and self.classifier is not None and self.routing.get('classifier_csv_sha256') != self.classifier.model['training_csv_sha256']:
            raise RuntimeError('Routing was tuned for different classification data; run --phase routing')
        if require_learning and (self.weights is None or self.classifier is None or self.routing is None):
            raise RuntimeError('Learning stages are incomplete; run --phase all')
        self.lock, self.cache, self.query_cache = threading.RLock(), OrderedDict(), OrderedDict()
        self.technology_options = sorted({d['technology'] for d in self.documents})
        self.category_options = sorted({d['category'] for d in self.documents})

    def _query(self, query):
        if query not in self.query_cache:
            self.query_cache[query] = self.encoder.encode([query], query=True)[0]
            if len(self.query_cache) > 64: self.query_cache.popitem(last=False)
        return self.query_cache[query]

    def zone_scores(self, vector, allowed=None):
        n = len(self.documents)
        values = np.full((n, len(ZONES)), -np.inf, dtype=np.float32)
        winners = np.full((n, len(ZONES)), -1, dtype=np.int64)
        torch = self.torch
        q = torch.as_tensor(vector, device=self.device, dtype=torch.float32)
        scanned = 0
        with torch.inference_mode():
            for column, zone in enumerate(ZONES):
                mapping, matrix = self.mappings[zone], self.matrices[zone]
                rows = np.arange(len(matrix)) if allowed is None else np.flatnonzero(allowed[mapping])
                scanned += len(rows)
                scores = np.empty(len(rows), dtype=np.float32)
                for start in range(0, len(rows), 32768):
                    indices = rows[start:start+32768]
                    block = self.resident[zone][torch.as_tensor(indices, device=self.device)] if self.resident[zone] is not None else torch.as_tensor(np.array(matrix[indices]), device=self.device)
                    block = block.float()
                    scores[start:start+len(indices)] = ((block @ q) / torch.linalg.vector_norm(block, dim=1).clamp_min(1e-12)).cpu().numpy()
                if len(rows):
                    maxima, local = aggregate_documents(scores, mapping[rows], n)
                    valid = np.isfinite(maxima)
                    values[:, column] = maxima
                    winners[valid, column] = rows[local[valid]]
        return values, winners, scanned

    def rank(self, query, mode='zones', technology='', category='', route_k=None):
        if mode not in MODES: raise ValueError('Unknown retrieval mode')
        if not query.strip() or len(query)>4000: raise ValueError('Enter 1–4000 query characters')
        if technology and technology not in self.technology_options: raise ValueError('Unknown technology')
        if category and category not in self.category_options: raise ValueError('Unknown category')
        q = self._query(query.strip())
        allowed = np.ones(len(self.documents), dtype=bool)
        if technology: allowed &= np.array([d['technology']==technology for d in self.documents])
        if category: allowed &= np.array([d['category']==category for d in self.documents])
        selected = []
        if mode.startswith('routed-'):
            if self.classifier is None or self.routing is None: raise RuntimeError('Train classifiers and routing first')
            method = mode.removeprefix('routed-')
            scores = self.classifier.query_scores(query, q, method)
            k = route_k or self.routing['top_classes'][method]
            possible = set(np.unique(self.classifier.document_classes[allowed]).tolist())
            selected = [int(c) for c in np.argsort(-scores, kind='stable') if c in possible][:k]
            allowed &= np.isin(self.classifier.document_classes, selected)
        zones, winners, scanned = self.zone_scores(q, None if allowed.all() else allowed)
        if mode == 'dense':
            scores = zones.max(axis=1)
            winning_zone = np.argmax(zones, axis=1)
            contributions = None
        else:
            if self.weights is None: raise RuntimeError('Fit zone weights first')
            scores, contributions = weighted_zone_scores(zones, self.weights)
            winning_zone = np.argmax(np.where(np.isfinite(zones), contributions, -np.inf), axis=1)
        candidates = np.flatnonzero(allowed & np.isfinite(scores))
        order = candidates[np.lexsort((self.ids[candidates], -scores[candidates]))]
        return {'order':order, 'scores':scores, 'zones':zones, 'winners':winners,
                'winning_zone':winning_zone, 'contributions':contributions,
                'scanned_vectors':scanned, 'candidate_documents':int(allowed.sum()), 'selected_classes':selected}

    def search(self, query, limit=100, technology='', category='', use_cache=True, offset=0, page_size=None, mode='routed-centroid'):
        if not 1 <= limit <= len(self.documents): raise ValueError('Invalid result depth')
        if offset<0 or (page_size is not None and not 1<=page_size<=1000): raise ValueError('Invalid pagination')
        started = time.perf_counter()
        key = (query.strip(), mode, technology, category)
        with self.lock:
            hit = use_cache and key in self.cache
            if hit:
                ranking = self.cache.pop(key)
            else:
                ranking = self.rank(query, mode, technology, category)
            if use_cache:
                self.cache[key] = ranking
                while len(self.cache)>16: self.cache.popitem(last=False)
            order = ranking['order'][:limit]
            stop = len(order) if page_size is None else min(offset+page_size,len(order))
            results = []
            for position in range(offset,stop):
                index = int(order[position]); column = int(ranking['winning_zone'][index]); zone = ZONES[column]
                row = self.stores[zone].execute('SELECT passage_id,text FROM passages WHERE passage_index=?',(int(ranking['winners'][index,column]),)).fetchone()
                results.append({**self.documents[index], 'rank':position+1,'score':float(ranking['scores'][index]),
                    'zone_scores':{z:(float(ranking['zones'][index,j]) if np.isfinite(ranking['zones'][index,j]) else None) for j,z in enumerate(ZONES)},
                    'zone_contributions':({z:float(ranking['contributions'][index,j]) for j,z in enumerate(ZONES)} if ranking['contributions'] is not None else None),
                    'passage':{'passage_id':row['passage_id'],'text':row['text'],'heading':zone.title(),'anchor':'','kind':zone}})
            return {'query':query,'mode':mode,'index_version':self.summary['fingerprint'], 'results':results,
                    'total_documents':len(ranking['order']),'retrieved_depth':len(order),'returned':len(results),
                    'cache_hit':hit,'elapsed_ms':round((time.perf_counter()-started)*1000,2),
                    'scanned_vectors':ranking['scanned_vectors'],'candidate_documents':ranking['candidate_documents'],
                    'selected_classes':ranking['selected_classes']}

    def warmup(self):
        start=time.perf_counter(); self.zone_scores(self._query('technical documentation'))
        self.startup_warmup_ms=round((time.perf_counter()-start)*1000,2)
        return self.startup_warmup_ms

    def close(self):
        self.document_db.close()
        for db in self.stores.values(): db.close()
        self.cache.clear();self.query_cache.clear();self.resident.clear()
