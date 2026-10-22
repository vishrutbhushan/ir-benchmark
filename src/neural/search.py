"""Exhaustive hand-coded cosine ranking and document aggregation; durable index."""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time
import numpy as np
from src.neural.common import connect
from src.neural.model import Encoder


def aggregate_documents(scores,doc_indices,document_count):
    """Hand-coded max-passage scoring with lowest passage index as deterministic tie break."""
    scores=np.asarray(scores,dtype=np.float32)
    doc_indices=np.asarray(doc_indices,dtype=np.int64)
    maxima=np.full(document_count,-np.inf,dtype=np.float32)
    np.maximum.at(maxima,doc_indices,scores)
    winners=np.full(document_count,len(scores),dtype=np.int64)
    candidates=np.flatnonzero(scores==maxima[doc_indices])
    np.minimum.at(winners,doc_indices[candidates],candidates)
    return maxima,winners


class SearchEngine:
    def __init__(self,root,device='cuda'):
        self.root=Path(root)
        self.summary=json.loads((self.root/'embeddings/summary.json').read_text())
        checkpoint=json.loads((self.root/'embeddings/checkpoint.json').read_text())
        if not checkpoint.get('complete'):
            raise RuntimeError('Embedding build is incomplete')
        passage_summary=json.loads((self.root/'passages/summary.json').read_text())
        extraction=json.loads((self.root/'extraction/summary.json').read_text())
        if self.summary.get('input_fingerprint')!=passage_summary['fingerprint'] or passage_summary.get('input_fingerprint')!=extraction['fingerprint']:
            raise RuntimeError('Upstream corpus changed; rebuild embeddings before searching')
        if self.summary['passages']!=passage_summary['passages'] or self.summary['documents']!=extraction['retained_documents']:
            raise RuntimeError('Artifact counts disagree; rebuild the neural pipeline')
        self.encoder=Encoder(self.root/'model',device)
        self.torch=self.encoder.torch
        self.device=device
        self.document_db=connect(self.root/'extraction/corpus.sqlite',True)
        self.passage_db=connect(self.root/'passages/passages.sqlite',True)
        self.documents=[dict(row) for row in self.document_db.execute('SELECT doc_index,doc_id,title,url,technology,category FROM documents ORDER BY doc_index')]
        self.ids=np.array([r['doc_id'] for r in self.documents])
        self.doc_indices=np.load(self.root/'embeddings/doc_indices.npy',mmap_mode='r')
        matrix=np.load(self.root/'embeddings/vectors.npy',mmap_mode='r')
        self.matrix=matrix
        self.resident=None
        if device=='cuda':
            self.torch.cuda.empty_cache()
            free,_=self.torch.cuda.mem_get_info()
            if matrix.nbytes < free*.55:
                self.resident=self.torch.as_tensor(np.array(matrix),device=device)
        self.lock=threading.RLock()
        self.cache=OrderedDict()
        self.query_cache=OrderedDict()
        self.technology_options=sorted({d['technology'] for d in self.documents})
        self.category_options=sorted({d['category'] for d in self.documents})

    def warmup(self):
        """Prepare lazy model/CUDA kernels before the server reports readiness."""
        started=time.perf_counter()
        vector=self.encoder.encode(['Find relevant technical API documentation'],query=True)[0]
        self._cosines(vector)
        self.startup_warmup_ms=round((time.perf_counter()-started)*1000,2)
        return self.startup_warmup_ms

    def _cosines(self,vector):
        # FP32 accumulation/output explicitly avoids FP16 dot-product accumulation.
        torch=self.torch
        q=torch.as_tensor(vector,device=self.device,dtype=torch.float32)
        n=len(self.matrix)
        scores=np.empty(n,dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,n,32768):
                end=min(start+32768,n)
                block=self.resident[start:end] if self.resident is not None else torch.as_tensor(np.array(self.matrix[start:end]),device=self.device)
                # Storage is FP16; normalize again in FP32 to compute cosine of persisted values.
                block=block.float()
                norms=torch.linalg.vector_norm(block,dim=1).clamp_min(1e-12)
                scores[start:end]=((block@q)/norms).cpu().numpy()
        return scores

    def search(self,query,limit=100,technology='',category='',use_cache=True,offset=0,page_size=None):
        query=query.strip()
        if not query or len(query)>4000:
            raise ValueError('Enter a query between 1 and 4000 characters')
        if not 1<=limit<=1000:
            raise ValueError('limit must be between 1 and 1000')
        if offset<0 or (page_size is not None and not 1<=page_size<=1000):
            raise ValueError('Invalid pagination offset or page size')
        if technology and technology not in self.technology_options:
            raise ValueError('Unknown technology')
        if category and category not in self.category_options:
            raise ValueError('Unknown category')
        key=(self.summary['fingerprint'],query,technology,category,limit)
        started=time.perf_counter()
        with self.lock:
            if use_cache and key in self.cache:
                ranking=self.cache.pop(key);self.cache[key]=ranking
                return self._render_ranked(ranking,offset,page_size,started,cache_hit=True)
            embedded=time.perf_counter()
            if use_cache and query in self.query_cache:
                q=self.query_cache.pop(query);self.query_cache[query]=q
            else:
                try:
                    q=self.encoder.encode([query],query=True,max_length=512)[0]
                except ValueError as error:
                    if 'Input exceeds' in str(error):
                        raise ValueError('Query is too long for the model. Please shorten it.') from None
                    raise
                if use_cache:
                    self.query_cache[query]=q
                    while len(self.query_cache)>64:self.query_cache.popitem(last=False)
            if not np.isfinite(q).all() or abs(float(np.linalg.norm(q))-1)>.002:
                raise ValueError('The model did not produce a valid query vector')
            embedding_ms=(time.perf_counter()-embedded)*1000
            rank_start=time.perf_counter()
            scores=self._cosines(q)
            maxima,winners=aggregate_documents(scores,self.doc_indices,len(self.documents))
            valid=np.isfinite(maxima)
            if technology:
                valid &= np.array([d['technology']==technology for d in self.documents])
            if category:
                valid &= np.array([d['category']==category for d in self.documents])
            candidates=np.flatnonzero(valid)
            # Deterministic document-ID tie breaking, after aggregating EVERY passage.
            order=candidates[np.lexsort((self.ids[candidates],-maxima[candidates]))][:limit]
            ranking_ms=(time.perf_counter()-rank_start)*1000
            ranking={'query':query,'index_version':self.summary['fingerprint'],
                     'total_documents':len(candidates),'retrieved_depth':len(order),
                     'embedding_ms':round(embedding_ms,2),'ranking_ms':round(ranking_ms,2),
                     '_order':order,'_winning_passages':winners[order],'_scores':maxima[order]}
            if use_cache:
                self.cache[key]=ranking
                while len(self.cache)>64:
                    self.cache.popitem(last=False)
            return self._render_ranked(ranking,offset,page_size,started,cache_hit=False)

    def _render_ranked(self,ranking,offset,page_size,started,cache_hit):
        """Fetch only visible passages; keep compact ranked IDs/scores in the LRU."""
        stop=len(ranking['_order']) if page_size is None else min(offset+page_size,len(ranking['_order']))
        results=[]
        for position in range(offset,stop):
            index=int(ranking['_order'][position])
            passage_index=int(ranking['_winning_passages'][position])
            passage=dict(self.passage_db.execute('SELECT passage_id,heading,anchor,kind,text FROM passages WHERE passage_index=?',(passage_index,)).fetchone())
            results.append({**self.documents[index],'rank':position+1,'score':float(ranking['_scores'][position]),'passage':passage})
        payload={key:value for key,value in ranking.items() if not key.startswith('_')}
        if cache_hit:
            payload.update(embedding_ms=0.,ranking_ms=0.)
        return {**payload,'returned':len(results),'elapsed_ms':round((time.perf_counter()-started)*1000,2),'cache_hit':cache_hit,'results':results}

    def close(self):
        self.document_db.close()
        self.passage_db.close()
        self.cache.clear()
        self.query_cache.clear()
        self.resident=None
