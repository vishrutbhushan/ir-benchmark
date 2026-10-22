"""Recompute four zone-cosine features for frozen development pairs.

This reads the existing extraction; it does not build or replace production indexes,
assign relevance labels, fit weights, or use final evaluation queries.
"""
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.neural.common import connect, digest_file, report
from src.neural.model import Encoder
from src.neural.preprocessing.chunk import passage_parts

from .zones import ZONES, document_zones


def load_examples(dataset):
    """One self-contained training table; no joins or evidence dependency."""
    with (Path(dataset) / 'training.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    splits = {}
    for row in rows:
        if row['split'] != splits.setdefault(row['query_id'], row['split']):
            raise ValueError('All judgments of a query must share its split')
        if row['relevant'] not in {'0', '1'}:
            raise ValueError('Relevance must be binary')
    return rows


def build_features(root, device='cuda', batch_size=32):
    root = Path(root)
    dataset = Path('data/neural/zone_weights')
    source = dataset / 'training.csv'
    rows = load_examples(dataset)
    queries = {row['query_id']: row['query_text'] for row in rows}
    with Path('queries.jsonl').open() as stream:
        evaluation = {json.loads(line)['query_text'].strip().casefold() for line in stream if line.strip()}
    if any(query.strip().casefold() in evaluation for query in queries.values()):
        raise ValueError('Development query duplicates a reserved evaluation query')
    if len({(r['query_id'], r['doc_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate judgment pair')
    encoder = Encoder(root / 'model', device)
    # Source zones can be long; only their checked token windows reach the model.
    encoder.tokenizer.model_max_length = 10**12
    documents = {}
    unique_texts, text_indices = [], {}
    mapping = {}
    with closing(connect(root / 'extraction/corpus.sqlite', True)) as db:
        for doc_id in sorted({r['doc_id'] for r in rows}):
            found = db.execute('SELECT record_json FROM documents WHERE doc_id=?', (doc_id,)).fetchone()
            if found is None:
                raise ValueError(f'Document absent from frozen corpus: {doc_id}')
            record = json.loads(found[0])
            documents[doc_id] = record
            mapping[doc_id] = {}
            for zone, text in document_zones(record).items():
                indices = []
                # Same rule for every zone: fixed token windows, no title/heading prefixes.
                for piece, _, _ in passage_parts(text, encoder.tokenizer, 496, 64):
                    key = hashlib.sha256(piece.encode()).hexdigest()
                    if key not in text_indices:
                        text_indices[key] = len(unique_texts)
                        unique_texts.append(piece)
                    indices.append(text_indices[key])
                mapping[doc_id][zone] = indices
    out = root / 'development'
    out.mkdir(parents=True, exist_ok=True)
    vectors = np.empty((len(unique_texts), encoder.dimension), dtype=np.float32)
    for start in range(0, len(unique_texts), batch_size):
        end = min(start + batch_size, len(unique_texts))
        vectors[start:end] = encoder.encode(unique_texts[start:end])
        if end % 512 == 0 or end == len(unique_texts):
            print(f'Development zone vectors: {end}/{len(unique_texts)}', flush=True)
    ids = sorted(queries)
    query_vectors = {}
    for start in range(0, len(ids), 16):
        keys = ids[start:start + 16]
        query_vectors.update(zip(keys, encoder.encode([queries[k] for k in keys], query=True)))
    for row in rows:
        query = query_vectors[row['query_id']]
        for zone in ZONES:
            indices = mapping[row['doc_id']][zone]
            row[f'has_{zone}'] = int(bool(indices))
            row[f'{zone}_cosine'] = ''
            if indices:
                values = vectors[indices]
                scores = (values @ query) / (np.linalg.norm(values, axis=1) * np.linalg.norm(query))
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite zone cosine')
                row[f'{zone}_cosine'] = f'{float(scores.max()):.8f}'
    destination = source
    with destination.open('w', newline='', encoding='utf-8') as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    stats = {
        'queries': len(queries), 'pairs': len(rows), 'unique_documents': len(documents),
        'categories': len({r['query_category'] for r in rows}),
        'technologies': len({r['doc_technology'] for r in rows}),
        'relevant_pairs': sum(int(r['relevant']) for r in rows),
        'nonrelevant_pairs': sum(1-int(r['relevant']) for r in rows),
        'usable_pairs': sum(int(r['use_for_weight_learning']) for r in rows),
        'unique_zone_windows': len(unique_texts),
        'feature_rule': 'max cosine per pure zone; 496-token content budget; up to 64 overlap; no context prefixes',
        'scope': 'Current structured extraction subset only, not a production zone-index rebuild',
        'title_caveat': 'Features use existing titles, including existing manifest/URL fallbacks',
        'zones': list(ZONES),
        'training_csv_sha256': digest_file(destination),
        'corpus_fingerprint': json.loads((root / 'extraction/summary.json').read_text())['fingerprint'],
    }
    report(out, 'Zone-weight development data', stats, {
        'relevance': dict(Counter('relevant' if int(r['relevant']) else 'nonrelevant' for r in rows)),
        'query_categories': dict(Counter({r['query_id']: r['query_category'] for r in rows}.values())),
        'split_pairs': dict(Counter(r['split'] for r in rows)),
        'zone_presence': {z: sum(int(r[f'has_{z}']) for r in rows) for z in ZONES},
    })
    print(f'Saved {destination}; labels unchanged; no weights fitted.', flush=True)
    return stats
