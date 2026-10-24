"""Complete zone-index, reference cosine, retrieval and cache verification."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from src.neural.common import digest_file,report
from src.neural.learning.zone_weights.zones import ZONES
from src.neural.preprocessing.integrity import audit_zone_coverage,audit_vector_mapping
from src.neural.scoring import aggregate_documents
from src.neural.search import MODES


def verify(engine,root,query_path):
    root=Path(root);started=time.perf_counter()
    coverage=audit_zone_coverage(engine.document_db,engine.stores)
    q=engine._query('Python asynchronous task cancellation')
    actual,_,_=engine.zone_scores(q)
    hashes={};norms={};errors={}
    for column,zone in enumerate(ZONES):
        matrix,mapping=engine.matrices[zone],engine.mappings[zone]
        if audit_vector_mapping(engine.stores[zone],mapping)!=len(matrix):raise RuntimeError('Vector mapping size mismatch')
        scores=np.empty(len(matrix),dtype=np.float32);low,high=2.,0.
        for start in range(0,len(matrix),8192):
            block=np.asarray(matrix[start:start+8192],dtype=np.float32);length=np.linalg.norm(block,axis=1)
            if not np.isfinite(block).all() or np.any(np.abs(length-1)>.002):raise RuntimeError('Invalid persisted vectors')
            low,high=min(low,float(length.min())),max(high,float(length.max()))
            scores[start:start+len(block)]=(block@q)/(length*np.linalg.norm(q))
        reference,_=aggregate_documents(scores,mapping,len(engine.documents))
        present=np.isfinite(reference)
        if not np.array_equal(present,np.isfinite(actual[:,column])):raise RuntimeError('Missing-zone score mismatch')
        error=float(np.max(np.abs(reference[present]-actual[present,column])))
        if error>1e-4:raise RuntimeError('GPU/NumPy cosine disagreement')
        errors[zone]=error;norms[zone]=[low,high]
        hashes[zone]=digest_file(root/f'zones/{zone}/embeddings/vectors.npy')
    queries=[json.loads(line) for line in Path(query_path).read_text().splitlines() if line.strip()]
    timings={};counts={}
    for mode in MODES:
        elapsed=[]
        for query in queries:
            result=engine.search(query['query_text'],mode=mode,limit=100,use_cache=False)
            ids=[r['doc_id'] for r in result['results']]
            if len(ids)!=100 or len(ids)!=len(set(ids)):raise RuntimeError('Invalid result depth/IDs')
            scores=[r['score'] for r in result['results']]
            if scores!=sorted(scores,reverse=True):raise RuntimeError('Rank order mismatch')
            elapsed.append(result['elapsed_ms'])
        timings[mode+'_p50_ms']=float(np.median(elapsed));timings[mode+'_p95_ms']=float(np.percentile(elapsed,95))
        counts[mode]=len(queries)
        print(f'Verified {mode}: {len(queries)} queries.',flush=True)
    first=engine.search('PostgreSQL JSONB indexing',limit=30,offset=0,page_size=10)
    second=engine.search('PostgreSQL JSONB indexing',limit=30,offset=10,page_size=10)
    if not second['cache_hit'] or second['results'][0]['rank']!=11 or {r['doc_id'] for r in first['results']}&{r['doc_id'] for r in second['results']}:raise RuntimeError('Pagination/cache mismatch')
    filtered=engine.search('reference documentation',limit=10,technology='Python')
    if any(r['technology']!='Python' for r in filtered['results']):raise RuntimeError('Technology filter mismatch')
    stats={'index_fingerprint':engine.summary['fingerprint'],'documents':len(engine.documents),'windows':sum(len(m) for m in engine.matrices.values()),
           'coverage':coverage,'vectors_sha256':hashes,'norm_ranges':norms,'cosine_reference_max_error':errors,
           'query_runs':counts,'all_vector_document_mappings_checked':True,'cached_pagination_checked':True,
           'query_suite_sha256':digest_file(query_path),'seconds':round(time.perf_counter()-started,2),'latencies':timings}
    report(root/'verification','Complete neural verification',stats,{'latencies_ms':timings,'query_categories':dict(Counter(q['category'] for q in queries))})
    return stats
