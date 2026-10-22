"""Full artifact checks and engineering smoke queries; no invented relevance labels."""
from collections import Counter
import json
import time
import numpy as np
from src.neural.common import records, report, digest_file


def verify(engine,root,query_path):
    started=time.perf_counter()
    extraction=json.loads((root/'extraction/summary.json').read_text())
    all_decisions=list(records(root/'extraction/decisions.jsonl'))
    if len(all_decisions)!=extraction['source_documents'] or len({r['doc_id'] for r in all_decisions})!=len(all_decisions):
        raise RuntimeError('Source document accounting failed')
    retained={r['doc_id'] for r in all_decisions if r['status']=='retained'}
    if retained!={d['doc_id'] for d in engine.documents}:
        raise RuntimeError('Corrected dataset and searchable documents differ')
    db=engine.passage_db
    if db.execute('SELECT count(*) FROM passages').fetchone()[0]!=len(engine.matrix):
        raise RuntimeError('Passage/vector count mismatch')
    represented={r[0] for r in db.execute('SELECT DISTINCT doc_index FROM passages')}
    if represented!=set(range(len(engine.documents))):
        raise RuntimeError('Some retained documents have no searchable passage')
    from src.neural.preprocessing.integrity import audit_passage_coverage
    coverage=audit_passage_coverage(engine.document_db,db)
    verified_characters=coverage['source_characters']
    minimum,maximum=2.,0.
    for start in range(0,len(engine.matrix),10000):
        values=engine.matrix[start:start+10000].astype(np.float32)
        norms=np.linalg.norm(values,axis=1)
        if not np.isfinite(values).all() or np.any(np.abs(norms-1)>.002):
            raise RuntimeError('Full embedding verification failed')
        minimum,maximum=min(minimum,float(norms.min())),max(maximum,float(norms.max()))
    # Compare actual persisted-vector dot products against a separate NumPy cosine reference.
    q=engine.encoder.encode(['Python asynchronous task cancellation'],query=True)[0]
    scores=engine._cosines(q)
    error=0.
    for start in range(0,len(engine.matrix),32768):
        values=engine.matrix[start:start+32768].astype(np.float32)
        reference=(values@q)/(np.linalg.norm(values,axis=1)*np.linalg.norm(q))
        error=max(error,float(np.max(np.abs(scores[start:start+len(values)]-reference))))
    if error>1e-4:
        raise RuntimeError(f'GPU/NumPy cosine disagreement: {error}')
    queries=list(records(query_path))
    output=root/'verification';output.mkdir(parents=True,exist_ok=True)
    timings={}
    with (output/'rankings.jsonl').open('w',encoding='utf-8') as stream:
        latencies=[]
        for query in queries:
            result=engine.search(query['query_text'],limit=100,use_cache=False)
            ids=[r['doc_id'] for r in result['results']]
            if len(ids)!=min(100,len(retained)):
                raise RuntimeError('A smoke query failed to return the expected ranked depth')
            if len(ids)!=len(set(ids)) or any(ids[i] not in retained for i in range(len(ids))):
                raise RuntimeError('Invalid returned document IDs')
            if any(result['results'][i]['score']<result['results'][i+1]['score'] for i in range(len(ids)-1)):
                raise RuntimeError('Results not descending')
            latencies.append(result['elapsed_ms'])
            stream.write(json.dumps({'query_id':query['query_id'],'query_category':query['category'],**result},ensure_ascii=False)+'\n')
            print(f'Verified dense {query["query_id"]}: {result["elapsed_ms"]:.0f} ms, {len(ids)} unique results',flush=True)
        timings['dense_p50_ms']=round(float(np.percentile(latencies,50)),2)
        timings['dense_p95_ms']=round(float(np.percentile(latencies,95)),2)
    uncached=engine.search('PostgreSQL JSONB indexes',limit=100)
    cached=engine.search('PostgreSQL JSONB indexes',limit=100)
    if not cached['cache_hit'] or cached['results']!=uncached['results']:
        raise RuntimeError('Cache identity failed')
    filtered=engine.search('reference documentation',limit=20,technology=engine.technology_options[0])
    if any(r['technology']!=engine.technology_options[0] for r in filtered['results']):
        raise RuntimeError('Technology filter failed')
    stats={'source_documents_accounted':len(all_decisions),'searchable_documents':len(retained),'searchable_passages':len(engine.matrix),'all_documents_have_passages':True,'all_section_characters_covered':True,'all_passages_match_source_offsets':True,'section_characters_verified':verified_characters,'all_vectors_finite':True,'norm_min':minimum,'norm_max':maximum,'cosine_reference_rows':len(engine.matrix),'cosine_reference_max_error':error,'vectors_sha256':digest_file(root/'embeddings/vectors.npy'),'document_mapping_sha256':digest_file(root/'embeddings/doc_indices.npy'),'candidate_queries':len(queries),'query_runs':len(queries),'ranking':'dense cosine','cached_search_ms':cached['elapsed_ms'],'gpu_vectors_resident':engine.resident is not None,'metrics_status':'Not evaluated: relevance judgments and final 20-query suite pending','seconds':round(time.perf_counter()-started,2),**timings}
    report(output,'Full-index correctness and retrieval smoke verification',stats,{'uncached_latency_ms':timings,'coverage':{'documents':len(retained),'passages':len(engine.matrix)},'candidate_query_categories':dict(Counter(q['category'] for q in queries))})
    return stats
