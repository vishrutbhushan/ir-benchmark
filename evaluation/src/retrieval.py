"""Export rankings and measure sequential backend retrieval with fully loaded indexes."""

import hashlib
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime,timezone

from config import load_pipeline_config,resolve_path
from .common import COLUMNS,write_csv,write_json

SYSTEMS={
    'bm25':('classical','bm25'),'bm25f':('classical','bm25f'),
    'neural-dense':('neural','dense'),'neural-zones':('neural','zones'),
    'neural-routed':('neural','routed'),'hybrid-rrf':('hybrid','rrf'),
}


def retrieve_and_measure(config_path):
    config=load_pipeline_config(config_path)
    settings=config['evaluation']['latency']
    if settings['cpu_threads'] != 1 or config['execution']['cpu_threads'] != 1:
        raise ValueError('Sequential evaluation requires exactly one CPU thread')
    import torch
    import psutil
    from retrieval import RetrievalRuntime
    from neural.src.common.gpu_lock import gpu_job_lock
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    query_path=resolve_path(config['paths']['queries'])
    query_hash=hashlib.sha256(query_path.read_bytes()).hexdigest()
    queries=[json.loads(line) for line in query_path.read_text().splitlines() if line.strip()]
    methods=[x['id'] for x in config['evaluation']['search_engines']]
    results=resolve_path(config['paths']['evaluation_results_dir'])
    samples=[];rankings={m:[] for m in methods};canonical={};detail={}
    process=psutil.Process()
    runtime=RetrievalRuntime(config_path)
    with gpu_job_lock():
        runtime.load()
        if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
            raise RuntimeError('A component changed the evaluation CPU thread limit')
        engine=runtime.neural
        if not all(v is not None and v.device.type==engine.device for v in engine.resident.values()):
            raise RuntimeError('The complete neural index is not resident on the configured device')
        index=runtime.classical['bm25'].index
        prepared=json.loads((resolve_path(config["paths"]["neural_output_dir"])/"step_01_extraction/summary.json").read_text())
        if index.corpus_manifest_sha256 != prepared["dataset_manifest_sha256"]:
            raise RuntimeError("Classical and neural corpus identities disagree")
        if len(index.postings)!=len(index.vocabulary):
            raise RuntimeError('The complete classical index was not loaded')
        residency=dict(classical_terms=len(index.postings),classical_documents=index.total_docs,
                       neural_vectors=sum(len(v) for v in engine.resident.values()),
                       device=engine.device,cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                       rss_bytes=process.memory_info().rss,
                       cuda_allocated_bytes=torch.cuda.memory_allocated() if engine.device=='cuda' else 0)
        print('All indexes loaded:',json.dumps(residency),flush=True)
        original_encode=engine.encoder.encode
        original_rank=engine.rank
        def sync():
            if engine.device=='cuda':torch.cuda.synchronize()
        def encode(*args,**kwargs):
            sync();started=time.perf_counter();value=original_encode(*args,**kwargs);sync()
            detail['encoding_ms']+=(time.perf_counter()-started)*1000
            return value
        def rank(*args,**kwargs):
            previous=detail['encoding_ms'];sync();started=time.perf_counter();value=original_rank(*args,**kwargs);sync()
            detail['ranking_ms']+=(time.perf_counter()-started)*1000-(detail['encoding_ms']-previous)
            detail['scanned_vectors']=value['scanned_vectors']
            detail['candidate_documents']=value['candidate_documents']
            return value
        engine.encoder.encode=encode;engine.rank=rank
        def reset():
            runtime.clear_caches()
            detail.update(encoding_ms=0.,ranking_ms=0.,scanned_vectors=0,candidate_documents=0)
        try:
            for query in queries[:settings['warmup_queries']]:
                for method in methods:
                    reset();runtime.retrieve(method,query['query_text'],config['evaluation']['results_per_query'],use_cache=False)
            for repetition in range(settings['repetitions']):
                for qi,query in enumerate(queries):
                    shift=(qi+repetition)%len(methods)
                    for method in methods[shift:]+methods[:shift]:
                        reset();sync();started=time.perf_counter()
                        rows=runtime.retrieve(method,query['query_text'],config['evaluation']['results_per_query'],use_cache=False)
                        sync();elapsed=(time.perf_counter()-started)*1000
                        ids=[r['doc_id'] for r in rows]
                        if len(ids)!=config['evaluation']['results_per_query'] or len(set(ids))!=len(ids):
                            raise RuntimeError('Invalid result depth or duplicate documents')
                        key=(method,query['query_id'])
                        if key in canonical and canonical[key]!=ids:
                            raise RuntimeError(f'Rankings changed across repetitions for {key}')
                        canonical[key]=ids
                        if repetition==0:
                            system,variant=SYSTEMS[method]
                            for position,row in enumerate(rows,1):
                                cluster=runtime.cluster_by_doc.get(row['doc_id'])
                                rankings[method].append(dict(engine=system,variant=variant,query_id=query['query_id'],query_text=query['query_text'],rank=position,
                                    doc_id=row['doc_id'],title=row.get('title',''),url=row.get('url',''),technology=row.get('technology',''),categories=json.dumps([str(cluster)]),score=float(row['score'])))
                        samples.append(dict(method=method,query_id=query['query_id'],repetition=repetition+1,total_ms=elapsed,**detail,
                                            rss_bytes=process.memory_info().rss,cuda_allocated_bytes=torch.cuda.memory_allocated() if engine.device=='cuda' else 0))
                    print(f"Pass {repetition+1}/{settings['repetitions']}: {query['query_id']}",flush=True)
        finally:
            runtime.close()
    for method,rows in rankings.items():
        system,variant=SYSTEMS[method]
        write_csv(results/system/f'{variant}.csv',rows,COLUMNS)
    write_csv(results/'latency.csv',samples,list(samples[0]))
    summary=[]
    for method in methods:
        rows=[r for r in samples if r['method']==method];by_query=defaultdict(list)
        for row in rows:by_query[row['query_id']].append(row['total_ms'])
        summary.append(dict(method=method,queries=len(by_query),samples=len(rows),mean_ms=statistics.mean(statistics.mean(v) for v in by_query.values()),
                            median_ms=statistics.median(r['total_ms'] for r in rows),encoding_mean_ms=statistics.mean(r['encoding_ms'] for r in rows),ranking_mean_ms=statistics.mean(r['ranking_ms'] for r in rows)))
    write_csv(results/'latency_summary.csv',summary,list(summary[0]))
    extraction=json.loads((resolve_path(config['paths']['neural_output_dir'])/'step_01_extraction/summary.json').read_text())
    run_id=datetime.now(timezone.utc).isoformat()
    common=dict(run_id=run_id,queries_sha256=query_hash,corpus_manifest_sha256=extraction['dataset_manifest_sha256'],corpus_documents=index.total_docs,
                query_count=len(queries),results_per_query=config['evaluation']['results_per_query'],device=engine.device,
                model_precision='bfloat16' if engine.device=='cuda' else 'float32',vector_precision='float32 from persisted float16',caches=False)
    classical=[dict(common,engine='classical',variant=v,parameters_from_config=config['classical']) for v in ('bm25','bm25f')]
    neural=[]
    for variant in config['evaluation']['neural_variants']:
        neural.append(dict(common,engine='neural',variant=variant['id'],mode=variant['mode'],weights=None if variant['mode']=='dense' else config['neural']['zone_weights'],
                           model_id=config['neural']['model_id'],model_revision=config['neural']['model_revision'],dimensions=engine.encoder.dimension,
                           index_fingerprint=engine.summary['fingerprint'],cluster_fingerprint=engine.clusters.model['training_fingerprint']))
    write_json(results/'classical/manifest.json',classical)
    write_json(results/'neural/manifest.json',neural)
    write_json(results/'latency_manifest.json',dict(common,settings=settings,residency=residency,
        scope='Backend query encoding/tokenization, scoring, top-k selection and in-memory result materialization; excludes index/model loading, UI, document JSON, HTTP and browser',
        scheduling='One process, one CPU thread, one GPU workload; sequential queries and rotated method order'))
    for row in summary:print(f"{row['method']}: {row['mean_ms']:.2f} ms mean",flush=True)
