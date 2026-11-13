"""Load the six retrieval engines and return ranked documents without UI work."""

from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

from config import ROOT, load_pipeline_config, resolve_path
from neural.src.common.gpu_lock import gpu_job_lock


@contextmanager
def source_package(directory):
    saved = {k:v for k,v in sys.modules.items() if k=='src' or k.startswith('src.')}
    for name in saved: del sys.modules[name]
    sys.path.insert(0,str(directory))
    try:
        yield
    finally:
        sys.path.remove(str(directory))
        for name in list(sys.modules):
            if name=='src' or name.startswith('src.'): del sys.modules[name]
        sys.modules.update(saved)


class RetrievalRuntime:
    def __init__(self, config_path=None):
        self.config_path=Path(config_path or os.environ.get('IR_CONFIG', ROOT/'config/pipeline.json')).resolve()
        os.environ['IR_CONFIG']=str(self.config_path)
        os.environ['NEURAL_CONFIG']=str(self.config_path)
        self.config=load_pipeline_config(self.config_path)
        for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[key]=str(self.config["execution"]["cpu_threads"])
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        self.classical={}
        self.neural=None
        self.cluster_by_doc={}

    def load_classical(self):
        if self.classical: return
        with source_package(ROOT/'classical'):
            from src.retrieval.inverted_index import InvertedIndex
            from src.retrieval.bm25_scorer import BM25Scorer
            from src.retrieval.bm25f_scorer import BM25FScorer
            from src.retrieval.query_processor import QueryProcessor
            index=InvertedIndex.load(resolve_path(self.config['paths']['classical_output_dir'])/'stage_03_retrieval/inverted_index.json')
            c=self.config['classical']
            self.classical={
                'bm25':QueryProcessor(index,BM25Scorer(index,k1=c['k1'],b=c['b'])),
                'bm25f':QueryProcessor(index,BM25FScorer(index,k1=c['k1'],zone_weights=c['zone_weights'],zone_b=c['zone_b'],hierarchical_boost=c['hierarchical_boost']),routing_mode=c['routing_mode']),
            }
            for engine in self.classical.values(): engine.logger.disabled=True

    def load_neural(self):
        if self.neural is not None: return
        with source_package(ROOT/'neural'):
            from src.retrieval.step_08_search import SearchEngine
            p=self.config['paths']
            self.neural=SearchEngine(resolve_path(p['neural_output_dir']),self.config['neural']['device'],resolve_path(p['corpus_dir']),corpus_zip=resolve_path(p['filtered_archive']))
        self.cluster_by_doc={d['doc_id']:int(self.neural.clusters.labels[i]) for i,d in enumerate(self.neural.documents)}

    def load(self):
        print('Loading the complete classical index into RAM...',flush=True)
        self.load_classical()
        print('Loading the neural model and complete vector index...',flush=True)
        self.load_neural()

    def clear_caches(self):
        for engine in self.classical.values(): engine.cache.clear()
        if self.neural is not None:
            self.neural.cache.clear()
            self.neural.query_cache.clear()

    def retrieve(self, method, query, limit=None, use_cache=True):
        """Execute backend query processing and ranking using already loaded engines."""
        limit=limit or self.config['neural']['result_limit']
        if method in ('bm25','bm25f'):
            if not self.classical: raise RuntimeError('Load the classical index first')
            return self.classical[method].search(query,top_k=limit,use_cache=use_cache)['results']
        if self.neural is None: raise RuntimeError('Load the neural index first')
        if method=='hybrid-rrf':
            rrf=self.config['evaluation']['rrf']
            first=self.retrieve('bm25f',query,rrf['bm25f_depth'],use_cache)
            second=self.neural.search(query,mode='routed-cluster',page_size=rrf['routed_depth'],use_cache=use_cache)['results']
            scores={};documents={}
            for rows,weight in ((first,rrf['bm25f_weight']),(second,rrf['routed_weight'])):
                for rank,row in enumerate(rows,1):
                    did=row['doc_id'];scores[did]=scores.get(did,0)+weight/(rrf['k']+rank)
                    documents.setdefault(did,row)
            order=sorted(scores,key=lambda did:(-scores[did],did))[:limit]
            return [{**documents[did],'rank':rank,'score':scores[did]} for rank,did in enumerate(order,1)]
        modes={'neural-dense':'dense','neural-zones':'zones','neural-routed':'routed-cluster'}
        if method not in modes: raise ValueError('Unknown retrieval engine')
        return self.neural.search(query,mode=modes[method],page_size=limit,use_cache=use_cache)['results']

    def search(self, method, query, limit=None):
        started=time.perf_counter()
        if method in ('bm25','bm25f'):
            rows=self.retrieve(method,query,limit)
        else:
            with gpu_job_lock(): rows=self.retrieve(method,query,limit)
        results=[]
        for rank,row in enumerate(rows,1):
            did=row['doc_id'];cluster=self.cluster_by_doc.get(did)
            results.append(dict(rank=rank,doc_id=did,title=row.get('title',''),technology=row.get('technology',''),
                                category=f'Cluster {cluster}' if cluster is not None else row.get('category',''),
                                score=round(float(row['score']),6),url=row.get('url',''),saved_html=f'/document/{did}'))
        names={x['id']:x['name'] for x in self.config['evaluation']['search_engines']}
        return dict(method=method,method_name=names[method],query=query,results=results,elapsed_ms=round((time.perf_counter()-started)*1000,2))

    def close(self):
        if self.neural is not None: self.neural.close()
        self.classical.clear()
