"""Choose class-pruning depth against exhaustive validation-query retrieval."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.neural.common import report,write_json,digest_file


def tune_routing(root, engine):
    with Path('data/neural/zone_weights/training.csv').open() as stream:rows=list(csv.DictReader(stream))
    queries={r['query_id']:r['query_text'] for r in rows if r['split']=='validation'}
    if not queries:raise ValueError('Routing requires held-out development queries')
    classes=len(engine.classifier.centroids)
    measurements={method:{k:[] for k in range(1,classes+1)} for method in ('nb','centroid')}
    candidate_fractions={method:{k:[] for k in range(1,classes+1)} for method in measurements}
    for qid,query in queries.items():
        reference=engine.rank(query,'zones')['order'][:20]
        vector=engine._query(query)
        for method in measurements:
            order=np.argsort(-engine.classifier.query_scores(query,vector,method),kind='stable')
            for k in range(1,classes+1):
                allowed=np.isin(engine.classifier.document_classes,order[:k])
                measurements[method][k].append(float(allowed[reference].mean()))
                candidate_fractions[method][k].append(float(allowed.mean()))
    chosen={}
    for method in measurements:
        chosen[method]=next(k for k in range(1,classes+1) if np.mean(measurements[method][k])>=.98 and min(measurements[method][k])>=.9)
    model={'weights_sha256':digest_file(Path(root)/'weights/model.json'),'classifier_csv_sha256':engine.classifier.model['training_csv_sha256'],'index_fingerprint':engine.summary['fingerprint'],'top_classes':chosen,
           'validation_queries':list(queries),'selection_rule':'Smallest top-k classes with mean exact-zone top-20 recall >=0.98 and each query >=0.90',
           'recall':{m:{str(k):float(np.mean(v)) for k,v in measurements[m].items()} for m in measurements},
           'candidate_fraction':{m:{str(k):float(np.mean(v)) for k,v in candidate_fractions[m].items()} for m in measurements},
           'meaning':'Recall of exhaustive weighted-zone top-20, not relevance Recall@20; validation is not a guarantee on new queries'}
    write_json(Path(root)/'routing/model.json',model)
    engine.routing=model
    latencies={}
    # Record actual end-to-end scans after choosing k, separately from membership recall.
    for mode in ('zones','routed-nb','routed-centroid'):
        values=[]
        for query in queries.values():
            start=time.perf_counter();engine.rank(query,mode);values.append((time.perf_counter()-start)*1000)
        latencies[mode]=float(np.median(values))
    model['warm_query_p50_ms']=latencies
    write_json(Path(root)/'routing/model.json',model)
    report(Path(root)/'routing','Validated category routing',model,{'selected_top_classes':chosen,'warm_query_p50_ms':latencies,
           'selected_candidate_fraction':{m:model['candidate_fraction'][m][str(k)] for m,k in chosen.items()}})
    print(json.dumps({'top_classes':chosen,'latency_ms':latencies}),flush=True)
    return model
