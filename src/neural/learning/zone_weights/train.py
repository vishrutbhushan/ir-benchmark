"""Learn four nonnegative zone weights from binary relevance judgments."""
from collections import defaultdict, Counter
import csv
import json
from pathlib import Path

import numpy as np

from src.neural.common import digest_file, report, write_json
from src.neural.evaluation.metrics import ndcg, average_precision
from src.neural.learning.zone_weights.zones import ZONES
from src.neural.scoring import weighted_zone_scores


def refresh_features(root, engine):
    root = Path(root)
    path = Path('data/neural/zone_weights/training.csv')
    with path.open(newline='', encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    by_id={doc['doc_id']:i for i,doc in enumerate(engine.documents)}
    groups=defaultdict(list)
    for row in rows:
        if row['doc_id'] not in by_id: raise ValueError(f'Labeled document no longer in corpus: {row["doc_id"]}')
        groups[row['query_id']].append(row)
    for qid, group in groups.items():
        if len({(r['query_text'],r['split']) for r in group})!=1:raise ValueError('Inconsistent query/split')
        values,_,_=engine.zone_scores(engine._query(group[0]['query_text']))
        for row in group:
            doc=engine.documents[by_id[row['doc_id']]]
            row.update(doc_title=doc['title'],doc_url=doc['url'],doc_technology=doc['technology'])
            for column,zone in enumerate(ZONES):
                score=values[by_id[row['doc_id']],column]
                row[f'has_{zone}']=str(int(np.isfinite(score)))
                row[f'{zone}_cosine']=f'{float(score):.8f}' if np.isfinite(score) else ''
    temporary=path.with_suffix('.csv.part')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
    report(root/'development','Full-index zone learning features',{'pairs':len(rows),'queries':len(groups),'index_fingerprint':engine.summary['fingerprint'],'training_csv_sha256':digest_file(path)}, {'pairs_by_split':dict(Counter(r['split'] for r in rows))})
    print(f'Refreshed {len(rows)} full-index zone feature rows.',flush=True)
    return rows


def fit_weights(root, engine):
    root=Path(root)
    path=Path('data/neural/zone_weights/training.csv')
    rows=refresh_features(root,engine)
    groups=defaultdict(list)
    for row in rows:groups[row['query_id']].append(row)
    x=np.array([[float(r[f'{z}_cosine']) if r[f'has_{z}']=='1' else -np.inf for z in ZONES] for r in rows])
    y=np.array([int(r['relevant']) for r in rows]);train=np.array([r['split']=='train' for r in rows])
    if not train.any() or train.all():raise ValueError('Need separate training and validation queries')
    best=None
    for a in range(21):
        for b in range(21-a):
            for c in range(21-a-b):
                weights=np.array([a,b,c,20-a-b-c],dtype=np.float64)/20
                prediction,_=weighted_zone_scores(x[train],weights)
                loss=float(np.mean((prediction-y[train])**2))
                key=(round(loss,12),float(np.sum((weights-.25)**2)),tuple(weights))
                if best is None or key<best[0]:best=(key,weights)
    weights=best[1];scores,_=weighted_zone_scores(x,weights)
    evaluations={}
    for split in ('train','validation'):
        losses=(scores[np.array([r['split']==split for r in rows])]-y[np.array([r['split']==split for r in rows])])**2
        ndcgs,aps=[],[]
        for qid,group in groups.items():
            if group[0]['split']!=split:continue
            indices=[i for i,r in enumerate(rows) if r['query_id']==qid]
            ranking=[rows[i]['doc_id'] for i in sorted(indices,key=lambda i:(-scores[i],rows[i]['doc_id']))]
            truth={r['doc_id']:int(r['relevant']) for r in group}
            ndcgs.append(ndcg(ranking,truth));aps.append(average_precision(ranking,truth))
        evaluations[split]={'mse':float(losses.mean()),'judged_pool_ndcg_at_5':float(np.mean(ndcgs)),'judged_pool_map':float(np.mean(aps)),'queries':len(ndcgs)}
    model={'zones':list(ZONES),'weights':weights.tolist(),'index_fingerprint':engine.summary['fingerprint'],
           'training_csv_sha256':digest_file(path),'objective':'training-only squared error to binary relevance; deterministic simplex grid step 0.05',
           'missing_zone_policy':'renormalize over present zones; equal-present fallback when all present weights are zero',
           'evaluation':evaluations,'judgment_scope':'Small judged development pools, not final retrieval evaluation'}
    write_json(root/'weights/model.json',model)
    report(root/'weights','Learned four-zone weights',model,{'weights':dict(zip(ZONES,weights.tolist())),'validation':{k:v for k,v in evaluations['validation'].items() if k!='queries'}})
    engine.weights=weights.astype(np.float32)
    print(json.dumps(model,indent=2),flush=True)
    return model
