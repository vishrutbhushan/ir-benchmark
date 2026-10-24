"""Full-ranking neural evaluation and a shared classic-ranking import contract."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from src.neural.common import digest_file,report
from src.neural.evaluation.metrics import ndcg,average_precision
from src.neural.search import MODES


def evaluate(root, engine, classic_rankings=None):
    root=Path(root);source=Path('data/evaluation/qrels.csv')
    if not source.exists():raise ValueError('Create the judged query/document pool before evaluation')
    with source.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    truth=defaultdict(dict);notes={}
    for row in rows:
        key=(row['query_id'],row['doc_id'])
        if row['doc_id'] in truth[row['query_id']]:raise ValueError('Duplicate qrel')
        grade=int(row['grade'])
        if grade not in {0,1,2}:raise ValueError('Qrel grades must be 0, 1 or 2')
        truth[row['query_id']][row['doc_id']]=grade;notes[key]=row.get('reason','')
    queries=[json.loads(line) for line in Path('queries.jsonl').read_text().splitlines() if line.strip()]
    query_text={q['query_id']:q['query_text'] for q in queries}
    if any(row.get('query_text') and row['query_text']!=query_text.get(row['query_id']) for row in rows):
        raise ValueError('Qrel query text differs from the frozen query suite')
    if set(truth)!={q['query_id'] for q in queries}:raise ValueError('Every final query needs judgments')
    valid_ids={d['doc_id'] for d in engine.documents}
    if any(doc not in valid_ids for labels in truth.values() for doc in labels):raise ValueError('Qrel document outside shared corpus')
    out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    metrics=[];failure_rows=[]
    query_sha=digest_file('queries.jsonl')
    def score(query,mode,ranking):
        qid=query['query_id'];labels=truth[qid]
        if not any(g>0 for g in labels.values()):raise ValueError(f'{qid}: no judged relevant document')
        if len(ranking)!=len(set(ranking)) or any(doc not in valid_ids for doc in ranking):raise ValueError('Invalid ranked document IDs')
        metric={'query_id':qid,'category':query['category'],'mode':mode,'ndcg_at_5':ndcg(ranking,labels,5),'average_precision':average_precision(ranking,labels),
                'ranked_documents':len(ranking),'judged_documents':len(labels),'judged_relevant':sum(g>0 for g in labels.values()),
                'top5_judged_fraction':sum(doc in labels for doc in ranking[:5])/max(1,len(ranking[:5]))}
        metrics.append(metric)
        wrong=next((doc for doc in ranking[:5] if labels.get(doc)==0),None)
        missed=next((doc for doc,grade in labels.items() if grade>0 and doc not in ranking[:5]),None)
        failure_rows.append({'query_id':qid,'category':query['category'],'mode':mode,'top_nonrelevant_doc':wrong or '',
                             'nonrelevant_reason':notes.get((qid,wrong),''),'relevant_outside_top5':missed or '',
                             'relevance_reason':notes.get((qid,missed),'')})
    with gzip.open(out/'rankings.jsonl.gz','wt',encoding='utf-8') as stream:
        for mode in MODES:
            for query in queries:
                ranked=engine.rank(query['query_text'],mode)
                ids=[engine.documents[int(i)]['doc_id'] for i in ranked['order']]
                score(query,mode,ids)
                stream.write(json.dumps({'query_id':query['query_id'],'mode':mode,'query_suite_sha256':query_sha,
                                         'index_fingerprint':engine.summary['fingerprint'],'doc_ids':ids})+'\n')
    if classic_rankings:
        with Path(classic_rankings).open() as stream:classic=[json.loads(line) for line in stream if line.strip()]
        if {r['query_id'] for r in classic}!={q['query_id'] for q in queries} or len(classic)!=len(queries):raise ValueError('Classic rankings must contain every query exactly once')
        by_query={q['query_id']:q for q in queries}
        for row in classic:
            if row.get('query_suite_sha256')!=query_sha:raise ValueError('Classic query-suite hash mismatch')
            ranking=row['doc_ids']
            if any(doc not in truth[row['query_id']] for doc in ranking[:20]):
                raise ValueError('Extend the common judged pool with classic top-20 results before comparison')
            score(by_query[row['query_id']],'classic',ranking)
    for filename,items in [('per_query.csv',metrics),('failure_cases.csv',failure_rows)]:
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    results={}
    for mode in sorted({r['mode'] for r in metrics}):
        selected=[r for r in metrics if r['mode']==mode]
        results[mode]={'ndcg_at_5':float(np.mean([r['ndcg_at_5'] for r in selected])),
                       'map':float(np.mean([r['average_precision'] for r in selected])),
                       'mean_top5_judged_fraction':float(np.mean([r['top5_judged_fraction'] for r in selected]))}
    categories={}
    for category in sorted({r['category'] for r in metrics}):
        categories[category]={}
        for mode in results:
            selected=[r for r in metrics if r['mode']==mode and r['category']==category]
            categories[category][mode]={name:float(np.mean([r[key] for r in selected])) for name,key in [('ndcg_at_5','ndcg_at_5'),('map','average_precision')]}
    stats={'queries':len(queries),'qrels':len(rows),'results':results,'by_category':categories,
           'query_suite_sha256':query_sha,'qrels_sha256':digest_file(source),
           'metric_scope':'Complete neural document rankings against pooled graded qrels; unjudged documents receive zero gain. Pool incompleteness limits MAP interpretation.',
           'classic_status':'Imported and compared' if classic_rankings else 'Awaiting teammate classic rankings; no classic score is invented',
           'classic_import_contract':{'query_id':'Q01','query_suite_sha256':query_sha,'doc_ids':['ordered original document IDs; top20 must be judged; full rankings for full MAP']}}
    report(out,'Neural retrieval evaluation',stats,{'ndcg_at_5':{m:r['ndcg_at_5'] for m,r in results.items()},'pooled_qrels_map':{m:r['map'] for m,r in results.items()}})
    print(json.dumps(results,indent=2),flush=True)
    return stats
