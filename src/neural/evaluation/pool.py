"""Export source-backed candidate pools for query/document relevance judgments."""
import csv
import json
from pathlib import Path
import re

from src.neural.common import digest_file,write_json
from src.neural.search import MODES


def make_pool(root, engine):
    root=Path(root);queries=[json.loads(line) for line in Path('queries.jsonl').read_text().splitlines() if line.strip()]
    all_rows=[]
    for query in queries:
        candidates={}
        for mode in MODES:
            ranked=engine.rank(query['query_text'],mode)
            depth=10 if mode in {'dense','zones'} else 5
            for index in ranked['order'][:depth]:candidates.setdefault(int(index),[]).append(mode)
        # Reviewed development documents may not be assumed relevant to a final query.
        for index,modes in sorted(candidates.items(),key=lambda item:engine.documents[item[0]]['doc_id']):
            doc=engine.documents[index]
            record=json.loads(engine.document_db.execute('SELECT record_json FROM documents WHERE doc_id=?',(doc['doc_id'],)).fetchone()[0])
            text=record['body_text']+'\n'+record['code_text']
            terms=[t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_-]*',query['query_text']) if len(t)>3 and t.lower() not in {'with','from','that','does','when','using','compared','without','rather','than'}]
            locations=[]
            for term in terms:
                match=re.search(re.escape(term),text,re.I)
                if match:locations.append((match.start(),term))
            excerpts=[]
            for offset,term in sorted(locations):
                piece=text[max(0,offset-130):offset+650].strip()
                if piece and piece not in excerpts:excerpts.append(piece)
            all_rows.append({'query_id':query['query_id'],'query_text':query['query_text'],'category':query['category'],
                'doc_id':doc['doc_id'],'title':doc['title'],'url':doc['url'],'candidate_modes':'|'.join(modes),
                'source_characters':len(text),'matched_query_terms':'|'.join(t for _,t in locations),
                'source_excerpt':'\n[...]\n'.join(excerpts[:6]) or text[:1000],
                'judgment_intent':query['judgment_intent']})
    out=root/'evaluation';out.mkdir(parents=True,exist_ok=True)
    with (out/'candidate_pool.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    write_json(out/'pool_manifest.json',{'query_suite_sha256':digest_file('queries.jsonl'),'index_fingerprint':engine.summary['fingerprint'],'pairs':len(all_rows),'scope':'Union of dense/weighted top10 and both routed top5; grades must be assigned from source evidence, never from rank'})
    print(f'Saved {len(all_rows)} source-backed candidates.',flush=True)
    return all_rows
