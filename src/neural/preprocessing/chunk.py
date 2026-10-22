"""Token-counted passages with section provenance; no stemming/stopword removal."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import time
from src.neural.common import MAX_TOKENS, OVERLAP, MODEL_REVISION, connect, digest_file, report
from src.neural.model import tokenizer


def passage_parts(text,tok,budget,overlap=OVERLAP):
    encoded = tok(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    if not offsets:
        return
    start = 0
    while start < len(offsets):
        end = min(start+budget,len(offsets))
        char_start = 0 if start==0 else offsets[start][0]
        char_end = len(text) if end==len(offsets) else offsets[end][0]
        # Tokenizers may map multiple byte tokens to the same Unicode character.
        while end < len(offsets) and offsets[end][0] <= char_start:
            end += 1
            char_end = len(text) if end==len(offsets) else offsets[end][0]
        if char_end > char_start:
            yield text[char_start:char_end],char_start,char_end
        if end == len(offsets):
            break
        start = max(start+1,end-min(overlap,budget//4))


_WORKER_TOKENIZER=None

def _initialize_tokenizer(model_path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER=tokenizer(model_path)
    # Long SOURCE sections are intentionally split; they are never sent whole to the model.
    _WORKER_TOKENIZER.model_max_length=10**12

def document_passages(task):
    doc_index,record_json=task
    tok=_WORKER_TOKENIZER
    document=json.loads(record_json)
    result=[]
    for section_index,section in enumerate(document['sections']):
        # Keep related prose, examples, and tables together; paragraph boundaries remain visible.
        text = '\n\n'.join(b['text'] for b in section['blocks'])
        heading = ' > '.join(section['heading_path'])
        context = f'Technology: {document["metadata"]["technology"]}\nTitle: {document["title"]}\nSection: {heading}'
        ids = tok.encode(context,add_special_tokens=False)
        context = tok.decode(ids[:96]) if len(ids)>96 else context
        prefix = context+'\n\n'
        budget = MAX_TOKENS-len(tok.encode(prefix,add_special_tokens=True))-8
        kind = '+'.join(sorted({b['kind'] for b in section['blocks']}))
        for content,start,end in passage_parts(text,tok,budget):
            embedding_text = prefix+content
            token_count = len(tok.encode(embedding_text,add_special_tokens=True))
            if token_count > MAX_TOKENS:
                # Context/text boundary can change tokenization. Split conservatively, never clip.
                pieces = list(passage_parts(content,tok,budget-16))
            else:
                pieces = [(content,0,len(content))]
            for piece,local_start,local_end in pieces:
                combined = prefix+piece
                token_count = len(tok.encode(combined,add_special_tokens=True))
                if token_count > MAX_TOKENS:
                    raise RuntimeError('Chunk token budget invariant failed')
                pid = hashlib.sha256(f'{document["doc_id"]}:{section_index}:{start+local_start}:{start+local_end}:{piece}'.encode()).hexdigest()[:24]
                result.append((pid,doc_index,section_index,heading,section['anchor'],kind,piece,combined,token_count,start+local_start,start+local_end))

    return doc_index,result


def run_chunking(extracted,out,model_path,workers=8):
    out.mkdir(parents=True,exist_ok=True)
    source_summary = json.loads((extracted/'summary.json').read_text())
    fingerprint = hashlib.sha256((source_summary['fingerprint']+MODEL_REVISION+str(MAX_TOKENS)+str(OVERLAP)+digest_file(__file__)).encode()).hexdigest()
    if (out/'summary.json').exists() and json.loads((out/'summary.json').read_text()).get('fingerprint')==fingerprint:
        print('Passages unchanged: using complete persistent passage store.',flush=True)
        return json.loads((out/'summary.json').read_text())
    source_db = connect(extracted/'corpus.sqlite',True)
    temp = out/'passages.build.sqlite'
    for suffix in ['', '-wal','-shm']:
        __import__('pathlib').Path(str(temp)+suffix).unlink(missing_ok=True)
    db = connect(temp)
    db.execute('CREATE TABLE passages (passage_index INTEGER PRIMARY KEY, passage_id TEXT UNIQUE, doc_index INTEGER NOT NULL, section_index INTEGER, heading TEXT, anchor TEXT, kind TEXT, text TEXT, embedding_text TEXT, tokens INTEGER, char_start INTEGER, char_end INTEGER)')
    counts, lengths = Counter(),Counter()
    n, started = 0,time.perf_counter()
    total = source_summary['retained_documents']
    from concurrent.futures import ProcessPoolExecutor
    tasks=((row['doc_index'],row['record_json']) for row in source_db.execute('SELECT doc_index,record_json FROM documents ORDER BY doc_index'))
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize_tokenizer,initargs=(str(model_path),)) as pool:
        for doc_index,rows in pool.map(document_passages,tasks,chunksize=1,buffersize=workers*2):
            for values in rows:
                db.execute('INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',(n,*values))
                n+=1;counts[values[5]]+=1;lengths[str((values[8]//64)*64)]+=1
            if (doc_index+1)%1000==0:
                db.commit();print(f'Chunked {doc_index+1}/{total} documents: {n} passages',flush=True)
    db.execute('CREATE INDEX passages_by_doc ON passages(doc_index)')
    represented = db.execute('SELECT count(DISTINCT doc_index) FROM passages').fetchone()[0]
    if represented != total:
        raise RuntimeError(f'Passage coverage {represented} != retained documents {total}')
    db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close(); source_db.close()
    temp.replace(out/'passages.sqlite')
    stats = {'fingerprint':fingerprint,'input_fingerprint':source_summary['fingerprint'],'documents':total,'passages':n,'max_total_tokens':MAX_TOKENS,'overlap_tokens':OVERLAP,'context_max_tokens':96,'dimension':1024,'fp16_vectors_gib':round(n*1024*2/2**30,3),'seconds':round(time.perf_counter()-started,2),'model_revision':MODEL_REVISION}
    report(out,'Full neural passage preparation',stats,{'passage_types':dict(counts),'token_buckets':dict(sorted(lengths.items(),key=lambda item:int(item[0])))})
    return stats
