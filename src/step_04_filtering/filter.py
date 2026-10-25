"""Whole-corpus suitability decisions and persistent independent extraction."""
from __future__ import annotations
from src.common.parallel import bounded_map
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from src.common.artifacts import connect, digest_file, records, report
from src.step_04_filtering.extract import extract_document
from src.step_04_filtering.fix import REVIEWED_EXCLUSIONS

POLICY = 'english-main-content-v1; ambiguous-language-quarantined; exact-query-alias-dedup; preserve-version-paths'

def language_eligible(check, record):
    if check['status'] == 'exclude':
        return False, ','.join(check['reasons'])
    if check['status'] == 'eligible':
        return True, 'validated_english'
    lang = check.get('detected_language')
    declared = check.get('html_lang','').split('-')[0]
    # Non-English locale paths are stronger evidence than English API identifiers.
    path = __import__('urllib.parse',fromlist=['urlsplit']).urlsplit(record['url']).path
    if re.search(r'/(?:jp|ja|zh(?:-cn|-tw|-hans|-hant)?|ko|ru|fr|de|es|pt(?:-br)?|it|tr|pl|uk|vi|id)(?:/|$)',path,re.I):
        return False,'non_english_locale_path'
    if declared not in {'','en'}:
        return False,'non_english_html_language'
    if lang == 'en':
        return True,'review_resolved_english_detector'
    if declared == 'en' and lang == 'unknown' and (check.get('code_chars',0)>=60 or check.get('visible_chars',0)>=60):
        return True,'short_english_api_document'
    return False,'ambiguous_language_quarantined'


def _extract(task):
    record, root = task
    try:
        raw = (Path(root)/record['html_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('source_sha256_changed')
        return record,extract_document(raw,record),None
    except Exception as error:
        return record,None,f'{type(error).__name__}: {error}'


def run_extraction(corpus_root, validation, out, repairs, workers=8):
    out.mkdir(parents=True,exist_ok=True)
    source = corpus_root/'metadata/corpus_manifest.jsonl'
    corrected = (corpus_root / 'metadata/corpus_state.json').exists()
    artifacts_complete = all((out / name).is_file() for name in ('corpus.sqlite', 'structured_records.jsonl', 'decisions.jsonl'))
    if corrected and artifacts_complete and (out / 'summary.json').exists():
        saved = json.loads((out / 'summary.json').read_text())
        if (saved.get('dataset_manifest_sha256') == digest_file(source)
                and saved.get('extractor_sha256') == digest_file(Path(__file__).with_name('extract.py'))):
            print('Corrected dataset extraction unchanged: reusing verified text and passage inputs.', flush=True)
            return saved
    fingerprint = hashlib.sha256((digest_file(source)+('' if corrected else digest_file(validation/'document_checks.jsonl'))+digest_file(Path(__file__))+digest_file(Path(__file__).with_name('extract.py'))+json.dumps(repairs,sort_keys=True)+POLICY).encode()).hexdigest()
    state_path = out/'summary.json'
    if artifacts_complete and state_path.exists() and json.loads(state_path.read_text()).get('fingerprint') == fingerprint:
        print('Extraction unchanged: using complete persistent corpus.',flush=True)
        return json.loads(state_path.read_text())
    all_records = sorted(records(source),key=lambda r:r['doc_id'])
    checks = {} if corrected else {c['doc_id']:c for c in records(validation/'document_checks.jsonl')}
    decisions, candidates = [], []
    for record in all_records:
        if record['doc_id'] in REVIEWED_EXCLUSIONS:
            decisions.append({'doc_id':record['doc_id'],'status':'excluded','reason':REVIEWED_EXCLUSIONS[record['doc_id']]})
            continue
        repaired = repairs.get(record['doc_id'],{})
        check = checks[record['doc_id']] if not corrected else {'status': 'eligible'}
        if not corrected and repaired.get('status') == 'repaired':
            record,check = repaired['record'],repaired['evidence']
        allowed,reason = language_eligible(check,record)
        if allowed:
            candidates.append(record)
        else:
            decisions.append({'doc_id':record['doc_id'],'status':'excluded','reason':reason})
    start = time.perf_counter()
    db_temp = out/'corpus.build.sqlite'
    for suffix in ['', '-wal','-shm']:
        Path(str(db_temp)+suffix).unlink(missing_ok=True)
    db = connect(db_temp)
    db.execute('CREATE TABLE documents (doc_index INTEGER PRIMARY KEY, doc_id TEXT UNIQUE NOT NULL, url TEXT NOT NULL, title TEXT NOT NULL, technology TEXT, category TEXT, record_json TEXT NOT NULL)')
    exact_aliases = {}
    technologies, categories, kinds = Counter(),Counter(),Counter()
    code_docs,total_chars,total_blocks = 0,0,0
    exported = out/'structured_records.jsonl.part'
    allowlist = out/'eligible_manifest.jsonl.part'
    with exported.open('w',encoding='utf-8') as stream, allowlist.open('w',encoding='utf-8') as allowed_stream, ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(source_record,document,error) in enumerate(bounded_map(pool, _extract, ((r, str(corpus_root)) for r in candidates), workers*24, chunksize=8),1):
            doc_id = source_record['doc_id']
            if error:
                # Parsing failure is explicit; byte-integrity changes are fatal, never silently dropped.
                if 'source_sha256_changed' in error or 'FileNotFoundError' in error:
                    raise RuntimeError(f'{doc_id}: {error}')
                decisions.append({'doc_id':doc_id,'status':'excluded','reason':'extraction_failed','error':error})
                continue
            size = len(document['body_text'])+len(document['code_text'])
            if size < 60:
                decisions.append({'doc_id':doc_id,'status':'excluded','reason':'empty_after_main_extraction','characters':size})
                continue
            from urllib.parse import urlsplit
            url = urlsplit(document['url'])
            # Only identical text at the same URL path (query aliases). Version paths remain distinct.
            duplicate_key = (source_record['technology'],url.netloc,url.path,document['metadata']['content_sha256'])
            if duplicate_key in exact_aliases:
                decisions.append({'doc_id':doc_id,'status':'excluded','reason':'exact_url_alias','canonical_doc_id':exact_aliases[duplicate_key]})
                continue
            exact_aliases[duplicate_key] = doc_id
            index = sum(categories.values())
            db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?)',(index,doc_id,document['url'],document['title'],source_record['technology'],source_record['category'],json.dumps(document,ensure_ascii=False)))
            stream.write(json.dumps(document,ensure_ascii=False)+'\n')
            allowed_stream.write(json.dumps(source_record,ensure_ascii=False)+'\n')
            decisions.append({'doc_id':doc_id,'status':'retained','reason':'usable_english_main_content','source':'live_repair' if source_record.get('source')=='live_repair' else 'original_capture'})
            technologies[source_record['technology']] += 1
            categories[source_record['category']] += 1
            total_chars += size
            code_docs += bool(document['code_text'])
            for section in document['sections']:
                kinds.update(b['kind'] for b in section['blocks'])
                total_blocks += len(section['blocks'])
            if i % 1000 == 0:
                db.commit(); print(f'Extracted {i}/{len(candidates)} candidates; retained {sum(categories.values())}',flush=True)
    db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    if len(decisions) != len(all_records):
        raise RuntimeError('Every source document must have exactly one final decision')
    if len({d['doc_id'] for d in decisions}) != len(all_records):
        raise RuntimeError('Duplicate document decisions')
    db_temp.replace(out/'corpus.sqlite')
    exported.replace(out/'structured_records.jsonl')
    allowlist.replace(out/'eligible_manifest.jsonl')
    with (out/'decisions.jsonl').open('w',encoding='utf-8') as stream:
        for decision in sorted(decisions,key=lambda d:d['doc_id']):
            stream.write(json.dumps(decision)+'\n')
    stats = {'fingerprint':fingerprint,'policy':POLICY,'eligibility_frozen':True,'source_documents':len(all_records),'retained_documents':sum(categories.values()),'excluded_documents':sum(d['status']=='excluded' for d in decisions),'code_documents':code_docs,'extracted_characters':total_chars,'blocks':total_blocks,'seconds':round(time.perf_counter()-start,2),'dataset_manifest_sha256':digest_file(source),'extractor_sha256':digest_file(Path(__file__).with_name('extract.py'))}
    if corrected:
        (out / 'eligible_manifest.jsonl').unlink(missing_ok=True)
    report(out,'Shared corpus filtering',stats,{'categories':dict(categories),'technologies':dict(technologies),'block_types':dict(kinds),'exclusions':dict(Counter(d['reason'] for d in decisions if d['status']=='excluded'))})
    return stats
