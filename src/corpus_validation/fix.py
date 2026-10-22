"""Apply accepted corrections to the working corpus, with a recovery backup."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from src.corpus_validation.paths import safe_destination
from src.neural.common import digest_file, records, report, write_json


def _check_file(task):
    root, row = task
    path = safe_destination(root, row['html_path'])
    if path.stat().st_size != row['html_bytes'] or digest_file(path) != row['sha256']:
        raise ValueError(f"Dataset file differs from its manifest: {row['doc_id']}")
    return row['doc_id']


def validate_dataset(corpus_root, workers=8):
    """Check every corrected HTML file and exact manifest/file coverage."""
    root = Path(corpus_root)
    manifest = root / 'metadata/corpus_manifest.jsonl'
    rows = list(records(manifest))
    ids = [r['doc_id'] for r in rows]
    paths = {r['html_path'] for r in rows}
    if len(set(ids)) != len(rows) or len(paths) != len(rows):
        raise ValueError('Duplicate document IDs or paths in dataset')
    actual = {p.relative_to(root).as_posix() for p in (root / 'documents/html').rglob('*.html')}
    if actual != paths:
        raise ValueError(f'Dataset HTML coverage mismatch: missing={len(paths-actual)}, extra={len(actual-paths)}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_check_file, ((root, r) for r in rows)):
            pass
    return {'documents': len(rows), 'all_files_sha256_verified': True,
            'manifest_sha256': digest_file(manifest),
            'categories': dict(Counter(r['category'] for r in rows)),
            'technologies': dict(Counter(r['technology'] for r in rows))}


def fix_dataset(corpus_root, extraction, backup, workers=8):
    root, extraction, backup = map(Path, (corpus_root, extraction, backup))
    manifest = root / 'metadata/corpus_manifest.jsonl'
    state_path = root / 'metadata/corpus_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['manifest_sha256'] != digest_file(manifest):
            raise ValueError('Corrected manifest changed; audit before applying further corrections')
        return state
    backup.mkdir(parents=True, exist_ok=True)
    metadata_backup = backup / 'metadata'
    if not metadata_backup.exists():
        shutil.copytree(root / 'metadata', metadata_backup)
    original = {r['doc_id']: r for r in records(metadata_backup / 'corpus_manifest.jsonl')}
    accepted = list(records(extraction / 'eligible_manifest.jsonl'))
    if not accepted or len({r['doc_id'] for r in accepted}) != len(accepted):
        raise ValueError('Accepted extraction must contain unique documents')
    corrected = []
    repaired = 0
    for row in accepted:
        old = original[row['doc_id']]
        target = safe_destination(root, old['html_path'])
        source = safe_destination(root, row['html_path'])
        if source != target:
            # Check the repaired file before replacing the bad capture.
            _check_file((root, row))
            saved = safe_destination(backup, old['html_path'])
            saved.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest_file(target) != row['sha256']:
                if saved.exists():
                    raise ValueError(f'Conflicting backup: {saved}')
                target.replace(saved)
            shutil.copyfile(source, target)
            repaired += 1
        corrected.append({**row, 'html_path': old['html_path'], 'filename': target.name})
    keep = {r['html_path'] for r in corrected}
    # Actual directory scans and the normal manifest must see exactly the same corpus.
    for path in (root / 'documents/html').rglob('*.html'):
        relative = path.relative_to(root).as_posix()
        if relative not in keep:
            saved = safe_destination(backup, relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                raise ValueError(f'Conflicting backup: {saved}')
            path.replace(saved)
    temp = manifest.with_suffix('.jsonl.part')
    with temp.open('w', encoding='utf-8') as stream:
        for row in corrected:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(manifest)
    # The immutable ZIP and backup retain old manifests; the main folder has one manifest.
    (root / 'metadata/corpus_manifest.original.jsonl').unlink(missing_ok=True)
    stats = validate_dataset(root, workers)
    stats.update(status='corrected', source_documents=len(original),
                 excluded_documents=len(original)-len(corrected), repaired_documents=repaired)
    _update_extraction_metadata(extraction, corrected, stats)
    write_json(state_path, stats)
    report(backup.parent / 'corpus_fix', 'Corrected shared working corpus', stats,
           {'categories': stats['categories'], 'technologies': stats['technologies'],
            'outcome': {'retained': len(corrected), 'excluded': stats['excluded_documents'], 'repaired': repaired}})
    print(f"Corrected dataset itself: {len(corrected)} documents; every HTML hash verified.", flush=True)
    return stats


def _update_extraction_metadata(extraction, corrected, state):
    """Change source locations only; preserve verified text and embedding inputs."""
    from src.neural.common import connect
    rows = {r['doc_id']: r for r in corrected}
    db = connect(extraction / 'corpus.sqlite')
    try:
        if {r[0] for r in db.execute('SELECT doc_id FROM documents')} != set(rows):
            raise ValueError('Corrected dataset and extracted document IDs differ')
        temp = extraction / 'structured_records.jsonl.part'
        with temp.open('w', encoding='utf-8') as stream:
            for row in db.execute('SELECT doc_id,record_json FROM documents ORDER BY doc_index'):
                doc = json.loads(row['record_json'])
                doc['metadata'].update(rows[row['doc_id']])
                encoded = json.dumps(doc, ensure_ascii=False)
                db.execute('UPDATE documents SET record_json=? WHERE doc_id=?', (encoded, row['doc_id']))
                stream.write(encoded + '\n')
        db.commit()
        temp.replace(extraction / 'structured_records.jsonl')
    finally:
        db.close()
    summary_path = extraction / 'summary.json'
    stats = json.loads(summary_path.read_text())
    stats.update(dataset_manifest_sha256=state['manifest_sha256'],
                 extractor_sha256=digest_file(Path(__file__).parents[1] / 'neural/preprocessing/extract.py'))
    stats.pop('allowlist_sha256', None)
    write_json(summary_path, stats)
    # Eligibility decisions remain internal audit history; no companion list is needed.
    (extraction / 'eligible_manifest.jsonl').unlink()
