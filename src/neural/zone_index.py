"""Four independent durable zone indexes, using one encoder and fixed windows."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import hashlib
import json
from pathlib import Path

from src.neural.common import connect, digest_file, report
from src.neural.learning.zone_weights.zones import ZONES, document_zones
from src.neural.model import tokenizer, download_model
from src.neural.preprocessing.chunk import passage_parts
from src.utils.parallel import bounded_map

_TOKENIZER = None


def _initialize(model):
    global _TOKENIZER
    _TOKENIZER = tokenizer(model)
    _TOKENIZER.model_max_length = 10**12


def _document_windows(task):
    index, encoded = task
    doc = json.loads(encoded)
    result = {}
    for zone, text in document_zones(doc).items():
        pieces = []
        for piece, start, end in passage_parts(text, _TOKENIZER, 496, 64):
            tokens = len(_TOKENIZER.encode(piece, add_special_tokens=True))
            if tokens > 512:
                raise ValueError(f'{doc["doc_id"]}/{zone}: window exceeds token limit')
            pid = hashlib.sha256(f'{doc["doc_id"]}:{zone}:{start}:{end}'.encode()).hexdigest()[:24]
            pieces.append((pid, index, zone, piece, piece, tokens, start, end))
        result[zone] = pieces
    return result


def prepare_zones(root, workers=8):
    root = Path(root)
    extraction = json.loads((root/'extraction/summary.json').read_text())
    fingerprint = hashlib.sha256((extraction['fingerprint']+digest_file(__file__)+digest_file(Path(__file__).parent/'learning/zone_weights/zones.py')+'496:64:pure-zones-v1').encode()).hexdigest()
    summary = root/'zones/summary.json'
    if summary.exists() and json.loads(summary.read_text()).get('fingerprint') == fingerprint and all((root/f'zones/{z}/passages/passages.sqlite').exists() for z in ZONES):
        print('Four-zone windows unchanged: reusing the complete stores.', flush=True)
        return json.loads(summary.read_text())
    download_model(root/'model')
    databases, counts, coverage = {}, Counter(), Counter()
    previous, hashes = {}, {z:hashlib.sha256() for z in ZONES}
    def update_hash(digest, values):
        digest.update(json.dumps(list(values),ensure_ascii=False,separators=(',',':')).encode())
        digest.update(b'\n')
    for zone in ZONES:
        directory=root/f'zones/{zone}/passages'
        if (directory/'passages.sqlite').exists() and (directory/'summary.json').exists():
            saved=json.loads((directory/'summary.json').read_text())
            digest=hashlib.sha256()
            with closing(connect(directory/'passages.sqlite',True)) as store:
                for row in store.execute('SELECT doc_index,embedding_text,tokens,char_start,char_end FROM passages ORDER BY passage_index'):
                    update_hash(digest,row)
            previous[zone]=(digest.hexdigest(),saved)
    try:
        for zone in ZONES:
            directory = root/f'zones/{zone}/passages'
            directory.mkdir(parents=True, exist_ok=True)
            temp = directory/'passages.build.sqlite'
            for suffix in ('', '-wal', '-shm'):
                Path(str(temp)+suffix).unlink(missing_ok=True)
            db = connect(temp)
            db.execute('CREATE TABLE passages (passage_index INTEGER PRIMARY KEY,passage_id TEXT UNIQUE,doc_index INTEGER,zone TEXT,text TEXT,embedding_text TEXT,tokens INTEGER,char_start INTEGER,char_end INTEGER)')
            databases[zone] = db
        with closing(connect(root/'extraction/corpus.sqlite', True)) as db, ProcessPoolExecutor(max_workers=workers, initializer=_initialize, initargs=(str(root/'model'),)) as pool:
            tasks = ((r['doc_index'], r['record_json']) for r in db.execute('SELECT doc_index,record_json FROM documents ORDER BY doc_index'))
            for number, windows in enumerate(bounded_map(pool, _document_windows, tasks, workers*2), 1):
                for zone, pieces in windows.items():
                    if pieces:
                        coverage[zone] += 1
                    for piece in pieces:
                        databases[zone].execute('INSERT INTO passages VALUES (?,?,?,?,?,?,?,?,?)', (counts[zone], *piece))
                        update_hash(hashes[zone], (piece[1],piece[4],piece[5],piece[6],piece[7]))
                        counts[zone] += 1
                if number % 1000 == 0:
                    for store in databases.values(): store.commit()
                    print(f'Prepared four zones: {number}/{extraction["retained_documents"]} documents; {dict(counts)}', flush=True)
        for zone, db in databases.items():
            db.execute('CREATE INDEX by_document ON passages(doc_index)')
            db.commit(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
            directory = root/f'zones/{zone}/passages'
            (directory/'passages.build.sqlite').replace(directory/'passages.sqlite')
            content_digest=hashes[zone].hexdigest()
            identical=zone in previous and previous[zone][0]==content_digest
            zone_fingerprint=previous[zone][1]['fingerprint'] if identical else hashlib.sha256((content_digest+zone).encode()).hexdigest()
            print(f'{zone}: embedding inputs {"unchanged; reusing" if identical else "changed; embedding"}.',flush=True)
            stats = {'content_sha256':content_digest,'fingerprint':zone_fingerprint, 'documents':extraction['retained_documents'], 'passages':counts[zone], 'zone':zone, 'input_fingerprint':extraction['fingerprint'], 'documents_present':coverage[zone]}
            report(directory, f'{zone.title()} zone windows', stats, {'coverage':{'present':coverage[zone],'absent':extraction['retained_documents']-coverage[zone]}})
    finally:
        for db in databases.values(): db.close()
    stats = {'fingerprint':fingerprint, 'documents':extraction['retained_documents'], 'windows':dict(counts), 'zone_coverage':dict(coverage), 'zones':list(ZONES), 'content_tokens':496, 'overlap':64, 'source_fingerprint':extraction['fingerprint']}
    report(root/'zones', 'Complete fixed-zone index preparation', stats, {'windows':dict(counts),'coverage':dict(coverage)})
    return stats


def embed_zones(root, device='cuda', batch_size=512):
    from src.neural.embedding import run_embeddings
    root = Path(root)
    for zone in ZONES:
        print(f'Embedding complete {zone} zone...', flush=True)
        run_embeddings(root/f'zones/{zone}/passages', root/f'zones/{zone}/embeddings', root/'model', device, batch_size)
