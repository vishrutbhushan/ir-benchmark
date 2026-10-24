"""Regression checks for query-label consistency and corpus cache validation."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.neural.learning.zone_weights.features import load_examples
from src.neural.pipeline import run


class NeuralAuditTests(unittest.TestCase):
    def test_same_query_id_cannot_hide_different_query_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = [dict(query_id='q',query_text=text,split='train',relevant='1',use_for_weight_learning='1')
                    for text in ('original query', 'different query')]
            with (path/'training.csv').open('w', newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'one query text'):
                load_examples(path)

    def test_build_rejects_changed_html_even_with_corrected_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'corpus';metadata=root/'metadata';metadata.mkdir(parents=True)
            path=root/'documents/html/test/page.html';path.parent.mkdir(parents=True);path.write_bytes(b'evil')
            row=dict(doc_id='one',html_path=path.relative_to(root).as_posix(),html_bytes=4,
                     sha256=hashlib.sha256(b'good').hexdigest(),category='test',technology='test')
            (metadata/'corpus_manifest.jsonl').write_text(json.dumps(row)+'\n')
            (metadata/'corpus_state.json').write_text('{}')
            args=SimpleNamespace(output_dir=base/'outputs',logs_dir=base/'logs',phase='prepare',corpus_root=root,workers=1)
            with self.assertRaisesRegex(ValueError,'differs from its manifest'):
                run(args)


    def test_bounded_mapping_keeps_order_without_python314_keywords(self):
        from src.utils.parallel import bounded_map
        windows=[]
        class Executor:
            def map(self, function, values, chunksize=1):
                values=list(values);windows.append(len(values))
                return map(function, values)
        result=list(bounded_map(Executor(),lambda n:n*n,range(8),3))
        self.assertEqual(result,[n*n for n in range(8)])
        self.assertEqual(windows,[3,3,2])


    def test_valid_range_but_wrong_vector_document_mapping_is_rejected(self):
        import sqlite3
        from src.neural.preprocessing.integrity import audit_vector_mapping
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.execute('CREATE TABLE passages (passage_index INTEGER,doc_index INTEGER)')
        db.executemany('INSERT INTO passages VALUES (?,?)',[(0,0),(1,1)])
        self.assertEqual(audit_vector_mapping(db,[0,1]),2)
        with self.assertRaisesRegex(RuntimeError,'differs from passage'):
            audit_vector_mapping(db,[1,0])
        db.close()
