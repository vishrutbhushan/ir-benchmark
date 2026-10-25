"""Regression checks for query-label consistency and corpus cache validation."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.step_04_weights.features import load_examples
from src.step_01_extraction.step_01_extraction import run_extraction
from src.step_05_classification.labels import load_training_examples


class NeuralAuditTests(unittest.TestCase):
    def test_prepared_zip_import_and_path_traversal_rejection(self):
        from zipfile import ZipFile
        from src.step_01_extraction.step_01_extraction import unpack_corpus

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "corpus.zip"
            with ZipFile(archive, "w") as out:
                out.writestr("technical_web_corpus/metadata/corpus_state.json", "{}")
                out.writestr(
                    "technical_web_corpus/documents/example.html",
                    "<main>Example</main>",
                )
            unpack_corpus(archive, root / "prepared")
            self.assertEqual(
                (root / "prepared/documents/example.html").read_text(),
                "<main>Example</main>",
            )
            with ZipFile(archive, "w") as out:
                out.writestr("../escape.txt", "unexpected")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                unpack_corpus(archive, root / "another")
            self.assertFalse((root / "escape.txt").exists())

    def test_compact_classification_labels_load_full_text_by_id(self):
        from src.common.artifacts import connect

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc = {
                "title": "API",
                "headings": ["Usage"],
                "body_text": "Full body " * 5000,
                "code_text": "if ready:\n    run()",
            }
            db = connect(root / "step_01_extraction/corpus.sqlite")
            db.execute("CREATE TABLE documents (doc_id TEXT,record_json TEXT)")
            db.execute("INSERT INTO documents VALUES (?,?)", ("doc1", json.dumps(doc)))
            db.commit()
            db.close()
            source = root / "labels.csv"
            source.write_text(
                "doc_id,class_id,class_name,split\ndoc1,C01,cloud,train\n",
                encoding="utf-8-sig",
            )
            row = load_training_examples(root, source)[0]
            self.assertEqual(
                row["text"],
                "API\n\nUsage\n\n" + doc["body_text"] + "\n\n" + doc["code_text"],
            )
            self.assertEqual(row["class_id"], "C01")
            source.write_text(
                "doc_id,class_id,class_name,split\nmissing,C01,cloud,train\n"
            )
            with self.assertRaisesRegex(ValueError, "Unknown classification document"):
                load_training_examples(root, source)

    def test_same_query_id_cannot_hide_different_query_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = [
                dict(
                    query_id="q",
                    query_text=text,
                    split="train",
                    relevant="1",
                    use_for_weight_learning="1",
                )
                for text in ("original query", "different query")
            ]
            with (path / "training.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "one query text"):
                load_examples(path)

    def test_build_rejects_changed_html_even_with_corrected_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "corpus"
            metadata = root / "metadata"
            metadata.mkdir(parents=True)
            path = root / "documents/html/test/page.html"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"evil")
            row = dict(
                doc_id="one",
                html_path=path.relative_to(root).as_posix(),
                html_bytes=4,
                sha256=hashlib.sha256(b"good").hexdigest(),
                category="test",
                technology="test",
            )
            (metadata / "corpus_manifest.jsonl").write_text(json.dumps(row) + "\n")
            (metadata / "corpus_state.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                run_extraction(root, base / "outputs/step_01_extraction", 1)

    def test_bounded_mapping_keeps_order_without_python314_keywords(self):
        from src.common.parallel import bounded_map

        windows = []

        class Executor:
            def map(self, function, values, chunksize=1):
                values = list(values)
                windows.append(len(values))
                return map(function, values)

        result = list(bounded_map(Executor(), lambda n: n * n, range(8), 3))
        self.assertEqual(result, [n * n for n in range(8)])
        self.assertEqual(windows, [3, 3, 2])

    def test_valid_range_but_wrong_vector_document_mapping_is_rejected(self):
        import sqlite3
        from src.step_07_verification.integrity import audit_vector_mapping

        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE passages (passage_index INTEGER,doc_index INTEGER)")
        db.executemany("INSERT INTO passages VALUES (?,?)", [(0, 0), (1, 1)])
        self.assertEqual(audit_vector_mapping(db, [0, 1]), 2)
        with self.assertRaisesRegex(RuntimeError, "differs from passage"):
            audit_vector_mapping(db, [1, 0])
        db.close()
