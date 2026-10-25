"""Correctness gates for extraction, chunk coverage, and document ranking."""

import unittest
import numpy as np
from src.step_01_extraction.extract_html import extract_document, dates_from
from src.step_02_windows.windows import passage_parts
from src.step_08_search.step_08_search import aggregate_documents

RECORD = {
    "doc_id": "fixture",
    "url": "https://example.com/v3.11/reference",
    "title": "Fallback",
    "technology": "Python",
    "category": "programming-language",
}


class NeuralTests(unittest.TestCase):
    def test_main_content_code_whitespace_and_no_legacy_section_cap(self):
        prose = "Never remove the word not. " * 160
        raw = f'<html lang="en"><title>API</title><nav><h1>Navigation</h1>Trash</nav><main><h1>Function</h1><p>{prose}</p><pre><span>return</span> <span>value</span>\n    next_line()</pre></main></html>'.encode()
        result = extract_document(raw, RECORD)
        self.assertNotIn("Navigation", result["headings"])
        self.assertEqual(result["code_text"], "return value\n    next_line()")
        self.assertEqual(result["body_text"], prose.strip())
        self.assertGreater(len(result["sections"][0]["blocks"][0]["text"]), 2000)
        self.assertNotIn("return value", result["body_text"])

    def test_nested_lists_are_not_duplicated_and_code_is_typed(self):
        raw = b"<title>Instructions</title><main><h1>Example</h1><ul><li>alpha<ul><li>beta</li></ul></li><li><p>Run the following example safely</p><pre>x = 2\n  y = 3</pre></li></ul></main>"
        result = extract_document(raw, RECORD)
        self.assertEqual(result["body_text"].count("beta"), 1)
        self.assertIn("x = 2\n  y = 3", result["code_text"])

    def test_rfc_preformatted_content_is_prose(self):
        raw = b"<title>RFC</title><pre>Internet standard protocol\n\n  1. Introduction\nDo not discard the original RFC prose.</pre>"
        result = extract_document(raw, {**RECORD, "technology": "RFCs"})
        self.assertIn("Internet standard", result["body_text"])
        self.assertFalse(result["code_text"])

    def test_dates_are_valid_mentions_not_crawl_dates(self):
        self.assertEqual(
            [d["value"] for d in dates_from("2024-02-29 2023-02-29 2024-02-29")],
            ["2024-02-29"],
        )

    def test_chunk_character_coverage_with_real_tokenizer(self):
        from pathlib import Path

        if not (Path("outputs/step_03_embeddings/model/tokenizer.json")).exists():
            self.skipTest("Pinned tokenizer not downloaded")
        from src.common.encoder import tokenizer

        tok = tokenizer("outputs/step_03_embeddings/model")
        text = (
            "Do NOT remove C++, react-router-dom, Python 3.11, café, or 🚀.\n    return value\n"
            * 80
        )
        chunks = list(passage_parts(text, tok, 96, 16))
        covered = np.zeros(len(text), dtype=bool)
        for piece, start, end in chunks:
            self.assertEqual(piece, text[start:end])
            covered[start:end] = True
            self.assertLessEqual(len(tok.encode(piece, add_special_tokens=False)), 100)
        self.assertTrue(covered.all())
        self.assertGreater(len(chunks), 1)

    def test_document_aggregation_all_passages_negative_scores_and_ties(self):
        maxima, winners = aggregate_documents([-0.4, -0.2, -0.1, -0.2], [0, 1, 0, 1], 3)
        np.testing.assert_allclose(maxima[:2], [-0.1, -0.2])
        self.assertTrue(np.isneginf(maxima[2]))
        self.assertEqual(winners.tolist(), [2, 1, 4])


class DurableIndexTests(unittest.TestCase):
    def test_embedding_checkpoint_resumes_completed_windows(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import torch
        from src.common.artifacts import connect, write_json
        from src.step_03_embeddings.step_03_embeddings import run_embeddings

        calls = []

        class FakeEncoder:
            fail = True

            def __init__(self, *args):
                self.torch = torch
                self.dimension = 8

            def encode(self, texts):
                calls.extend(map(int, texts))
                if self.fail and min(map(int, texts)) >= 2048:
                    raise RuntimeError("simulated process interruption")
                result = np.zeros((len(texts), 8), dtype=np.float32)
                for row, text in enumerate(texts):
                    result[row, int(text) % 8] = 1
                return result

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "passages"
            source.mkdir()
            write_json(
                source / "summary.json",
                {"fingerprint": "test-input", "passages": 2052, "documents": 2},
            )
            db = connect(source / "passages.sqlite")
            db.execute(
                "CREATE TABLE passages(passage_index INTEGER PRIMARY KEY,doc_index INTEGER,embedding_text TEXT,tokens INTEGER)"
            )
            db.executemany(
                "INSERT INTO passages VALUES (?,?,?,?)",
                ((i, i % 2, str(i), 1) for i in range(2052)),
            )
            db.commit()
            db.close()
            with (
                patch("src.step_03_embeddings.step_03_embeddings.Encoder", FakeEncoder),
                patch("src.step_03_embeddings.step_03_embeddings.report"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "simulated process interruption"
                ):
                    run_embeddings(
                        source,
                        root / "embeddings",
                        root / "step_03_embeddings/model",
                        "cpu",
                        512,
                    )
                checkpoint = json.loads(
                    (root / "embeddings/checkpoint.json").read_text()
                )
                self.assertEqual(checkpoint["completed_rows"], 2048)
                calls.clear()
                FakeEncoder.fail = False
                run_embeddings(
                    source,
                    root / "embeddings",
                    root / "step_03_embeddings/model",
                    "cpu",
                    512,
                )
            self.assertEqual(calls, list(range(2048, 2052)))
            vectors = np.load(root / "embeddings/vectors.npy")
            for i in range(len(vectors)):
                self.assertEqual(int(vectors[i].argmax()), i % 8)
            self.assertTrue(
                json.loads((root / "embeddings/checkpoint.json").read_text())[
                    "complete"
                ]
            )

    def test_full_coverage_gate_rejects_a_missing_character(self):
        from src.step_07_verification.integrity import audit_zone_coverage
        from src.common.zones import ZONES
        import json
        import sqlite3

        documents = sqlite3.connect(":memory:")
        documents.row_factory = sqlite3.Row
        documents.execute("CREATE TABLE documents(doc_index INTEGER,record_json TEXT)")
        record = {
            "doc_id": "fixture",
            "title": "",
            "headings": [],
            "sections": [{"blocks": [{"kind": "prose", "text": "complete content"}]}],
        }
        documents.execute("INSERT INTO documents VALUES (?,?)", (0, json.dumps(record)))
        stores = {}
        for zone in ZONES:
            db = sqlite3.connect(":memory:")
            db.row_factory = sqlite3.Row
            db.execute(
                "CREATE TABLE passages(passage_index INTEGER,doc_index INTEGER,char_start INTEGER,char_end INTEGER,tokens INTEGER,text TEXT)"
            )
            stores[zone] = db
        stores["body"].execute(
            'INSERT INTO passages VALUES (0,0,0,15,2,"complete conten")'
        )
        with self.assertRaisesRegex(RuntimeError, "truncated or omitted"):
            audit_zone_coverage(documents, stores)
        documents.close()
        for db in stores.values():
            db.close()

    def test_rank_cache_fetches_only_visible_snippets_and_preserves_document_ranks(
        self,
    ):
        import sqlite3
        import threading
        from collections import OrderedDict
        from src.step_08_search.step_08_search import SearchEngine

        engine = SearchEngine.__new__(SearchEngine)
        engine.summary = {"fingerprint": "fixture"}
        engine.documents = [
            {
                "doc_index": i,
                "doc_id": f"doc_{i:03}",
                "title": f"Title {i}",
                "url": "https://example.com",
                "technology": "Python",
                "category": "programming-language",
            }
            for i in range(25)
        ]
        engine.ids = np.array([d["doc_id"] for d in engine.documents])
        engine.lock = threading.RLock()
        engine.cache = OrderedDict()
        engine.query_cache = OrderedDict()
        engine.technology_options = ["Python"]
        engine.category_options = ["programming-language"]
        cosine_calls = []

        def rank(*args, **kwargs):
            cosine_calls.append(1)
            scores = np.linspace(1.0, 0.0, 25, dtype=np.float32)
            zones = np.full((25, 4), -np.inf, dtype=np.float32)
            zones[:, 2] = scores
            winners = np.full((25, 4), -1, dtype=np.int64)
            winners[:, 2] = np.arange(25)
            return {
                "order": np.arange(25),
                "scores": scores,
                "zones": zones,
                "winners": winners,
                "winning_zone": np.full(25, 2),
                "contributions": None,
                "scanned_vectors": 25,
                "candidate_documents": 25,
                "selected_classes": [],
            }

        engine.rank = rank
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute(
            "CREATE TABLE passages(passage_index INTEGER PRIMARY KEY,passage_id TEXT,text TEXT)"
        )
        db.executemany(
            "INSERT INTO passages VALUES (?,?,?)",
            [(i, str(i), f"Passage {i}") for i in range(25)],
        )
        engine.stores = {"body": db}
        selects = []
        db.set_trace_callback(
            lambda sql: selects.append(sql) if sql.startswith("SELECT") else None
        )
        first = engine.search("test", limit=25, offset=0, page_size=3)
        second = engine.search("test", limit=25, offset=3, page_size=3)
        self.assertEqual(len(selects), 6)
        self.assertEqual(len(cosine_calls), 1)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["retrieved_depth"], 25)
        self.assertEqual(first["returned"], 3)
        self.assertEqual([r["rank"] for r in second["results"]], [4, 5, 6])
        self.assertFalse(
            {r["doc_id"] for r in first["results"]}
            & {r["doc_id"] for r in second["results"]}
        )
        full = engine.search("test", limit=25)
        self.assertEqual(full["results"][:3], first["results"])
        self.assertEqual(full["returned"], 25)
        self.assertAlmostEqual(full["results"][-1]["score"], 0.0, places=6)
        db.close()


if __name__ == "__main__":
    unittest.main()
