"""Textbook scoring, learning and extraction regressions."""

import unittest
from types import SimpleNamespace
import numpy as np
from src.step_08_search.scoring import weighted_zone_scores
from src.common.learning_metrics import classification_metrics
from src.step_05_classification.step_05_classification import (
    memberships,
    build_document_vectors,
)
import tempfile
from pathlib import Path
from src.step_01_extraction.extract_html import extract_document
from src.step_08_search.step_08_search import SearchEngine


class LearningTests(unittest.TestCase):
    def test_category_routing_respects_explicit_technology_filter(self):
        engine = SearchEngine.__new__(SearchEngine)
        engine.documents = [
            {"technology": "Python", "category": "language"},
            {"technology": "Docker", "category": "containers"},
        ]
        engine.ids = np.array(["python-doc", "docker-doc"])
        engine.technology_options = ["Python", "Docker"]
        engine.category_options = ["language", "containers"]
        engine._query = lambda query: np.array([1.0, 0.0])
        engine.classifier = SimpleNamespace(
            document_classes=np.array([[0, -1], [1, -1]]),
            query_scores=lambda *args: np.array([0.9, 0.1]),
        )
        engine.routing = {}
        engine.routing = {"schema": "category-postings-v2"}
        engine.config = {"route_classes": 1, "zone_weighting": True}
        engine.routing_rows = {
            f"{c}_{z}": np.array([c])
            for c in (0, 1)
            for z in ("title", "headings", "body", "code")
        }
        engine.weights = np.array([1.0, 0.0, 0.0, 0.0])
        allowed_seen = []

        def zone_scores(vector, allowed, selected_rows=None):
            allowed_seen.append(allowed.copy())
            return (
                np.array(
                    [[0.9, -np.inf, -np.inf, -np.inf], [0.8, -np.inf, -np.inf, -np.inf]]
                ),
                np.zeros((2, 4), dtype=int),
                1,
            )

        engine.zone_scores = zone_scores
        result = engine.rank("Python", mode="routed-centroid", technology="Docker")
        self.assertEqual(result["selected_classes"], [1])
        self.assertEqual(result["order"].tolist(), [1])
        np.testing.assert_array_equal(allowed_seen[0], [False, True])

    def test_absent_zones_have_zero_contribution(self):
        values = np.array(
            [[0.2, 0.8, -np.inf, -np.inf], [-np.inf, -np.inf, 0.6, -np.inf]]
        )
        score, parts = weighted_zone_scores(values, [0.1, 0.3, 0.4, 0.2])
        np.testing.assert_allclose(score, [0.26, 0.24], atol=1e-6)
        np.testing.assert_allclose(parts.sum(axis=1), score, atol=1e-6)

    def test_zone_fallback_and_empty_documents(self):
        score, _ = weighted_zone_scores([[-np.inf, 0.8], [-np.inf, -np.inf]], [1, 0])
        self.assertAlmostEqual(score[0], 0.0, places=6)
        self.assertTrue(np.isneginf(score[1]))

    def test_secondary_membership_margin_and_ties(self):
        actual = memberships(
            np.array([[0.8, 0.79, 0.1], [0.8, 0.7, 0.1], [0.5, 0.5, 0.1]]), 0.03
        )
        np.testing.assert_array_equal(actual, [[0, 1], [0, -1], [0, 1]])
        np.testing.assert_array_equal(
            memberships(np.array([[0.8, 0.79]]), 0.03, 1), [[0, -1]]
        )

    def test_zone_balanced_document_vectors(self):
        engine = SimpleNamespace(
            documents=[{}],
            encoder=SimpleNamespace(dimension=2),
            matrices={
                "title": np.array([[1.0, 0.0]]),
                "headings": np.empty((0, 2)),
                "body": np.array([[0.0, 1.0]] * 10),
                "code": np.empty((0, 2)),
            },
            mappings={
                "title": np.array([0]),
                "headings": np.array([], dtype=int),
                "body": np.array([0] * 10),
                "code": np.array([], dtype=int),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "step_05_classification").mkdir()
            values = build_document_vectors(directory, engine)
        np.testing.assert_allclose(values, [[2**-0.5, 2**-0.5]], atol=1e-6)

    def test_macro_f1_is_not_accuracy(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 5) / 2)

    def test_table_cells_remain_separate_without_synthetic_headers(self):
        record = {
            "doc_id": "x",
            "title": "API",
            "url": "https://example.com",
            "technology": "test",
        }
        raw = b"<title>API</title><main><p>This explains the technical fields in the following table.</p><table><tr><th>Name</th><th>Type</th></tr><tr><td>id</td><td>int</td></tr></table></main>"
        result = extract_document(raw, record)
        self.assertIn("Name Type id int", result["body_text"])
        self.assertNotIn("Name: id", result["body_text"])
        self.assertNotIn(
            "table", {b["kind"] for s in result["sections"] for b in s["blocks"]}
        )

    def test_standalone_multiline_code_preserves_indentation(self):
        record = {
            "doc_id": "x",
            "title": "API",
            "url": "https://example.com",
            "technology": "test",
        }
        raw = b"<title>API</title><main><p>This is ordinary text before the source code example in this document.</p><code>if ready:\n    run()</code></main>"
        result = extract_document(raw, record)
        self.assertEqual(result["code_text"], "if ready:\n    run()")
        self.assertNotIn("run()", result["body_text"])
