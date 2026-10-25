"""Configuration and book-style learning contracts."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.common.config import load_config, default_mode
from src.step_08_search.scoring import weighted_zone_scores


class ConfigurationTests(unittest.TestCase):
    def test_flags_choose_the_default_search_mode(self):
        values = load_config()
        self.assertEqual(default_mode(values), "routed-centroid")
        values["category_routing"] = False
        self.assertEqual(default_mode(values), "zones")
        values["zone_weighting"] = False
        self.assertEqual(default_mode(values), "dense")

    def test_invalid_window_settings_are_rejected(self):
        values = load_config()
        values["overlap"] = values["max_tokens"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(values))
            with patch.dict(os.environ, {"NEURAL_CONFIG": str(path)}):
                with self.assertRaises(ValueError):
                    load_config()

    def test_fixed_weight_formula_uses_cosine_without_rescaling_features(self):
        score, parts = weighted_zone_scores(
            [[0.7, 0.65, 0.85, 0.3]], [0.1, 0.2, 0.4, 0.3]
        )
        np.testing.assert_allclose(score, [0.63], atol=1e-6)
        np.testing.assert_allclose(parts.sum(axis=1), score)
        with self.assertRaises(ValueError):
            weighted_zone_scores([[0.7, 0.65, 0.85, 0.3]], [0.1, 0.2, 0.4, 0.4])
