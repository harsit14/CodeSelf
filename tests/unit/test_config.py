from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import ConfigLoadError, config_get, load_config_file  # noqa: E402


class ConfigLoaderTests(unittest.TestCase):
    def test_load_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"experiment": {"seed": 123}}), encoding="utf-8")

            config = load_config_file(path)

        self.assertEqual(config_get(config, "experiment.seed"), 123)
        self.assertEqual(config_get(config, "missing.value", "fallback"), "fallback")

    def test_load_simple_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: test",
                        "  seed: 20260601",
                        "generation:",
                        "  backend: mock",
                        "  samples_per_task: 4",
                        "  temperature: 0.8",
                        "  top_p: 0.95",
                        "evaluation:",
                        "  ks: [1, 5, 10]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config_file(path)

        self.assertEqual(config_get(config, "experiment.name"), "test")
        self.assertEqual(config_get(config, "generation.samples_per_task"), 4)
        self.assertEqual(config_get(config, "evaluation.ks"), [1, 5, 10])

    def test_yaml_parser_rejects_odd_indentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text("experiment:\n seed: 1\n", encoding="utf-8")

            with self.assertRaises(ConfigLoadError):
                load_config_file(path)


if __name__ == "__main__":
    unittest.main()
