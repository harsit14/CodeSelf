from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.reporting import (  # noqa: E402
    build_reproducibility_manifest,
    collect_artifacts,
    write_reproducibility_archive,
)


class ReproducibilityTests(unittest.TestCase):
    def test_collect_artifacts_includes_public_surface_and_excludes_runtime_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "configs").mkdir()
            (root / "outputs" / "reports").mkdir(parents=True)
            (root / "src" / "pkg" / "module.py").write_text("x = 1\n", encoding="utf-8")
            (root / "configs" / "example.json").write_text("{}\n", encoding="utf-8")
            (root / "outputs" / "reports" / "keep.example.md").write_text(
                "# example\n", encoding="utf-8"
            )
            (root / "outputs" / "reports" / "runtime.md").write_text(
                "# runtime\n", encoding="utf-8"
            )

            artifacts = collect_artifacts(root)
            paths = {artifact.path for artifact in artifacts}

        self.assertIn("src/pkg/module.py", paths)
        self.assertIn("configs/example.json", paths)
        self.assertNotIn("outputs/reports/keep.example.md", paths)
        self.assertNotIn("outputs/reports/runtime.md", paths)

    def test_manifest_renders_markdown_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "codeself").mkdir(parents=True)
            (root / "configs" / "experiments").mkdir(parents=True)
            (root / "data" / "splits").mkdir(parents=True)
            (root / "outputs" / "checkpoints" / "run").mkdir(parents=True)
            (root / "scripts").mkdir()
            (root / "README.md").write_text("# CodeSelf\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='codeself'\n", encoding="utf-8")
            (root / "src" / "codeself" / "__init__.py").write_text(
                "__version__ = '9.9.9'\n", encoding="utf-8"
            )
            (root / "scripts" / "demo.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "configs" / "experiments" / "run.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "backend": "transformers",
                            "name": "policy-model",
                            "tokenizer": "policy-tokenizer",
                            "revision": "policy-rev",
                            "tokenizer_revision": "tokenizer-rev",
                            "value_model": {
                                "enabled": True,
                                "name": "critic-model",
                                "revision": "critic-rev",
                            },
                        },
                        "generation": {"model": "generator-model"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "data" / "splits" / "split_manifest.json").write_text(
                "{\"split_hash\":\"abc\"}\n",
                encoding="utf-8",
            )
            (root / "outputs" / "checkpoints" / "run" / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "kind": "policy_model_state",
                                "path": "state/policy_model.pt",
                                "bytes": 123,
                                "sha256": "a" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_reproducibility_manifest(
                root,
                generated_at_utc="2026-06-01T00:00:00+00:00",
            )
            archive_path = root / "manifest.tar.gz"
            write_reproducibility_archive(manifest, root, archive_path)
            markdown = manifest.to_markdown()
            model_roles = {
                reference.role: reference.name
                for reference in manifest.model_references
            }
            config_paths = {artifact.path for artifact in manifest.config_artifacts}
            dataset_paths = {artifact.path for artifact in manifest.dataset_artifacts}
            environment_paths = {artifact.path for artifact in manifest.environment_artifacts}

            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
                manifest_file = archive.extractfile("codeself_reproducibility/manifest.json")
                assert manifest_file is not None
                payload = json.loads(manifest_file.read().decode("utf-8"))

        self.assertEqual(manifest.package_version, "9.9.9")
        self.assertIn("CodeSelf Reproducibility Manifest", markdown)
        self.assertIn("Experiment Config Hashes", markdown)
        self.assertIn("Model References", markdown)
        self.assertIn("configs/experiments/run.json", config_paths)
        self.assertIn("data/splits/split_manifest.json", dataset_paths)
        self.assertIn("pyproject.toml", environment_paths)
        self.assertEqual(model_roles["policy"], "policy-model")
        self.assertEqual(model_roles["value_model"], "critic-model")
        self.assertEqual(model_roles["generation"], "generator-model")
        self.assertEqual(manifest.checkpoint_artifacts[0].sha256, "a" * 64)
        self.assertIn("codeself_reproducibility/manifest.json", names)
        self.assertIn("codeself_reproducibility/README.md", names)
        self.assertEqual(payload["artifact_count"], manifest.artifact_count)
        self.assertEqual(payload["model_reference_count"], 3)

    def test_reproducibility_script_writes_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "manifest.json"
            markdown_path = Path(tmpdir) / "manifest.md"
            archive_path = Path(tmpdir) / "manifest.tar.gz"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "make_reproducibility_manifest.py"),
                    "--output",
                    str(json_path),
                    "--markdown-output",
                    str(markdown_path),
                    "--archive",
                    str(archive_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(archive_path.exists())
            self.assertIn("artifact_count", completed.stdout)


if __name__ == "__main__":
    unittest.main()
