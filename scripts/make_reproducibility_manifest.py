#!/usr/bin/env python3
"""Write CodeSelf reproducibility metadata and an optional artifact archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.reporting import (  # noqa: E402
    build_reproducibility_manifest,
    write_manifest_json,
    write_manifest_markdown,
    write_reproducibility_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to inspect.")
    parser.add_argument("--output", type=Path, required=True, help="JSON manifest path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown manifest path.")
    parser.add_argument("--archive", type=Path, help="Optional tar.gz archive path.")
    args = parser.parse_args()

    manifest = build_reproducibility_manifest(args.root)
    write_manifest_json(manifest, args.output)
    print(f"wrote reproducibility manifest to {args.output}")
    print(f"artifact_count: {manifest.artifact_count}")
    print(f"config_artifact_count: {len(manifest.config_artifacts)}")
    print(f"dataset_artifact_count: {len(manifest.dataset_artifacts)}")
    print(f"environment_artifact_count: {len(manifest.environment_artifacts)}")
    print(f"model_reference_count: {len(manifest.model_references)}")
    print(f"checkpoint_manifest_count: {len(manifest.checkpoint_manifests)}")
    print(f"checkpoint_artifact_count: {len(manifest.checkpoint_artifacts)}")

    if args.markdown_output is not None:
        write_manifest_markdown(manifest, args.markdown_output)
        print(f"wrote Markdown manifest to {args.markdown_output}")

    if args.archive is not None:
        write_reproducibility_archive(manifest, args.root, args.archive)
        print(f"wrote reproducibility archive to {args.archive}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
