#!/usr/bin/env python
# scripts/download_models_from_manifest.py
"""Download model files listed in models/active_models.json from a GitHub release.

Usage: python scripts/download_models_from_manifest.py <release_tag>
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: download_models_from_manifest.py <release_tag>", file=sys.stderr)
        sys.exit(1)

    tag = sys.argv[1]
    manifest_path = Path("models/active_models.json")
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    env = {**os.environ}  # GH_TOKEN must be set by caller

    failed = []
    for pos, info in manifest.get("models", {}).items():
        fname = info["file"]
        print(f"Downloading {pos}: {fname}")
        r = subprocess.run(
            ["gh", "release", "download", tag,
             "--pattern", fname, "--dir", "models", "--clobber"],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr.strip()}", file=sys.stderr)
            failed.append(fname)

    if failed:
        print(f"Failed to download: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
