#!/usr/bin/env python3
"""Build a manifest from a model directory you already downloaded.

    python examples/make_manifest.py ./weights/qwen2.5-7b-instruct \
        --source hf:Qwen/Qwen2.5-7B-Instruct \
        --revision <the 40-char commit sha you downloaded> \
        --license apache-2.0 \
        --out models/qwen2.5-7b-instruct.json

Then commit the manifest. From that point on, `punk verify` is a tripwire: if
those files ever change, you find out in seconds instead of never.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from punkai.registry.manifest import FileEntry, ModelManifest  # noqa: E402
from punkai.registry.verify import scan_directory, sha256_file  # noqa: E402

SKIP = {".gitattributes", ".DS_Store"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--source", required=True, help="e.g. hf:Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", required=True, help="the full 40-char commit sha")
    parser.add_argument("--license", required=True, help="e.g. apache-2.0 (or 'unknown')")
    parser.add_argument("--name")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP:
            continue
        rel = path.relative_to(root).as_posix()
        print(f"hashing {rel} …", file=sys.stderr)
        files.append(FileEntry(rel, sha256_file(path), path.stat().st_size))

    manifest = ModelManifest(
        name=args.name or root.name,
        source=args.source,
        revision=args.revision,
        license=args.license,
        files=files,
    )
    manifest.save(args.out)
    print(f"\nwrote {args.out} ({len(files)} files)")

    findings = scan_directory(root)
    if findings:
        print("\nheads up -- this directory contains:")
        for finding in findings:
            print(f"  {finding}")
        print("\nRead anything flagged 'remote-code' before you ever pass trust_remote_code=True.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
