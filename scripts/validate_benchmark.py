"""Verify every file under `data/` matches `manifest.json`.

Exits 0 on success, nonzero if any file is missing, has the wrong size,
wrong md5, wrong line count, or wrong torch entry count. This is the
guardrail that would have caught the .t7 truncation described in the
LFS-setup commit.

Usage:
    python scripts/validate_benchmark.py [--root data] [--manifest manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def line_count(path: Path) -> int:
    return sum(1 for _ in path.open("rb"))


def entry_count(path: Path) -> int:
    import torch  # lazy
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return len(obj) if hasattr(obj, "__len__") else 0


def check_one(entry: dict, root: Path) -> list[str]:
    path = root / entry["path"]
    errs: list[str] = []
    if not path.is_file():
        return [f"{entry['path']}: missing"]

    size = path.stat().st_size
    if size != entry["size"]:
        errs.append(f"{entry['path']}: size {size} != expected {entry['size']}")

    got_md5 = md5_of(path)
    if got_md5 != entry["md5"]:
        errs.append(f"{entry['path']}: md5 {got_md5} != expected {entry['md5']}")

    if entry.get("kind") == "pair-list" and "line_count" in entry:
        got = line_count(path)
        if got != entry["line_count"]:
            errs.append(
                f"{entry['path']}: line_count {got} != expected {entry['line_count']}"
            )
    if entry.get("kind") == "metadata" and "entry_count" in entry:
        try:
            got = entry_count(path)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{entry['path']}: torch.load failed: {type(e).__name__}: {e}")
        else:
            if got != entry["entry_count"]:
                errs.append(
                    f"{entry['path']}: entry_count {got} != expected {entry['entry_count']}"
                )
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent / "data",
        type=Path,
    )
    ap.add_argument(
        "--manifest",
        default=Path(__file__).resolve().parent.parent / "manifest.json",
        type=Path,
    )
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.root.is_dir():
        print(f"Benchmark root not found: {args.root}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text())
    files = manifest.get("files", [])

    all_errs: list[str] = []
    for entry in files:
        all_errs.extend(check_one(entry, args.root))

    if all_errs:
        print(f"FAIL: {len(all_errs)} problem(s) across {len(files)} files:")
        for e in all_errs:
            print(f"  - {e}")
        return 2

    print(f"OK: {len(files)} files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
