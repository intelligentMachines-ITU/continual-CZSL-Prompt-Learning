"""Walk the benchmark and emit `manifest.json` with per-file integrity info.

The manifest records, for every file under `CCZL_benchmark/`:
- `path` (relative to the benchmark root, forward-slash separated)
- `size` in bytes
- `md5` hash of the raw bytes on disk
- `kind`: `"pair-list"` (newline-delimited attr/obj pairs), `"metadata"`
  (torch-saved list[dict]), or `"other"`
- `line_count` for pair-list files
- `entry_count` for metadata files (loads with torch)

Use this together with `scripts/validate_benchmark.py` to catch
truncation, corruption, or accidental modification of any benchmark
file — which is exactly the failure mode the original .t7 truncation
would have exhibited.

Usage:
    python scripts/build_manifest.py [--root CCZL_benchmark] [--out manifest.json]
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


def classify(path: Path) -> str:
    name = path.name
    if name.endswith("_pairs.txt"):
        return "pair-list"
    if name.endswith(".t7"):
        return "metadata"
    return "other"


def entry_count(path: Path) -> int:
    import torch  # lazy
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return len(obj) if hasattr(obj, "__len__") else 0


def build(root: Path) -> list[dict]:
    entries: list[dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "MISSING.md":
            continue  # placeholder, don't lock its hash
        kind = classify(p)
        rec: dict = {
            "path": str(p.relative_to(root)).replace("\\", "/"),
            "size": p.stat().st_size,
            "md5": md5_of(p),
            "kind": kind,
        }
        if kind == "pair-list":
            rec["line_count"] = line_count(p)
        elif kind == "metadata":
            try:
                rec["entry_count"] = entry_count(p)
            except Exception as e:  # noqa: BLE001
                rec["entry_count_error"] = f"{type(e).__name__}: {e}"
        entries.append(rec)
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent / "CCZL_benchmark",
        type=Path,
    )
    ap.add_argument(
        "--out",
        default=Path(__file__).resolve().parent.parent / "manifest.json",
        type=Path,
    )
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"Benchmark root not found: {args.root}", file=sys.stderr)
        return 1

    entries = build(args.root)
    args.out.write_text(json.dumps(
        {"root": str(args.root.name), "files": entries},
        indent=2,
        ensure_ascii=False,
    ) + "\n")
    print(f"wrote {args.out}  ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
