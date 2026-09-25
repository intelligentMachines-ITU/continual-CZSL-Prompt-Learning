"""Convert every `.t7` metadata file in the benchmark to a sibling `.json`.

Each `.t7` in `CCZL_benchmark/` is a `torch.save(list[dict])` where each
dict is `{image, attr, obj, set}`. This script emits a `.json` companion
next to each `.t7` so downstream users can read the metadata without a
Torch dependency.

The script is idempotent: safe to re-run. It skips files whose `.json`
sibling is newer than the `.t7`.

Usage:
    python scripts/convert_t7_to_json.py [--force] [--root CCZL_benchmark]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(t7_path: Path, force: bool = False) -> tuple[Path, int]:
    import torch  # lazy so `--help` works without torch installed

    json_path = t7_path.with_suffix(".json")
    if json_path.exists() and not force:
        if json_path.stat().st_mtime >= t7_path.stat().st_mtime:
            return json_path, -1  # skipped
    obj = torch.load(t7_path, map_location="cpu", weights_only=False)
    with json_path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False)
    return json_path, len(obj) if hasattr(obj, "__len__") else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=Path(__file__).resolve().parent.parent / "CCZL_benchmark",
        type=Path,
        help="Benchmark root (default: CCZL_benchmark/ next to this script).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .json files even if newer than the .t7.",
    )
    args = ap.parse_args(argv)

    t7_files = sorted(args.root.rglob("*.t7"))
    if not t7_files:
        print(f"No .t7 files found under {args.root}", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    for t7 in t7_files:
        try:
            out, n = convert(t7, force=args.force)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t7}: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        rel = t7.relative_to(args.root)
        if n == -1:
            print(f"  skip  {rel}  (json newer)")
            skipped += 1
        else:
            print(f"  wrote {rel.with_suffix('.json')}  ({n} entries)")
            converted += 1
    print(f"\nconverted={converted}, skipped={skipped}, total={len(t7_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
