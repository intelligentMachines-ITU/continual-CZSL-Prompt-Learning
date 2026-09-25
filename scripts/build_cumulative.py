"""Rebuild the cumulative metadata file for a given session.

Semantics (verified against the 9 intact cumulative files):

    cumulative_0..N = concat(
        session_scoped_metadata_0,
        session_scoped_metadata_1,
        ...,
        session_scoped_metadata_N,
    )

Each session-scoped file is a `torch.save(list[dict])` with entries
`{image, attr, obj, set}`. The cumulative file is the concatenation of
these lists in session order, with all original `set` labels preserved.
This script writes the result via `torch.save` to
`<dataset>/session_<N>/cumulative/metadata_cumulative.t7`.

Usage:
    python scripts/build_cumulative.py <dataset> <session_id>
    python scripts/build_cumulative.py --all             # rebuild every cumulative file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from repo root or scripts/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cczsl_benchmark.loaders import (  # noqa: E402
    DATASETS,
    default_data_root,
    list_sessions,
    _canonical_dataset,
    _session_dir,
)


def rebuild(dataset: str, session_id: int, data_root: Path) -> Path:
    import torch  # lazy

    ds = _canonical_dataset(dataset)
    if session_id < 1:
        raise ValueError(
            f"Cumulative files exist only for session_N with N >= 1 "
            f"(session_0 has no prior sessions to accumulate)."
        )

    combined: list[dict] = []
    for sid in range(session_id + 1):
        sd = _session_dir(data_root, ds, sid)
        src = sd / "metadata_compositional-split-natural.t7"
        if not src.exists():
            raise FileNotFoundError(f"Missing session-scoped metadata: {src}")
        combined.extend(torch.load(src, map_location="cpu", weights_only=False))

    out_dir = _session_dir(data_root, ds, session_id) / "cumulative"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metadata_cumulative.t7"
    torch.save(combined, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", help="Dataset name (aliases OK); omit with --all.")
    ap.add_argument("session_id", nargs="?", type=int, help="Session index (>= 1); omit with --all.")
    ap.add_argument(
        "--data-root",
        default=default_data_root(),
        type=Path,
        help="Benchmark root (default: data/ next to the package).",
    )
    ap.add_argument("--all", action="store_true", help="Rebuild every cumulative file.")
    args = ap.parse_args(argv)

    if args.all:
        for ds in DATASETS:
            for sid in list_sessions(ds, data_root=args.data_root):
                if sid == 0:
                    continue
                out = rebuild(ds, sid, args.data_root)
                print(f"  wrote {out.relative_to(args.data_root)}")
        return 0

    if args.dataset is None or args.session_id is None:
        ap.error("Provide `<dataset> <session_id>` or `--all`.")

    out = rebuild(args.dataset, args.session_id, args.data_root)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
