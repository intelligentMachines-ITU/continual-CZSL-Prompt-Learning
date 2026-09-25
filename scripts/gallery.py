"""Render a browsable HTML gallery for a benchmark session.

Groups a session's images by (attr, obj) pair and emits an HTML file
that lays out a fixed number of sample thumbnails per pair. Useful for
sanity-checking splits, spotting label noise, and figures in papers.

The script does not copy image bytes. It writes `<img src>` references
that resolve at browser-render time against `--images-root`. The user
supplies the image data separately (see the README download section).

Usage:
    python scripts/gallery.py <dataset> <session_id> \
        --images-root /path/to/<dataset>/images \
        [--cumulative]                 [--per-pair 4] \
        [--split {train,val,test,all}] [--seed 0] \
        [--out gallery.html]           [--max-pairs 0]

Examples:
    # Session-scoped gallery, four random samples per pair, from the
    # training split only.
    python scripts/gallery.py mit-states 1 \
        --images-root ~/data/mit-states/images \
        --split train --per-pair 4 --out gallery_ms_1_train.html

    # Cumulative gallery through session 3, one sample per pair.
    python scripts/gallery.py cgqa 3 --cumulative \
        --images-root ~/data/cgqa/images --per-pair 1 \
        --out gallery_cgqa_cum_3.html
"""

from __future__ import annotations

import argparse
import html
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cczsl_benchmark import load_metadata, load_session  # noqa: E402


_HTML_HEADER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{
    --thumb: 128px;
    --pad: 6px;
    --fg: #222;
    --muted: #777;
    --bg: #fff;
    --border: #e5e5e5;
    --stripe: #fafafa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--fg);
    background: var(--bg);
    margin: 24px;
    line-height: 1.4;
  }}
  header h1 {{ margin: 0 0 4px 0; }}
  header .sub {{ color: var(--muted); font-size: 0.9rem; }}
  .stats {{
    display: flex; flex-wrap: wrap; gap: 12px;
    margin: 12px 0 24px 0; font-size: 0.9rem;
  }}
  .stats span {{
    background: var(--stripe); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 4px;
  }}
  section.pair {{
    border-top: 1px solid var(--border);
    padding: 12px 0 20px 0;
  }}
  section.pair h2 {{
    font-size: 1rem; margin: 0 0 6px 0;
  }}
  section.pair .count {{ color: var(--muted); font-weight: normal; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(var(--thumb), 1fr));
    gap: var(--pad);
  }}
  .cell {{
    position: relative;
    aspect-ratio: 1 / 1;
    background: var(--stripe);
    border: 1px solid var(--border);
    overflow: hidden;
    border-radius: 3px;
  }}
  .cell img {{
    width: 100%; height: 100%; object-fit: cover; display: block;
  }}
  .cell .caption {{
    position: absolute; inset: auto 0 0 0;
    background: rgba(0,0,0,0.55); color: #fff;
    font-size: 0.7rem; padding: 2px 4px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .set-train {{ box-shadow: inset 0 0 0 2px #2b7cd3; }}
  .set-val   {{ box-shadow: inset 0 0 0 2px #d29a2b; }}
  .set-test  {{ box-shadow: inset 0 0 0 2px #d3492b; }}
  .legend {{ font-size: 0.85rem; color: var(--muted); }}
  .legend span {{ margin-right: 12px; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</header>
<div class="stats">
  <span><b>pairs shown:</b> {n_pairs}</span>
  <span><b>total pair candidates:</b> {n_pair_candidates}</span>
  <span><b>images shown:</b> {n_shown}</span>
  <span><b>per-pair cap:</b> {per_pair}</span>
  <span><b>split filter:</b> {split}</span>
  <span><b>seed:</b> {seed}</span>
</div>
<div class="legend">
  <span><span class="swatch" style="background:#2b7cd3"></span>train</span>
  <span><span class="swatch" style="background:#d29a2b"></span>val</span>
  <span><span class="swatch" style="background:#d3492b"></span>test</span>
</div>
"""

_HTML_FOOTER = "</body></html>\n"


def _write_pair(fh, attr: str, obj: str, samples: list[dict], images_root: Path) -> int:
    fh.write(f'<section class="pair">\n')
    fh.write(
        f'<h2>{html.escape(attr)} {html.escape(obj)} '
        f'<span class="count">({len(samples)} shown)</span></h2>\n'
    )
    fh.write('<div class="grid">\n')
    for e in samples:
        rel = e["image"]
        src = str(images_root / rel).replace("\\", "/")
        fh.write(
            f'  <div class="cell set-{html.escape(e["set"])}">'
            f'<img loading="lazy" src="{html.escape(src)}" '
            f'alt="{html.escape(f"{attr} {obj}")}">'
            f'<div class="caption">{html.escape(rel)}</div>'
            f'</div>\n'
        )
    fh.write('</div>\n</section>\n')
    return len(samples)


def build_gallery(
    dataset: str,
    session_id: int,
    images_root: Path,
    cumulative: bool = False,
    split: str = "all",
    per_pair: int = 4,
    max_pairs: int = 0,
    seed: int = 0,
    out_path: Path = Path("gallery.html"),
) -> Path:
    md = load_metadata(dataset, session_id, cumulative=cumulative)
    if split != "all":
        md = [e for e in md if e["set"] == split]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in md:
        groups[(e["attr"], e["obj"])].append(e)

    rng = random.Random(seed)
    ordered_pairs = sorted(groups)
    if max_pairs > 0:
        ordered_pairs = ordered_pairs[:max_pairs]

    session = load_session(dataset, session_id)
    kind = "cumulative" if cumulative else "session-scoped"
    title = f"{session.dataset} / session_{session_id} / {kind}"
    subtitle = (
        f"images root: <code>{html.escape(str(images_root))}</code> "
        f"&middot; split filter: <code>{split}</code>"
    )

    with out_path.open("w") as fh:
        n_shown = 0
        # We know the totals after sampling, but we want them in the header.
        # Write a placeholder, then patch. Simpler: buffer body strings.
        body_chunks: list[str] = []
        for pair in ordered_pairs:
            candidates = groups[pair]
            k = min(per_pair, len(candidates)) if per_pair > 0 else len(candidates)
            samples = rng.sample(candidates, k) if k < len(candidates) else list(candidates)
            attr, obj = pair
            chunk_fh = _StringSink()
            _write_pair(chunk_fh, attr, obj, samples, images_root)
            body_chunks.append(chunk_fh.getvalue())
            n_shown += len(samples)

        header = _HTML_HEADER.format(
            title=html.escape(title),
            subtitle=subtitle,
            n_pairs=len(ordered_pairs),
            n_pair_candidates=len(groups),
            n_shown=n_shown,
            per_pair=("unbounded" if per_pair <= 0 else per_pair),
            split=split,
            seed=seed,
        )
        fh.write(header)
        for chunk in body_chunks:
            fh.write(chunk)
        fh.write(_HTML_FOOTER)

    return out_path


class _StringSink:
    """Minimal file-like buffer so `_write_pair` can accept either sink."""
    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, s: str) -> None:
        self._parts.append(s)

    def getvalue(self) -> str:
        return "".join(self._parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("session_id", type=int)
    ap.add_argument("--images-root", required=True, type=Path,
                    help="Directory that holds the raw image files. The 'image' "
                         "field in each metadata entry is joined against this path.")
    ap.add_argument("--cumulative", action="store_true",
                    help="Use cumulative_0..N metadata instead of session-scoped.")
    ap.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    ap.add_argument("--per-pair", type=int, default=4,
                    help="Cap images shown per (attr, obj) pair. Use 0 for unlimited.")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Cap number of pairs shown. 0 means all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("gallery.html"))
    args = ap.parse_args(argv)

    out = build_gallery(
        dataset=args.dataset,
        session_id=args.session_id,
        images_root=args.images_root.resolve(),
        cumulative=args.cumulative,
        split=args.split,
        per_pair=args.per_pair,
        max_pairs=args.max_pairs,
        seed=args.seed,
        out_path=args.out,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
