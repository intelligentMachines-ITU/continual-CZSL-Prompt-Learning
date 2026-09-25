"""Lightweight loaders for the CCZSL benchmark splits.

The benchmark lives on disk under ``data/<dataset>/session_<N>/``
with the layout:

    train_pairs.txt, val_pairs.txt, test_pairs.txt, all_pairs.txt
        one "attr obj" per line
    metadata_compositional-split-natural.t7
        torch.save(list[dict]) with per-image entries
        {image: str, attr: str, obj: str, set: "train"|"val"|"test"}
    cumulative/metadata_cumulative.t7          (sessions N >= 1 only)
        same schema, covering the union of sessions 0..N

Only pair-level access requires no dependencies. Image-level metadata
loading requires torch (imported lazily on demand).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DATASETS: tuple[str, ...] = ("mit-states", "ut-zappos", "CGQA")

# Case-insensitive alias table. Users can pass "cgqa", "mit_states", etc.
_DATASET_ALIASES: dict[str, str] = {
    "mit-states": "mit-states",
    "mit_states": "mit-states",
    "mitstates": "mit-states",
    "ut-zappos": "ut-zappos",
    "ut_zappos": "ut-zappos",
    "utzappos": "ut-zappos",
    "cgqa": "CGQA",
    "c-gqa": "CGQA",
    "c_gqa": "CGQA",
}


def default_data_root() -> Path:
    """Return the on-disk root of ``data/`` shipped with this repo."""
    return Path(__file__).resolve().parent.parent / "data"


def _canonical_dataset(name: str) -> str:
    key = name.strip().lower()
    if key in _DATASET_ALIASES:
        return _DATASET_ALIASES[key]
    if name in DATASETS:
        return name
    raise ValueError(
        f"Unknown dataset {name!r}. Expected one of {DATASETS} "
        f"(aliases accepted: {sorted(_DATASET_ALIASES)})."
    )


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


@dataclass(frozen=True)
class SessionSplit:
    """A single-session view of the benchmark.

    Pair lists are lists of ``(attr, obj)`` tuples. ``metadata_path`` and
    ``cumulative_metadata_path`` point at the ``.t7`` files; use
    :func:`load_metadata` (or ``torch.load`` directly) to read them.
    """

    dataset: str
    session_id: int
    data_root: Path
    train_pairs: list[tuple[str, str]] = field(default_factory=list)
    val_pairs: list[tuple[str, str]] = field(default_factory=list)
    test_pairs: list[tuple[str, str]] = field(default_factory=list)
    all_pairs: list[tuple[str, str]] = field(default_factory=list)
    metadata_path: Path | None = None
    cumulative_metadata_path: Path | None = None

    @property
    def attrs(self) -> list[str]:
        return sorted({a for a, _ in self.all_pairs})

    @property
    def objects(self) -> list[str]:
        return sorted({o for _, o in self.all_pairs})

    @property
    def session_dir(self) -> Path:
        return self.data_root / self.dataset / f"session_{self.session_id}"


def _session_dir(data_root: Path, dataset: str, session_id: int) -> Path:
    return data_root / dataset / f"session_{session_id}"


def load_session(
    dataset: str,
    session_id: int,
    data_root: Path | str | None = None,
) -> SessionSplit:
    """Load a single session's pair splits and metadata paths.

    Parameters
    ----------
    dataset
        One of ``"mit-states"``, ``"ut-zappos"``, ``"CGQA"`` (aliases accepted).
    session_id
        Integer session index (0-based).
    data_root
        Optional override for the benchmark root. Defaults to the
        ``data/`` directory shipped alongside this package.
    """
    root = Path(data_root) if data_root is not None else default_data_root()
    ds = _canonical_dataset(dataset)
    sd = _session_dir(root, ds, session_id)
    if not sd.is_dir():
        raise FileNotFoundError(f"Session directory not found: {sd}")

    metadata_path = sd / "metadata_compositional-split-natural.t7"
    cumulative_path = sd / "cumulative" / "metadata_cumulative.t7"

    return SessionSplit(
        dataset=ds,
        session_id=session_id,
        data_root=root,
        train_pairs=_read_pairs(sd / "train_pairs.txt"),
        val_pairs=_read_pairs(sd / "val_pairs.txt"),
        test_pairs=_read_pairs(sd / "test_pairs.txt"),
        all_pairs=_read_pairs(sd / "all_pairs.txt"),
        metadata_path=metadata_path if metadata_path.exists() else None,
        cumulative_metadata_path=cumulative_path if cumulative_path.exists() else None,
    )


def load_metadata(
    dataset: str,
    session_id: int,
    cumulative: bool = False,
    data_root: Path | str | None = None,
) -> list[dict]:
    """Load per-image metadata for a session.

    Returns a list of dicts with keys ``{image, attr, obj, set}``. Requires
    ``torch`` to be installed (imported lazily so pair-only workflows have
    zero heavy dependencies).
    """
    import torch  # lazy

    session = load_session(dataset, session_id, data_root=data_root)
    path = session.cumulative_metadata_path if cumulative else session.metadata_path
    if path is None:
        kind = "cumulative" if cumulative else "session-scoped"
        raise FileNotFoundError(
            f"No {kind} metadata for {session.dataset}/session_{session.session_id}"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def list_sessions(
    dataset: str,
    data_root: Path | str | None = None,
) -> list[int]:
    """Return the sorted list of session ids present on disk for ``dataset``."""
    root = Path(data_root) if data_root is not None else default_data_root()
    ds = _canonical_dataset(dataset)
    ds_dir = root / ds
    if not ds_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")
    ids: list[int] = []
    for p in ds_dir.glob("session_*"):
        if p.is_dir():
            try:
                ids.append(int(p.name.removeprefix("session_")))
            except ValueError:
                continue
    return sorted(ids)
