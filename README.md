# Prompt-Based Continual Compositional Zero-Shot Learning

[[Paper (arXiv:2512.09172)]](https://arxiv.org/abs/2512.09172)

This repository hosts the benchmark and reference loaders for
**PROMPT-BASED CONTINUAL COMPOSITIONAL ZERO-SHOT LEARNING** (PromptCCZSL).

## Overview

PromptCCZSL is a vision–language framework for Continual Compositional
Zero-Shot Learning (CCZSL), where models incrementally learn evolving
attribute–object compositions while retaining previously acquired
knowledge. Unlike conventional continual learning, CCZSL allows
attributes and objects to recur across sessions while their
*compositions* change, creating challenges such as semantic
interference, compositional drift, and catastrophic forgetting.

The central goal of PromptCCZSL is a better balance between **stability**
(retaining previously learned primitive and compositional knowledge) and
**plasticity** (adapting to newly introduced compositions). The
framework is evaluated on the three CCZSL benchmarks in this repository
and compared with existing VLM and non-VLM baselines.

![Problem setting](cczsl-problem.png)

## Method

![PromptCCZSL architecture](method.png)

See the [paper](https://arxiv.org/abs/2512.09172) for a full description
of the architecture and training procedure.

## Datasets

The benchmark is built on top of three compositional zero-shot learning
datasets, re-partitioned into continual-learning sessions:

| Dataset       | Sessions | Total attrs | Total objs | Total pairs | Total images |
| ------------- | -------- | ----------- | ---------- | ----------- | ------------ |
| MIT-States    | 4 (0-3)  | 106         | 207        | 721         | 53,753       |
| UT-Zappos     | 3 (0-2)  | 15          | 10         | 43          | 29,126       |
| C-GQA         | 6 (0-5)  | 233         | 384        | 3,305       | 39,298       |

(*Total* here is measured at the largest cumulative session; see per-session
data cards below.)

### Per-dataset session data cards

#### MIT-States

| Session | attrs | objs | pairs (train/val/test/all) | images (train/val/test) | cumulative images |
| ------- | ----- | ---- | -------------------------- | ----------------------- | ----------------- |
| 0       |    37 |   71 | 161 / 76 / 90 / 237        | 3689 / 1139 / 1332      | —                 |
| 1       |    59 |  126 | 278 / 145 / 187 / 439      | 6919 / 2546 / 2879      | 12,320            |
| 2       |    80 |  163 | 361 / 164 / 229 / 565      | 9010 / 3039 / 3869      | 34,422            |
| 3       |   106 |  207 | 462 / 215 / 294 / 721      | 10720 / 3696 / 4915     | 53,753            |

#### UT-Zappos

| Session | attrs | objs | pairs (train/val/test/all) | images (train/val/test) | cumulative images |
| ------- | ----- | ---- | -------------------------- | ----------------------- | ----------------- |
| 0       |     8 |    6 | 24 / 7 / 9 / 34            | 8448 / 774 / 1444       | —                 |
| 1       |    11 |    8 | 27 / 10 / 14 / 39          | 5436 / 1013 / 961       | 18,076            |
| 2       |    15 |   10 | 32 / 13 / 13 / 43          | 9114 / 1427 / 509       | *(missing)*       |

> **Note (UT-Zappos small-N caveat).** UT-Zappos has very few pairs per
> session (≤ 32 train pairs). Reported metrics on this dataset are more
> sensitive to seed and split choice than on MIT-States or C-GQA.
>
> **Note (UT-Zappos session_2 cumulative).** The cumulative metadata
> file for `ut-zappos/session_2` is currently missing (see
> `CCZL_benchmark/ut-zappos/session_2/cumulative/MISSING.md`). Session-
> scoped experiments are unaffected.

#### C-GQA

| Session | attrs | objs | pairs (train/val/test/all) | images (train/val/test) | cumulative images |
| ------- | ----- | ---- | -------------------------- | ----------------------- | ----------------- |
| 0       |   233 |  363 | — / — / — / 3305           | 10014 / 2743 / 1961     | —                 |
| 1       |   161 |  151 | 491 / 168 / 172 / 678      | 1798 / 342 / 338        | 17,196            |
| 2       |   172 |  360 | 772 / 358 / 265 / 1093     | 6531 / 1551 / 1030      | 26,308            |
| 3       |   178 |  384 | 836 / 366 / 275 / 1131     | 4541 / 1527 / 975       | 33,351            |
| 4       |   183 |  206 | 562 / 225 / 166 / 776      | 2375 / 613 / 421        | 36,760            |
| 5       |   170 |  238 | 539 / 217 / 203 / 784      | 1661 / 504 / 373        | 39,298            |

> **Note (C-GQA session_0 semantics).** C-GQA session_0 has no per-split
> pair files (`train_pairs.txt` / `val_pairs.txt` / `test_pairs.txt`).
> Its `all_pairs.txt` is a **master pool** listing every pair used across
> the benchmark (3305 pairs). The image-level split assignments still
> live in the session_0 metadata file (`set` field), and are usable as a
> base warm-start.

## Directory layout

```
CCZL_benchmark/
├── mit-states/
│   ├── session_0/
│   │   ├── train_pairs.txt          # "attr obj" per line
│   │   ├── val_pairs.txt
│   │   ├── test_pairs.txt
│   │   ├── all_pairs.txt            # = train ∪ val ∪ test (sorted)
│   │   └── metadata_compositional-split-natural.t7    # torch.save(list[dict])
│   ├── session_1/
│   │   ├── ... (as above)
│   │   └── cumulative/
│   │       └── metadata_cumulative.t7      # images across sessions 0..1
│   ├── session_2/ ...
│   └── session_3/ ...
├── ut-zappos/     # same layout, sessions 0-2
└── CGQA/          # same layout, sessions 0-5

cczsl_benchmark/         # importable Python package (loaders)
scripts/                 # convert_t7_to_json, build_manifest, validate_benchmark
manifest.json            # per-file integrity manifest
```

### File formats

- **`{train,val,test,all}_pairs.txt`** — plain text, one
  `attr<space>obj` per line, sorted.
- **`metadata_compositional-split-natural.t7`** — despite the `.t7`
  extension, these are **`torch.save`** archives (not Torch7), holding a
  `list[dict]` with keys `{image, attr, obj, set}` where `set` is one of
  `"train" | "val" | "test"`. Load with `torch.load(..., weights_only=False)`
  or with `cczsl_benchmark.load_metadata` (below).
- **`cumulative/metadata_cumulative.t7`** — same schema, covering the
  union of sessions `0..N` for session `N`. Used for joint-evaluation
  protocols where a model trained through session `N` is evaluated on
  everything seen so far.

## Download instructions (image data)

The image data is **not** redistributed in this repository. Obtain it
directly from the original sources and place it at a path of your
choice; the loader records the filename in the `image` field of each
metadata entry.

- **MIT-States** — https://web.mit.edu/phillipi/Public/states_and_transformations/
  Extract so images live at `<mit-states-root>/images/<attr>_<obj>/*.jpg`.
- **UT-Zappos50K** — https://vision.cs.utexas.edu/projects/finegrained/utzap50k/
  Follow the fine-grained CZSL preprocessing from Naeem et al. 2021.
- **C-GQA** — derived from GQA (https://cs.stanford.edu/people/dorarad/gqa/).
  Use the C-GQA release from Naeem et al. 2021.

## Usage

Loader API (stdlib only for pair access; `torch` imported lazily for
image-level metadata):

```python
from cczsl_benchmark import load_session, load_metadata, list_sessions

# Sessions available on disk
list_sessions("mit-states")            # [0, 1, 2, 3]

# One session's pair splits and metadata paths
s = load_session("mit-states", 1)
s.train_pairs                          # [("ancient", "highway"), ...]
s.attrs, s.objects                     # sorted uniques from all_pairs
s.metadata_path                        # Path to the .t7
s.cumulative_metadata_path             # Path to cumulative/.t7 (or None)

# Image-level metadata
md = load_metadata("mit-states", 1)                    # session-scoped
md_c = load_metadata("cgqa", 3, cumulative=True)       # cumulative 0..3
md[0]  # {'image': 'ancient_highway/234952612_570360a0f5_z.jpg',
       #  'attr': 'ancient', 'obj': 'highway', 'set': 'train'}
```

Dataset name aliases are accepted (`cgqa`, `C-GQA`, `mit_states`, etc.).

If you would rather work with plain JSON, run
`python scripts/convert_t7_to_json.py` to write a `.json` companion next
to every `.t7` (the JSONs are ignored by git; regenerate as needed).

## Evaluation protocol

The benchmark supports two evaluation modes per session `N`:

1. **Session-scoped** — train and evaluate on `session_N` alone using
   `metadata_compositional-split-natural.t7`. Measures raw plasticity.
2. **Cumulative (0..N)** — evaluate on the union of sessions `0..N` using
   `cumulative/metadata_cumulative.t7`. Measures stability + plasticity
   jointly, and is the CCZSL protocol reported in the paper.

Refer to the [paper](https://arxiv.org/abs/2512.09172) for the exact
metric definitions used in the tables.

## Reproducibility and integrity

Every file under `CCZL_benchmark/` is recorded in `manifest.json` with
its size, MD5, and structural count (line count for pair files, entry
count for `.t7` metadata files). To verify a fresh clone:

```bash
python scripts/validate_benchmark.py
# OK: 71 files verified
```

To regenerate the manifest after intentional benchmark changes:

```bash
python scripts/build_manifest.py
```

Run `validate_benchmark.py` in CI or as a pre-push hook to catch
truncation, corruption, or unintended modification of any file.

## Git LFS

All `.t7` metadata files (and other likely-large binaries — `.pt`,
`.pth`, `.ckpt`, `.safetensors`, archives) are stored via Git LFS. To
work with the repository:

```bash
git lfs install                  # once per machine
git clone <this-repo>            # LFS pointers are resolved automatically
```

If you clone without LFS installed, `.t7` files will appear as small
text pointer files. Run `git lfs install && git lfs pull` to retrieve
the real bytes.

## License

The benchmark artifacts and code in this repository are released under
CC-BY-4.0 (see `LICENSE`). The underlying image datasets are **not**
redistributed here and remain under their original licenses.

## Citation

```bibtex
@article{maryam2025promptcczsl,
  title  = {Prompt-Based Continual Compositional Zero-Shot Learning},
  author = {Sauda Maryam and Sara Nadeem and Faisal Qureshi and Mohsen Ali},
  year   = {2025},
  eprint = {2512.09172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url    = {https://arxiv.org/abs/2512.09172}
}
```

See also `CITATION.cff` for reference-manager import.
