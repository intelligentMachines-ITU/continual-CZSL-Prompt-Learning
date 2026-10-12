# Prompt-Based Continual Compositional Zero-Shot Learning

[[Paper (arXiv:2512.09172)]](https://arxiv.org/abs/2512.09172)

This branch holds the training and evaluation code for
**Prompt-Based Continual Compositional Zero-Shot Learning** (PromptCCZSL).
It runs on two continual benchmarks:

1. **PromptCCZSL benchmark**: MIT-States, UT-Zappos and C-GQA, split into
   continual sessions. The benchmark files and loaders are on the
   [`fzq`](https://github.com/intelligentMachines-ITU/continual-CZSL-Prompt-Learning/tree/fzq) branch.
2. **ComPIL benchmark**: the MIT-States and C-GQA task splits from
   [CompIL](https://github.com/Lixsp11/CompIL), with 5 tasks each.

## Overview

A Continual Compositional Zero-Shot Learning (CCZSL) model learns new
attribute-object compositions session by session. It must keep what it
learned in earlier sessions. Attributes and objects recur across sessions,
but their compositions change. This causes semantic interference,
compositional drift and catastrophic forgetting.

PromptCCZSL balances *stability* (keeping primitive and compositional
knowledge) with *plasticity* (adapting to new compositions).

![Problem setting](https://raw.githubusercontent.com/intelligentMachines-ITU/continual-CZSL-Prompt-Learning/fzq/cczsl-problem.png)

## Architecture

![PromptCCZSL architecture](https://raw.githubusercontent.com/intelligentMachines-ITU/continual-CZSL-Prompt-Learning/fzq/method.png)

The model uses a frozen CLIP ViT-L/14 backbone. Separate prompt branches
handle the attribute, the object and the composition. Adapters and
cross-modal attention align each prompt with the image features.

Session 0 trains the model on the first session alone (`train.py`).
Each later session (`session_train.py`) starts from the previous session's
model and keeps a frozen copy of it as the teacher. The training loss is:

| Term | Flag (default) | Purpose |
|---|---|---|
| Task loss + distillation | `--alpha` (0.65) | `(1 - alpha) * task + alpha * KD`. KD is a KL loss on the attribute, object and pair logits of the classes the teacher already knows (temperature 2.5) |
| Cosine anchor | `--L_CAL` (0.05) | Keeps the student's soft attribute/object prompts close to the teacher's |
| Orthogonality | `--L_ORTHO` (0.05) | Keeps the new primitive prompts orthogonal to the old ones |
| Diversity | `--L_DIV` (0.005) | Spreads the new attribute and object prompts apart |

The paper describes the architecture and training procedure in full.

## Repository layout

```
code/
├── train.py                 # session 0
├── session_train.py         # sessions 1+ (teacher-student)
├── test.py                  # evaluation
├── dataset.py               # CompositionDataset, selected by --session
├── parameters.py            # command-line flags
├── clip_modules/            # CLIP wrapper and tokenizer
├── config/troika/           # yml configs: <dataset>.yml (PromptCCZSL), <dataset>-compil.yml (ComPIL)
├── datasets/                # per-dataset session configs (data paths live here)
├── run/                     # PromptCCZSL benchmark scripts
└── run_compil/              # ComPIL benchmark scripts and summarize_compil.py
```

## Setup

```bash
conda create --name troika python=3.7
conda activate troika
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r code/requirements.txt
```

Download the CLIP checkpoint to `code/model/`:

```bash
mkdir -p code/model && cd code/model
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
```

Images, benchmark metadata (`.t7`), checkpoints and logs are not stored in
this branch.

## Data

### Image datasets

Get the images from the original sources:

* **MIT-States**: https://web.mit.edu/phillipi/Public/states_and_transformations/
* **UT-Zappos50K**: https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ (fine-grained CZSL preprocessing from Naeem et al. 2021)
* **C-GQA**: the C-GQA release from Naeem et al. 2021

### PromptCCZSL benchmark

Copy the session files from the
[`fzq`](https://github.com/intelligentMachines-ITU/continual-CZSL-Prompt-Learning/tree/fzq)
branch to `code/download_data/CCZSL_benchmark/<dataset>/`. The `fzq` README
lists the per-session statistics.

| Dataset    | Sessions | `--session` names                   | Config                |
| ---------- | -------- | ----------------------------------- | --------------------- |
| MIT-States | 4 (0-3)  | `mit-states-s0` .. `mit-states-s3`  | `mit-states.yml`      |
| UT-Zappos  | 3 (0-2)  | `utzappos-s0` .. `utzappos-s2`      | `ut-zappos.yml`       |
| C-GQA      | 6 (0-5)  | `cgqa-s0` .. `cgqa-s5`              | `cgqa.yml`            |

### ComPIL benchmark

The raw CompIL splits come from
[Lixsp11/CompIL](https://github.com/Lixsp11/CompIL) (`benchmarks/`).
`data/ComPIL/build_compil.py` turns them into per-task inputs
(`data/ComPIL/generated/<dataset>/task{k}/`), using the image metadata of
the PromptCCZSL benchmark. `data/ComPIL/README.md` gives the full commands.

| Dataset    | Tasks   | `--session` names                                | Config                   |
| ---------- | ------- | ------------------------------------------------ | ------------------------ |
| MIT-States | 5 (0-4) | `mit-states-compil-s0` .. `mit-states-compil-s4` | `mit-states-compil.yml`  |
| C-GQA      | 5 (0-4) | `cgqa-compil-s0` .. `cgqa-compil-s4`             | `cgqa-compil.yml`        |

### Set the data paths

By default the code reads data from `code/download_data/CCZSL_benchmark/<dataset>/`
(relative to the repository). To use another location, either symlink it
there or change `DATASET_PATHS` (PromptCCZSL) and `images_root` (ComPIL) in
`code/datasets/*_read_datasets.py`.

## PromptCCZSL benchmark

Train all sessions of one dataset. Each script runs from `code/`, so it can
be started from anywhere:

```bash
bash code/run/run_train_mit.sh        # MIT-States, sessions 0-3
bash code/run/run_train_utzappos.sh   # UT-Zappos,  sessions 0-2
bash code/run/run_train_cgqa.sh       # C-GQA,      sessions 0-4
```

Models and logs go to `code/data/model/<dataset>/session{k}/`. Session `k`
loads `session{k-1}/val_best.pt` as both the student initialisation and the
teacher.

Evaluate a session (from `code/`):

```bash
python -u test.py \
  --clip_arch model/ViT-L-14.pt \
  --yml_path config/troika/mit-states.yml \
  --num_workers 4 --seed 0 \
  --save_path data/model/mit-states/session3/ \
  --load_model data/model/mit-states/session3/val_best.pt \
  --session mit-states-s3 \
  --teacher_session mit-states-s2
```

## ComPIL benchmark

Run from `code/`. Each script takes `TASKS` (or `STEPS`) to run a subset,
for example `TASKS="2 3 4"`.

```bash
# 1. Train tasks 0-4
bash run_compil/run_train_mit_ComPIL.sh
bash run_compil/run_train_cgqa_ComPIL.sh

# 2. Main test: seen (S), unseen (U), AUC and HM per task
bash run_compil/run_test_mit_ComPIL.sh
bash run_compil/run_test_cgqa_ComPIL.sh

# 3. Forgetting table: S(k, j) for every step k and task j <= k
bash run_compil/run_fs_mit_ComPIL.sh
bash run_compil/run_fs_cgqa_ComPIL.sh

# 4. Summary tables: per-task S/U/AUC/HM, mU, mAUC and fS
python run_compil/summarize_compil.py mit
python run_compil/summarize_compil.py cgqa
```

Outputs go to `code/data/model/ComPIL/{mit,cgqa}/`.

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

## Acknowledgement

This code builds on:

* [Troika](https://github.com/bighuang624/Troika)
* [DFSP](https://github.com/Forest-art/DFSP)
* [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)
* [CompIL](https://github.com/Lixsp11/CompIL) (ComPIL benchmark splits)
