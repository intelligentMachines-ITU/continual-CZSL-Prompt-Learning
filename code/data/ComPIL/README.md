# ComPIL data

This folder holds the ComPIL continual compositional zero-shot benchmark. It contains the raw split files and the per-task inputs generated from them for Troika (PromptCCZSL). It is a copy of `csp_compil/data/ComPIL`.

## Source

The raw splits come from [Lixsp11/CompIL](https://github.com/Lixsp11/CompIL), folder `benchmarks/`, at commit `d8bc197`. They are copied unchanged into `c-gqa-compil/` and `mit-states-compil/`.

Each raw folder contains these files, with one `attr obj` pair per line:

| File | Contents |
|---|---|
| `train_pairs.task{k}.txt` | Train pairs introduced in task `k` only, not accumulated |
| `val_pairs.seen.task{k}.txt`, `test_pairs.seen.task{k}.txt` | Seen val/test pairs of task `k`, a subset of that task's train pairs |
| `val_pairs.unseen.txt`, `test_pairs.unseen.txt` | Unseen pairs, shared by all tasks |

The raw splits contain pair lists only, with no images or metadata. Both datasets have 5 tasks (`k = 0..4`).

## Generated files

`generated/` is created by [`build_compil.py`](build_compil.py) in this folder. Run it from `code/`:

```bash
# C-GQA
python data/ComPIL/build_compil.py \
  --compil_dir data/ComPIL/c-gqa-compil \
  --t7 "/home/vclab-hafiz/saranadeem_itu_pk/CCZSLS1UtZappos/code/download_data/data/CGQA/meta/012345metadata_compositional-split-natural.t7" \
  --out_dir data/ComPIL/generated/cgqa

# MIT-States: merge the four PromptCCZSL session metadata files
M="/home/vclab-hafiz/saranadeem_itu_pk/CCZSLS1UtZappos/code/download_data/data copy/mit-states"
python data/ComPIL/build_compil.py \
  --compil_dir data/ComPIL/mit-states-compil \
  --t7 "$M/session 0/metadata_compositional-split-natural.t7" \
       "$M/session 1/1metadata_compositional-split-natural.t7" \
       "$M/session 2/2metadata_compositional-split-natural.t7" \
       "$M/session 3/3metadata_compositional-split-natural.t7" \
  --out_dir data/ComPIL/generated/mit-states
```

The script needs `torch`; use the `troika` conda env. Running it again overwrites `generated/<dataset>/`.

Default settings, which follow the CompIL paper's evaluation for test:

| Flag | Default | Effect |
|---|---|---|
| `--val_scope` | `accumulated` | val seen images of tasks `0..k` (`current`: task `k` only) |
| `--test_scope` | `accumulated` | test seen images of tasks `0..k` (`current`: task `k` only) |
| `--no_test_future` | off | when off, later tasks' seen test pairs count as unseen at task `k` |
| `--no_fs` | off | when off, also writes the per-cell test sets for the forgetting metric fS |

### Layout

```
generated/<dataset>/
  summary.json            per-task counts and .t7 paths, read by the session config
  task{k}/
    metadata.t7           image records (image, attr, obj, set) loaded at task k
                          (replaces the shared metadata_compil.t7 of older versions)
    train_pairs.txt       train pairs of tasks 0..k
    val_pairs.txt         seen val pairs of tasks 0..k + unseen val pairs
    test_pairs.txt        seen test pairs of tasks 0..k + seen test pairs of tasks k+1..4
                          (unseen at task k) + fixed unseen test pairs
    idx/attr2idx.json     attribute -> index
    idx/obj2idx.json      object -> index
    idx/pair2idx.json     "('attr', 'obj')" -> index (all pairs, same at every task)
    fs/task{j}.t7         test images for fS cell (step k, task j), j = 0..k
```

### How each file is built

1. **Checks.** The script stops with an error if any of these fail:
   - A task's train pairs repeat pairs from earlier tasks.
   - A task's seen val/test pairs are not part of that task's train pairs.
   - An unseen pair is also a seen pair.
2. **Image pool.** The script merges the `--t7` files and keeps only records whose pair appears in a ComPIL split, dropping records whose `set` is `NA`. Records are deduplicated by image name, and two different labels for the same image are an error. The script warns if a ComPIL pair has no images, or if an unseen pair has images outside its own split.
3. **Pair lists for task `k`.**
   - `train_pairs.txt` concatenates `train_pairs.task0..k` in task order.
   - `val_pairs.txt` concatenates the seen val lists for tasks `0..k`, then adds the unseen val pairs.
   - `test_pairs.txt` concatenates, in this order: the seen test pairs of tasks `0..k`, the seen test pairs of tasks `k+1..4` (unseen at task `k`), and the fixed unseen test pairs. The total stays the same at every task; only the split between seen and unseen moves as tasks are learned:

     | | Seen `0..k` | Tasks `k+1..4` (as unseen) | Fixed unseen | Total |
     |---|---|---|---|---|
     | C-GQA task 0 | 179 | 709 | 923 | 1,811 |
     | C-GQA task 2 | 525 | 363 | 923 | 1,811 |
     | C-GQA task 4 | 888 | 0 (no later tasks) | 923 | 1,811 |
     | MIT-States task 0 | 91 | 309 | 400 | 800 |
     | MIT-States task 2 | 250 | 150 | 400 | 800 |
     | MIT-States task 4 | 400 | 0 | 400 | 800 |

4. **Index maps.** Attributes and objects are listed in the order they first appear in training. Every task uses the full pair list, in the same order, so a pair has the same index at every task.
5. **`task{k}/metadata.t7`.** This file picks the images loaded at task `k`. The pair lists from step 3 only set the label space.
   - Train: images of `train_pairs.task{k}.txt` only. Earlier tasks' images are not replayed.
   - Val: seen val images of tasks `0..k` plus all unseen val images.
   - Test: seen test images of tasks `0..k`, plus the seen test images of tasks `k+1..4` (scored as unseen), plus all unseen test images. This is the full test split at every task.
6. **`task{k}/fs/task{j}.t7`.** The same as `task{k}/metadata.t7`, except that the test set keeps only task `j`'s seen test images. Its unseen test images, candidate pairs and seen pairs are the same as task `k`'s main test.

### What is evaluated

`test.py` lets the model choose among `train_pairs.txt` ∪ `val_pairs.txt` (val) or `train_pairs.txt` ∪ `test_pairs.txt` (test). It counts a pair as seen only if it is in `train_pairs.txt`. At task `k` that gives:

| | Candidates | Seen images | Unseen images |
|---|---|---|---|
| Val | train pairs of tasks `0..k` + unseen val pairs | seen val of tasks `0..k` | fixed unseen val |
| Test | train pairs of tasks `0..k` + seen test pairs of tasks `k+1..4` + unseen test pairs | seen test of tasks `0..k` | seen test of tasks `k+1..4` + fixed unseen test |

This matches the CompIL paper's definitions: at task t the seen compositions are those of tasks 1..t, and the unseen ones are those of tasks t+1..T plus the held-out unseen set. So `best_seen`, `best_unseen` and `AUC` in the test results are the paper's Sₜ, Uₜ and AUCₜ. fS uses S_{k,j}, the best seen accuracy on task `j` after step `k`, which comes from the fS cells. The paper doesn't state its candidate set at test time, so using all of these pairs as candidates is our reading of its definitions.

### Ordering guarantees

The training code depends on these:

- **Task `k`'s train pairs are a prefix of task `k+1`'s.** `session_train.py` treats the first `TRAIN_PAIRS[teacher_session]` logits as the old pairs for distillation. Training targets are positions in the final task's `train_pairs.txt`, so at task `k` every target falls among the new pairs.
- **Attribute, object and pair indices never change between tasks.** Labels come from the final task's index maps (`selected_session_g_pair2_idx`), and the student copies the teacher's embedding rows by position.

## Counts

| Dataset | Task | Attrs | Objs | Pairs | Train pairs | Val pairs | Test pairs |
|---|---|---|---|---|---|---|---|
| C-GQA | 0 | 413 | 674 | 7767 | 1377 | 1470 | 1811 |
| C-GQA | 1 | 413 | 674 | 7767 | 2383 | 1657 | 1811 |
| C-GQA | 2 | 413 | 674 | 7767 | 3463 | 1884 | 1811 |
| C-GQA | 3 | 413 | 674 | 7767 | 4529 | 2092 | 1811 |
| C-GQA | 4 | 413 | 674 | 7767 | 5592 | 2292 | 1811 |
| MIT-States | 0 | 115 | 245 | 1962 | 290 | 377 | 800 |
| MIT-States | 1 | 115 | 245 | 1962 | 536 | 424 | 800 |
| MIT-States | 2 | 115 | 245 | 1962 | 770 | 477 | 800 |
| MIT-States | 3 | 115 | 245 | 1962 | 1018 | 539 | 800 |
| MIT-States | 4 | 115 | 245 | 1962 | 1262 | 600 | 800 |

Every attribute and object already appears in task 0, so only the train and val pairs grow from task to task.

Images loaded at each task:

| Dataset | Task | Train images | Val images | Test images (unseen) | fS seen test images, task 0..k |
|---|---|---|---|---|---|
| C-GQA | 0 | 5,504 | 4,311 | 5,098 (4,310) | 788 |
| C-GQA | 1 | 4,811 | 4,964 | 5,098 (3,445) | 788, 865 |
| C-GQA | 2 | 5,488 | 5,714 | 5,098 (2,586) | 788, 865, 859 |
| C-GQA | 3 | 5,028 | 6,430 | 5,098 (1,856) | 788, 865, 859, 730 |
| C-GQA | 4 | 6,089 | 7,280 | 5,098 (1,144) | 788, 865, 859, 730, 712 |
| MIT-States | 0 | 7,061 | 9,070 | 12,995 (12,469) | 526 |
| MIT-States | 1 | 5,733 | 9,340 | 12,995 (11,905) | 526, 564 |
| MIT-States | 2 | 5,842 | 9,650 | 12,995 (11,496) | 526, 564, 409 |
| MIT-States | 3 | 5,994 | 10,041 | 12,995 (11,045) | 526, 564, 409, 451 |
| MIT-States | 4 | 5,708 | 10,420 | 12,995 (10,615) | 526, 564, 409, 451, 430 |

The train images over all tasks add up to the standard CZSL train splits (C-GQA 26,920, MIT-States 30,338), so no image is used in more than one task. At task 4, val covers the full val split, and the unseen test images are only the fixed unseen set (C-GQA 1,144, MIT-States 10,615).

## Using it

[`datasets/compil_read_datasets.py`](../../datasets/compil_read_datasets.py) reads `generated/<dataset>/` and provides the same session config as the PromptCCZSL readers. Two thin modules wrap it:

| `--dataset` | Module | Sessions | Image root |
|---|---|---|---|
| `cgqa-compil` | `datasets/cgqa_compil_read_datasets.py` | `cgqa-compil-s0` .. `cgqa-compil-s4` | `.../download_data/data/CGQA/` |
| `mit-states-compil` | `datasets/mit_compil_read_datasets.py` | `mit-states-compil-s0` .. `mit-states-compil-s4` | `.../download_data/data copy/mit-states/` |

Each fS cell is an extra session `<dataset>-s{k}-fs{j}` (j ≤ k), identical to `<dataset>-s{k}` except for its test images.

Hyperparameters come from `config/troika/{cgqa,mit-states}-compil.yml`, which set no `session`/`teacher_session`, so the scripts' `--session` flags apply.

Images load from `<image root>/images/`. Scripts (run from anywhere; they `cd` to `code/`):

| Script | What it does |
|---|---|
| `run_compil/run_train_{cgqa,mit}_ComPIL.sh` | trains task 0 with `train.py`, then tasks 1–4 with `session_train.py` |
| `run_compil/run_test_{cgqa,mit}_ComPIL.sh` | main test of each task's `val_best.pt`; writes `session{k}/val_best.closed.json` |
| `run_compil/run_fs_{cgqa,mit}_ComPIL.sh` | the 15 fS cells per dataset |
| `run_compil/summarize_compil.py {cgqa,mit}` | per-task S, U, AUC and HM, plus mU, mAUC, the S_{k,j} table and fS |
