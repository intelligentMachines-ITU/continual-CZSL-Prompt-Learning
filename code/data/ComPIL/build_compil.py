"""Build per-task CSP inputs from a ComPIL benchmark split.

ComPIL ships pair lists only (train_pairs.task{k}.txt, {val,test}_pairs.seen.task{k}.txt,
{val,test}_pairs.unseen.txt), each holding only that task's pairs. This script turns them
into the layout the session readers expect, for every task k:

    <out_dir>/task{k}/metadata.t7         image records used at task k (see "Images per task" below)
    <out_dir>/task{k}/train_pairs.txt     train pairs of tasks 0..k, in task order
    <out_dir>/task{k}/val_pairs.txt       seen val pairs of tasks 0..k + composable unseen val pairs
    <out_dir>/task{k}/test_pairs.txt      seen test pairs of tasks 0..k + seen test pairs of tasks
                                          k+1.. (unseen at task k) + composable unseen test pairs
    <out_dir>/task{k}/idx/{attr,obj,pair}2idx.json
    <out_dir>/task{k}/fs/task{j}.t7       test images for fS cell (step k, task j), j <= k
    <out_dir>/summary.json                counts used for ALL_ATTR_OBJ_PAIRS / TRAIN_PAIRS

Ordering guarantees (required by session_train.py and the global index maps):
  * task k's train pairs are a prefix of task k+1's train pairs;
  * task k's attrs, objs and pairs are prefixes of task k+1's. With the default test setup every
    task uses the full pair list, so later tasks' pairs can be predicted and scored as unseen.
An unseen pair joins a task once both of its primitives have been seen in training.

Images per task (task{k}/metadata.t7). The pair lists above define the label space; the .t7
defines which images are loaded:
  * train: images of train_pairs.task{k}.txt only (no replay of earlier tasks);
  * val/test with --{val,test}_scope current: images of {val,test}_pairs.seen.task{k}.txt
    plus all unseen val/test images;
  * val/test with --{val,test}_scope accumulated (default): seen images of tasks 0..k plus all
    unseen images;
  * test also includes the seen test images of tasks k+1.. as unseen images, following the
    CompIL paper (unseen at task t = compositions of tasks t+1..T+1). --no_test_future turns
    this off and keeps the pair list growing task by task.

fS cells (task{k}/fs/task{j}.t7): same val images and the same unseen test images as task k,
but the seen test images of task j only, so the best seen accuracy gives S_{k,j}.

Example:
    python data/ComPIL/build_compil.py \
        --compil_dir data/ComPIL/c-gqa-compil \
        --t7 ".../CGQA/meta/012345metadata_compositional-split-natural.t7" \
        --out_dir data/ComPIL/generated/cgqa
"""
import argparse
import collections
import json
import os

import torch


def read_pairs(path):
    with open(path, encoding="utf-8") as f:
        pairs = [tuple(line.split()) for line in f if line.strip()]
    bad = [p for p in pairs if len(p) != 2]
    if bad:
        raise ValueError(f"{path}: expected 'attr obj' per line, got {bad[:3]}")
    return pairs


def dedup(seq):
    return list(dict.fromkeys(seq))


def write_lines(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{a} {o}" for a, o in pairs) + "\n")


def write_json(path, keys):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): i for i, k in enumerate(keys)}, f, indent=2)


def merge_t7(paths, keep_pairs):
    """Merge image records from several .t7 files, keeping only ComPIL pairs."""
    records = {}
    for p in paths:
        for r in torch.load(p):
            pair = (r["attr"], r["obj"])
            if pair not in keep_pairs or r["set"] == "NA":
                continue
            prev = records.setdefault(r["image"], r)
            if (prev["attr"], prev["obj"], prev["set"]) != (r["attr"], r["obj"], r["set"]):
                raise ValueError(f"conflicting records for {r['image']}: {prev} vs {r} ({p})")
    return list(records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compil_dir", required=True, help="ComPIL split dir, e.g. data/ComPIL/c-gqa-compil")
    parser.add_argument("--t7", required=True, nargs="+", help="one or more metadata .t7 files to merge")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--val_scope", choices=["current", "accumulated"], default="accumulated",
                        help="seen val images of task k only, or of tasks 0..k")
    parser.add_argument("--test_scope", choices=["current", "accumulated"], default="accumulated",
                        help="seen test images of task k only, or of tasks 0..k")
    parser.add_argument("--no_test_future", action="store_true",
                        help="do not count later tasks' seen test pairs as unseen at task k")
    parser.add_argument("--no_fs", action="store_true",
                        help="skip the per-cell test sets for the forgetting metric fS")
    args = parser.parse_args()

    rd = lambda name: read_pairs(os.path.join(args.compil_dir, name))
    K = args.num_tasks
    train = [rd(f"train_pairs.task{k}.txt") for k in range(K)]
    val_seen = [rd(f"val_pairs.seen.task{k}.txt") for k in range(K)]
    test_seen = [rd(f"test_pairs.seen.task{k}.txt") for k in range(K)]
    val_unseen, test_unseen = rd("val_pairs.unseen.txt"), rd("test_pairs.unseen.txt")

    for k in range(K):
        earlier = set().union(*map(set, train[:k]))
        assert not set(train[k]) & earlier, f"task {k} repeats train pairs from earlier tasks"
        assert set(val_seen[k]) <= set(train[k]), f"task {k} seen val pairs not in its train pairs"
        assert set(test_seen[k]) <= set(train[k]), f"task {k} seen test pairs not in its train pairs"
    all_seen = set().union(*map(set, train))
    assert not (set(val_unseen) | set(test_unseen)) & all_seen, "unseen pairs overlap seen pairs"

    all_pairs = all_seen | set(val_unseen) | set(test_unseen)
    records = merge_t7(args.t7, all_pairs)
    covered = {(r["attr"], r["obj"]) for r in records}
    missing = sorted(all_pairs - covered)
    if missing:
        print(f"WARNING: {len(missing)} ComPIL pairs have no images, e.g. {missing[:5]}")

    # Image split labels must agree with the pair's ComPIL role.
    role = {p: "train" for p in all_seen}
    role.update({p: "val" for p in val_unseen})
    role.update({p: "test" for p in test_unseen})
    mismatch = collections.Counter(
        (role[(r["attr"], r["obj"])], r["set"]) for r in records
        if role[(r["attr"], r["obj"])] != "train" and role[(r["attr"], r["obj"])] != r["set"])
    if mismatch:
        print(f"WARNING: unseen pairs with images outside their split: {dict(mismatch)}")

    os.makedirs(args.out_dir, exist_ok=True)
    stale = os.path.join(args.out_dir, "metadata_compil.t7")  # shared .t7 from older versions
    if os.path.exists(stale):
        os.remove(stale)

    # Pass 1: accumulated pair lists and index order for every task.
    attrs, objs, pairs = [], [], []
    tr_acc, vs_acc, ts_acc = [], [], []
    tasks = []
    for k in range(K):
        tr_acc += train[k]
        vs_acc += val_seen[k]
        ts_acc += test_seen[k]
        attrs = dedup(attrs + sorted({a for a, _ in train[k]} - set(attrs)))
        objs = dedup(objs + sorted({o for _, o in train[k]} - set(objs)))
        known_a, known_o = set(attrs), set(objs)
        composable = lambda ps: [p for p in ps if p[0] in known_a and p[1] in known_o]
        vu, tu = composable(val_unseen), composable(test_unseen)
        tr_list, vl_list = dedup(tr_acc), dedup(vs_acc + vu)
        new_pairs = set(tr_list + vl_list + ts_acc + tu) - set(pairs)
        pairs = pairs + sorted(new_pairs)
        tasks.append({"attrs": list(attrs), "objs": list(objs), "pairs": list(pairs),
                      "train": tr_list, "val": vl_list, "vs_acc": list(vs_acc),
                      "ts_acc": list(ts_acc), "vu": vu, "tu": tu})

    future = not args.no_test_future
    summary = {"val_scope": args.val_scope, "test_scope": args.test_scope,
               "test_future": future, "tasks": []}

    def select(image_pairs):
        return [r for r in records if (r["attr"], r["obj"]) in image_pairs[r["set"]]]

    def save_t7(path, recs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(recs, path + ".tmp")
        os.replace(path + ".tmp", path)  # a reader never sees a half-written file

    # Pass 2: write each task's lists, index maps and image metadata.
    for k, t in enumerate(tasks):
        future_test = dedup([p for j in range(k + 1, K) for p in test_seen[j]]) if future else []
        ts_list = dedup(t["ts_acc"] + future_test + t["tu"])
        pair_list = tasks[-1]["pairs"] if future else t["pairs"]

        task_dir = os.path.join(args.out_dir, f"task{k}")
        os.makedirs(os.path.join(task_dir, "idx"), exist_ok=True)
        write_lines(os.path.join(task_dir, "train_pairs.txt"), t["train"])
        write_lines(os.path.join(task_dir, "val_pairs.txt"), t["val"])
        write_lines(os.path.join(task_dir, "test_pairs.txt"), ts_list)
        write_json(os.path.join(task_dir, "idx", "attr2idx.json"), t["attrs"])
        write_json(os.path.join(task_dir, "idx", "obj2idx.json"), t["objs"])
        write_json(os.path.join(task_dir, "idx", "pair2idx.json"), pair_list)

        unseen_test = set(future_test) | set(t["tu"])
        image_pairs = {
            "train": set(train[k]),
            "val": set(val_seen[k] if args.val_scope == "current" else t["vs_acc"]) | set(t["vu"]),
            "test": set(test_seen[k] if args.test_scope == "current" else t["ts_acc"]) | unseen_test,
        }
        task_records = select(image_pairs)
        save_t7(os.path.join(task_dir, "metadata.t7"), task_records)

        split_counts = collections.Counter(r["set"] for r in task_records)
        info = {"task": k, "t7": f"task{k}/metadata.t7", "attrs": len(t["attrs"]),
                "objs": len(t["objs"]), "pairs": len(pair_list), "train_pairs": len(t["train"]),
                "val_pairs": len(t["val"]), "test_pairs": len(ts_list),
                "images": {sp: split_counts[sp] for sp in ("train", "val", "test")},
                "unseen_test_images": sum(1 for r in task_records
                                          if r["set"] == "test" and (r["attr"], r["obj"]) in unseen_test)}

        if not args.no_fs:
            info["fs_t7"] = {}
            for j in range(k + 1):
                cell = dict(image_pairs, test=set(test_seen[j]) | unseen_test)
                save_t7(os.path.join(task_dir, "fs", f"task{j}.t7"), select(cell))
                info["fs_t7"][str(j)] = f"task{k}/fs/task{j}.t7"

        summary["tasks"].append(info)
        print({key: v for key, v in info.items() if key != "fs_t7"})

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir} ({len(records)} image records)")


if __name__ == "__main__":
    main()
