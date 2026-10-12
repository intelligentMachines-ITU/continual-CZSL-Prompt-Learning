"""Summarize Troika results on a ComPIL split: per-task S, U, AUC, HM, the averages mU / mAUC,
and the forgetting table S_{k,j} with fS (CompIL paper, Sec. "Evaluation").

Reads the JSON files test.py writes next to the model it loads:
    <out>/session{k}/<model>.closed.json                      main test  (run_test_*_ComPIL.sh)
    <out>/fs/s{k}_task{j}.closed.json                       fS cells    (run_fs_*_ComPIL.sh)

Usage (from code/): python run_compil/summarize_compil.py {cgqa,mit} [--model val_best]
"""
import argparse
import json
import os

NUM_TASKS = 5


def test_stats(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["test"]


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["cgqa", "mit"])
    parser.add_argument("--model", default="val_best", help="model file name without .pt")
    parser.add_argument("--root", default="data/model/ComPIL")
    args = parser.parse_args()
    out = os.path.join(args.root, args.dataset)

    print(f"## {args.dataset}: main test (seen = tasks 0..k, unseen = tasks k+1.. + fixed unseen)\n")
    print("| Task | S | U | AUC | HM |")
    print("|---|---|---|---|---|")
    main_stats = []
    for k in range(NUM_TASKS):
        st = test_stats(os.path.join(out, f"session{k}", f"{args.model}.closed.json"))
        main_stats.append(st)
        get = lambda key: st[key] if st else None
        print(f"| {k} | {pct(get('best_seen'))} | {pct(get('best_unseen'))} | {pct(get('AUC'))} | {pct(get('best_hm'))} |")
    done = [st for st in main_stats if st]
    if len(done) == NUM_TASKS:
        mU = sum(st["best_unseen"] for st in done) / NUM_TASKS
        mAUC = sum(st["AUC"] for st in done) / NUM_TASKS
        print(f"\nmU = {pct(mU)}, mAUC = {pct(mAUC)}; final S / U / AUC = task {NUM_TASKS - 1} row")
    else:
        print(f"\nmU / mAUC need all {NUM_TASKS} tasks ({len(done)} found)")

    print(f"\n## {args.dataset}: S_(k,j) = best seen accuracy on task j after step k\n")
    print("| Step k | " + " | ".join(f"task {j}" for j in range(NUM_TASKS)) + " |")
    print("|---|" + "---|" * NUM_TASKS)
    S = {}
    for k in range(NUM_TASKS):
        cells = []
        for j in range(NUM_TASKS):
            st = test_stats(os.path.join(out, "fs", f"s{k}_task{j}.closed.json")) if j <= k else None
            if st:
                S[k, j] = st["best_seen"]
            cells.append(pct(S.get((k, j))) if j <= k else "")
        print(f"| {k} | " + " | ".join(cells) + " |")
    last = NUM_TASKS - 1
    if all((j, j) in S and (last, j) in S for j in range(NUM_TASKS)):
        fS = sum(S[j, j] - S[last, j] for j in range(NUM_TASKS)) / NUM_TASKS
        print(f"\nfS = mean_j (S_(j,j) - S_({last},j)) = {pct(fS)}")
    else:
        print(f"\nfS needs the diagonal and the step-{last} row of the table")


if __name__ == "__main__":
    main()
