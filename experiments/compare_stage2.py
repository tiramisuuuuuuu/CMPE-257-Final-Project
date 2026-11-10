"""Small-piece test: OUR PORS vs the OFFICIAL PORS, on the SAME data split and SAME rule pool.

Uses the official repo's datasets (same splits) and their pre-mined SpectralRules pool,
so the only thing that differs is the Stage-2 algorithm.

Usage (from repo root, official repo cloned into ./official with datasets unzipped):
    python -m experiments.compare_stage2 --dataset default --trials 1 2 3 4 5
"""
import argparse, time
import numpy as np
import pandas as pd
from src.pors import pors
from src.eval import points_from_covs, hypervolume

def load(official_dir, dataset, trial, pool="sr"):
    d = f"{official_dir}/datasets/{dataset}"
    tr = pd.read_csv(f"{d}/data/train_{trial}.csv")
    va = pd.read_csv(f"{d}/data/valid_{trial}.csv")
    te = pd.read_csv(f"{d}/data/test_{trial}.csv")
    rules = pd.read_csv(f"{d}/rules/{pool}_on_train_{trial}.csv")
    return tr, va, te, rules

def coverages(df, rules):
    return [df.eval(q).values.astype(bool) for q in rules.rule_content]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_dir", default="official")
    ap.add_argument("--dataset", default="default")
    ap.add_argument("--trials", type=int, nargs="+", default=[1])
    ap.add_argument("--max_front", type=int, default=50)
    ap.add_argument("--max_rules", type=int, default=20)
    args = ap.parse_args()

    rows = []
    for t in args.trials:
        tr, va, te, rules = load(args.official_dir, args.dataset, t)
        y_tr, y_te = tr["class"].values.astype(bool), te["class"].values.astype(bool)
        c_tr, c_te = coverages(tr, rules), coverages(te, rules)
        t0 = time.time()
        front = pors(c_tr, y_tr, max_rules=args.max_rules, max_front=args.max_front)
        test_covs = [np.logical_or.reduce([c_te[j] for j in s["ids"]]) for s in front]
        hv = hypervolume(points_from_covs(test_covs, y_te))
        rows.append({"trial": t, "our_test_hv": hv, "time_s": time.time() - t0})
        print(rows[-1])
    df = pd.DataFrame(rows)
    print(f"\nOUR PORS on {args.dataset}: test HV = {df.our_test_hv.mean():.4f} ± {df.our_test_hv.std():.4f}")
    print("Compare with the official PORS (hvc-ss) on the same trials.")

if __name__ == "__main__":
    main()
