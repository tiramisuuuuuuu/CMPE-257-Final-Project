"""Full pipeline on one dataset, several seeds.

Usage (from repo root):
    python -m experiments.run --dataset default
    python -m experiments.run --dataset synthetic      # no download needed
"""
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.prep import LOADERS, Binarizer
from src.sca import coverage
from src.spectral import spectral_rules
from src.pors import pors, brute_force
from src.baselines import single_sca_baseline, tree_baseline
from src.eval import points_from_covs, hypervolume, precision_at_recall, plot_fronts, rule_to_text

def load_synthetic(n=20000, seed=0):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=10, n_informative=5,
                               weights=[0.93], flip_y=0.02, random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y.astype(bool), []

def run_once(df, y, cat, seed, args, name):
    df_tr, df_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, stratify=y, random_state=seed)
    b = Binarizer(n_bins=args.bins).fit(df_tr, cat)          # fit on TRAIN only
    L_tr, L_te, names = b.transform(df_tr), b.transform(df_te), b.names
    kw = dict(max_rules=args.max_rules, max_len=args.max_len, min_support=args.min_support)
    res, fronts = {}, {}

    # --- Ours: SpectralRules + PORS ---
    t0 = time.time()
    pool = spectral_rules(L_tr, y_tr, **kw)
    t1 = time.time()
    covs_tr = [coverage(L_tr, r) for r in pool]
    front = pors(covs_tr, y_tr, max_rules=args.pors_k)
    t2 = time.time()
    test_covs = [np.logical_or.reduce([coverage(L_te, pool[i]) for i in s["ids"]]) for s in front]
    fronts["SpectralRules + PORS"] = points_from_covs(test_covs, y_te)
    res["pool_size"] = len(pool)
    res["time_spectral_s"] = t1 - t0
    res["time_pors_s"] = t2 - t1

    # --- Brute force on the first 15 rules (sanity check for PORS) ---
    if args.brute:
        small = covs_tr[:15]
        t3 = time.time(); bf = brute_force(small, y_tr); t4 = time.time()
        pf_small = pors(small, y_tr, max_rules=15)
        res["brute_time_s"] = t4 - t3
        res["brute_hv_train"] = hypervolume([(s["prec"], s["rec"]) for s in bf])
        res["pors_small_hv_train"] = hypervolume([(s["prec"], s["rec"]) for s in pf_small])

    # --- Baseline 1: single SCA (beta = 1) ---
    rules1, sets1 = single_sca_baseline(L_tr, y_tr, beta=1.0, **kw)
    covs1 = [np.logical_or.reduce([coverage(L_te, rules1[i]) for i in s]) for s in sets1]
    if covs1:
        fronts["Single SCA"] = points_from_covs(covs1, y_te)

    # --- Baseline 2: decision tree leaves ---
    tree, leaf_ids, sets_t = tree_baseline(L_tr, y_tr, seed=seed)
    leaves_te = tree.apply(L_te)
    covs_t = [np.isin(leaves_te, leaf_ids[s]) for s in sets_t]
    fronts["Decision tree"] = points_from_covs(covs_t, y_te)

    for m, pts in fronts.items():
        res[f"{m} | HV"] = hypervolume(pts)
        res[f"{m} | P@R=0.5"] = precision_at_recall(pts, 0.5)

    if seed == args.seeds[0]:
        plot_fronts(fronts, f"{name} (test set, seed {seed})", f"results/{name}_front.png")
        with open(f"results/{name}_example_rules.txt", "w") as f:
            picks = front[::max(1, len(front)//8)]       # spread across the menu
            for s in picks:
                f.write(f"precision={s['prec']:.3f} recall={s['rec']:.3f} (train)\n")
                for i in sorted(s["ids"]):
                    f.write("   IF " + rule_to_text(pool[i], names) + "\n")
                f.write("\n")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max_rules", type=int, default=15)
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--min_support", type=int, default=10)
    ap.add_argument("--pors_k", type=int, default=10)
    ap.add_argument("--brute", action="store_true")
    args = ap.parse_args()

    df, y, cat = load_synthetic() if args.dataset == "synthetic" else LOADERS[args.dataset]()
    print(f"{args.dataset}: {len(y)} rows, {df.shape[1]} features, {y.mean():.2%} positive")
    rows = [run_once(df, y, cat, s, args, args.dataset) for s in args.seeds]
    table = pd.DataFrame(rows)
    summary = table.agg(["mean", "std"]).T
    print(summary.round(3).to_string())
    summary.to_csv(f"results/{args.dataset}_summary.csv")

if __name__ == "__main__":
    main()
