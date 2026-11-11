"""C8: run OUR SpectralRules on the official train splits and save the rules in the
OFFICIAL rule-file format (rule_id, rule_content as a pandas query string), so the
official PORS can read them (Test 2).

Usage (in venv, from repo root):
    python -m experiments.make_rules --dataset default --trials 1
Output: results/rules/<dataset>/ours_on_train_<trial>.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd
from src.prep import Binarizer
from src.spectral import PAPER_BETAS
from src.sca import sca

def literal_to_query(spec):
    col, op, v = spec
    c = f"`{col}`"
    if op == "isna":
        return f"({c} != {c})"                 # NaN != NaN is True -> "is missing"
    if op == "==":
        return f"({c} == {v!r})" if isinstance(v, str) else f"({c} == {float(v)!r})"
    return f"({c} {op} {float(v)!r})"

def rule_to_query(rule, specs):
    return " and ".join(literal_to_query(specs[j]) for j in rule)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_dir", default="official")
    ap.add_argument("--dataset", default="default")
    ap.add_argument("--trials", type=int, nargs="+", default=[1])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=6)
    ap.add_argument("--min_support", type=int, default=10)
    args = ap.parse_args()

    out_dir = f"results/rules/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    for t in args.trials:
        tr = pd.read_csv(f"{args.official_dir}/datasets/{args.dataset}/data/train_{t}.csv")
        y = tr["class"].values.astype(bool)
        X = tr.drop(columns=["class", "id"])
        b = Binarizer(n_bins=args.bins).fit(X)
        L = b.transform(X)

        t0 = time.time()
        rows = []
        per_beta = int(np.ceil(args.n / len(PAPER_BETAS)))
        seen = set()
        for bi, beta in enumerate(PAPER_BETAS):   # same loop as spectral_rules, but keeps beta index for the id
            for ri, rule in enumerate(sca(L, y, beta=beta, max_rules=per_beta,
                                          max_len=args.max_len, min_support=args.min_support)):
                key = L[:, rule].all(axis=1).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                rows.append([f"trial{bi}_{ri}_{len(rule)}", rule_to_query(rule, b.specs)])
        out = f"{out_dir}/ours_on_train_{t}.csv"
        pd.DataFrame(rows, columns=["rule_id", "rule_content"]).to_csv(out, index=False)
        print({"trial": t, "n_rules": len(rows), "literals": L.shape[1],
               "time_s": round(time.time() - t0, 1), "saved": out})

if __name__ == "__main__":
    main()
