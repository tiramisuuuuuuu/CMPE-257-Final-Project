"""Run the OFFICIAL PORS (Wen & Lou, KDD 2024) on one dataset/trial.
This script only CALLS their code from ./official (not copied).
Source: https://github.com/ChengyaoWen/Pareto-Optimal-Rule-Subset-Selection

Usage (in venv-official, from repo root):
    # their rules (Test 1 reference):
    python experiments/official_stage2.py --dataset default --trials 1
    # OUR rules through THEIR PORS (Test 2):
    python experiments/official_stage2.py --dataset default --trials 1 --rules ours
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "official", "pors"))


def rules_path(rules, dataset, i):
    if rules == "official":
        return os.path.join(ROOT, "official", "datasets", dataset, "rules", f"sr_on_train_{i}.csv")
    if rules == "ours":
        return os.path.join(ROOT, "results", "rules", dataset, f"ours_on_train_{i}.csv")
    return rules.format(dataset=dataset, trial=i)   # custom path template


def main():
    from pors import init_solutions, pors
    from utils.data_utils import data_preprocess, rule_predict
    from utils.moo_utils import eval_solutions

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="default")
    ap.add_argument("--trials", type=int, nargs="+", default=[1])
    ap.add_argument("--rules", default="official", help="official | ours | path template")
    ap.add_argument("--n_process", type=int, default=4)
    args = ap.parse_args()

    hvs = []
    for i in args.trials:
        d = os.path.join(ROOT, "official", "datasets", args.dataset)
        tr = pd.read_csv(f"{d}/data/train_{i}.csv")
        va = pd.read_csv(f"{d}/data/valid_{i}.csv")
        te = pd.read_csv(f"{d}/data/test_{i}.csv")
        r = pd.read_csv(rules_path(args.rules, args.dataset, i))
        t0 = time.time()
        ids, trc, trp, n = data_preprocess(tr, rule_predict(r, tr))
        _, vac, vap, _ = data_preprocess(va, rule_predict(r, va))
        _, tec, tep, _ = data_preprocess(te, rule_predict(r, te))
        sol, cov = init_solutions(ids, trc, trp, vac)
        opt, _ = pors(sol, cov, trp, vap, dataset_size=n, ssf_method="hvc-ss",
                      ssf_param={}, k=10, max_round=50, n_process=args.n_process)
        hv = eval_solutions(opt, ids, tec, tep)
        hvs.append(hv)
        print({"trial": i, "rules": args.rules, "n_rules": len(r), "test_hv": round(float(hv), 4),
               "time_s": round(time.time() - t0, 1)})

    print(f"\nTHEIR PORS on {args.dataset} with {args.rules} rules: "
          f"test HV = {np.mean(hvs):.4f} ± {np.std(hvs, ddof=1) if len(hvs) > 1 else 0:.4f}")


if __name__ == "__main__":   # required on Mac: stops helper processes from re-running everything
    main()
