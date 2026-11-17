# F1 + E1: run the authors' official code to get their numbers and the baselines.
# Only CALLS their code in ./official (nothing is copied).
# Source: https://github.com/ChengyaoWen/Pareto-Optimal-Rule-Subset-Selection
#
# Experiments:
#   hv    rule pool -> official PORS (hvc-ss) -> test hypervolume (HV)
#         pools: sr = SpectralRules (paper's method), te = tree ensemble (baseline)
#         both cut to 500 rules, same as the paper (Tables 3-4)
#   crsl  CRSL baseline (paper Table 5): best test recall when precision
#         must be >= 0.3 / 0.5 / 0.7. Compares CRSL vs PORS.
#
# Run (venv-official, from the repo root):
#   python -m experiments.official_baselines hv --datasets default --trials 1
#   python -m experiments.official_baselines hv
#   python -m experiments.official_baselines crsl
#
# Saves after every run and skips runs already saved, so you can stop and restart.
# Output: results/official/ (hv_runs.csv, crsl_runs.csv, fronts/*.csv)

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "official"
OUT_DIR = ROOT / "results" / "official"

DATASETS = ["default", "bank", "credit", "fraud"]
TRIALS = [1, 2, 3, 4, 5]
POOLS = ["sr", "te"]
PREC_FLOORS = [0.3, 0.5, 0.7]

# Settings used in the paper / official experiment.py
MAX_RULES = 500
K = 10
MAX_ROUND = 50


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def import_official():
    """Import the official modules (they use flat imports, so add their folder)."""
    sys.path.insert(0, str(OFFICIAL / "pors"))
    from crsl import greedy_opt
    from pors import init_solutions, pors
    from utils.data_utils import data_preprocess, rule_predict
    from utils.moo_utils import eval_single_solution, eval_solutions
    return dict(greedy_opt=greedy_opt, init_solutions=init_solutions, pors=pors,
                data_preprocess=data_preprocess, rule_predict=rule_predict,
                eval_single_solution=eval_single_solution, eval_solutions=eval_solutions)


def as_object_array(items: list) -> np.ndarray:
    """Wrap a list in a 1-D numpy object array.

    The official greedy_opt (CRSL) calls np.copy() on a list of (BitMap, BitMap)
    pairs. Old numpy turned that into an object array; numpy >= 1.24 raises
    "inhomogeneous shape" instead. Passing an object array avoids the crash
    without changing their code.
    """
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def load_splits(dataset: str, trial: int) -> dict[str, pd.DataFrame]:
    data_dir = OFFICIAL / "datasets" / dataset / "data"
    return {s: pd.read_csv(data_dir / f"{s}_{trial}.csv") for s in ("train", "valid", "test")}


def load_rules(dataset: str, pool: str, trial: int) -> pd.DataFrame:
    return pd.read_csv(OFFICIAL / "datasets" / dataset / "rules" / f"{pool}_on_train_{trial}.csv")


def coverages(off, rules: pd.DataFrame, splits: dict, tag: str | None = None):
    """Which rows each rule covers, on train / valid / test (official format)."""
    out = {}
    for name, df in splits.items():
        ids, cov, pos, size = off["data_preprocess"](df, off["rule_predict"](rules, df, tag=tag))
        out[name] = dict(ids=ids, cov=cov, pos=pos, size=size)
    return out


def limit_pool(rule_ids: list[str], pool: str) -> list[str]:
    """Keep 500 rules, the same way the official experiment.py does."""
    if pool == "sr":   # ids look like trialX_Y_Z; keep Y < 50 (50 rules x 10 betas)
        return [r for r in rule_ids if int(r.split("_")[1]) < MAX_RULES // 10]
    return rule_ids[:MAX_RULES]


# --------------------------------------------------------------------------- #
# Saving / resuming
# --------------------------------------------------------------------------- #
def already_done(csv_path: Path, **keys) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    mask = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        mask &= df[k].astype(str) == str(v)
    return bool(mask.any())


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def run_pors(off, cov: dict, rule_ids: list[str], n_process: int):
    """Official PORS with the paper's settings (hvc-ss, k=10, 50 rounds)."""
    solutions, cov_list = off["init_solutions"](rule_ids, cov["train"]["cov"],
                                                cov["train"]["pos"], cov["valid"]["cov"])
    front, _ = off["pors"](solutions, cov_list, cov["train"]["pos"], cov["valid"]["pos"],
                           dataset_size=cov["train"]["size"], ssf_method="hvc-ss",
                           ssf_param={}, k=K, max_round=MAX_ROUND, n_process=n_process)
    return front


def hv_experiment(off, datasets, trials, pools, n_process: int) -> None:
    out_csv = OUT_DIR / "hv_runs.csv"
    for dataset in datasets:
        for trial in trials:
            todo = [p for p in pools if not already_done(out_csv, dataset=dataset, pool=p, trial=trial)]
            if not todo:
                continue
            splits = load_splits(dataset, trial)
            for pool in todo:
                start = time.time()
                cov = coverages(off, load_rules(dataset, pool, trial), splits)
                rule_ids = limit_pool(cov["train"]["ids"], pool)
                front = run_pors(off, cov, rule_ids, n_process)
                hv = off["eval_solutions"](front, rule_ids, cov["test"]["cov"], cov["test"]["pos"])
                seconds = time.time() - start

                # test (precision, recall) of every rule set on the menu -> for plots
                points = [off["eval_single_solution"](s, rule_ids, cov["test"]["cov"], cov["test"]["pos"])
                          for s in front]
                front_csv = OUT_DIR / "fronts" / f"{dataset}_{pool}_{trial}.csv"
                front_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(points, columns=["precision", "recall"]).to_csv(front_csv, index=False)

                row = dict(dataset=dataset, pool=pool, trial=trial, n_rules=len(rule_ids),
                           menu_size=len(front), test_hv=round(float(hv), 4),
                           time_s=round(seconds, 1))
                append_rows(out_csv, [row])
                print(row, flush=True)


def best_recall_at_precision(off, front, rule_ids, cov, floor: float) -> float:
    """Pick the rule set with the highest VALID recall whose VALID precision >= floor,
    then report its TEST recall (0 if no rule set reaches the floor)."""
    best, best_recall = None, -1.0
    for s in front:
        prec, rec = off["eval_single_solution"](s, rule_ids, cov["valid"]["cov"], cov["valid"]["pos"])
        if prec >= floor and rec > best_recall:
            best, best_recall = s, rec
    if best is None:
        return 0.0
    return off["eval_single_solution"](best, rule_ids, cov["test"]["cov"], cov["test"]["pos"])[1]


def crsl_experiment(off, datasets, trials, n_process: int) -> None:
    out_csv = OUT_DIR / "crsl_runs.csv"
    for dataset in datasets:
        for trial in trials:
            if already_done(out_csv, dataset=dataset, trial=trial):
                continue
            splits = load_splits(dataset, trial)
            rows = []

            # PORS on SpectralRules: one menu, then pick a point for each floor
            start = time.time()
            cov = coverages(off, load_rules(dataset, "sr", trial), splits)
            front = run_pors(off, cov, cov["train"]["ids"], n_process)
            for floor in PREC_FLOORS:
                rec = best_recall_at_precision(off, front, cov["train"]["ids"], cov, floor)
                rows.append(dict(dataset=dataset, method="pors_sr", trial=trial,
                                 prec_floor=floor, test_recall=round(float(rec), 4)))
            pors_time = time.time() - start

            # CRSL: a separate rule pool and a separate greedy search per floor
            crsl_rules = load_rules(dataset, "crsl", trial)
            for floor in PREC_FLOORS:
                cov = coverages(off, crsl_rules, splits, tag=str(floor))
                rule_ids = cov["train"]["ids"][:MAX_RULES]
                solutions, cov_list = off["init_solutions"](rule_ids, cov["train"]["cov"],
                                                            cov["train"]["pos"], cov["valid"]["cov"])
                best = off["greedy_opt"](solutions, as_object_array(cov_list), cov["train"]["pos"],
                                         cov["valid"]["pos"], K, floor, MAX_ROUND, n_process)
                _, rec = off["eval_single_solution"](best, rule_ids, cov["test"]["cov"], cov["test"]["pos"])
                rows.append(dict(dataset=dataset, method="crsl", trial=trial,
                                 prec_floor=floor, test_recall=round(float(rec), 4)))

            append_rows(out_csv, rows)
            print(f"{dataset} trial {trial}: done (PORS part {pors_time:.0f}s)", flush=True)
            for r in rows:
                print("  ", r, flush=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the official code (F1 + E1).")
    ap.add_argument("experiment", choices=["hv", "crsl"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--trials", nargs="+", type=int, default=TRIALS)
    ap.add_argument("--pools", nargs="+", default=POOLS, choices=POOLS)
    ap.add_argument("--n_process", type=int, default=4)
    args = ap.parse_args()

    off = import_official()
    if args.experiment == "hv":
        hv_experiment(off, args.datasets, args.trials, args.pools, args.n_process)
    else:
        crsl_experiment(off, args.datasets, args.trials, args.n_process)


if __name__ == "__main__":   # needed on Mac/Windows: official PORS uses multiprocessing
    main()