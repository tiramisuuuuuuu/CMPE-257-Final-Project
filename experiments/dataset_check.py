# B3 + B5: check the authors' datasets and build the dataset table for the report.
#
# For every dataset and trial, counts rows, features and % positive in the
# train / valid / test splits, then averages over the 5 trials.
#
# Run (normal venv, from the repo root):
#   python -m experiments.dataset_check
#
# Saves to results/data/:
#   dataset_splits.csv   one row per (dataset, trial, split)
#   dataset_table.csv    one row per dataset (report table)
#   dataset_table.md     same table, ready to paste into the report

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASETS = ["default", "bank", "credit", "fraud"]
SPLITS = ["train", "valid", "test"]
TRIALS = [1, 2, 3, 4, 5]
NON_FEATURE_COLS = {"id", "class"}

# Full names used in the report.
PRETTY_NAMES = {
    "default": "Default of Credit Card Clients",
    "bank": "Bank Marketing",
    "credit": "Give Me Some Credit",
    "fraud": "Credit Card Fraud (ULB)",
}


def split_stats(csv_path: Path) -> dict:
    """Row count, feature count and positive rate for one CSV file."""
    df = pd.read_csv(csv_path)
    features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return {
        "rows": len(df),
        "features": len(features),
        "positives": int(df["class"].sum()),
        "pos_rate_pct": 100 * df["class"].mean(),
        "missing_values": int(df[features].isna().sum().sum()),
    }


def collect(official_dir: Path) -> pd.DataFrame:
    """Stats for every (dataset, trial, split)."""
    rows = []
    for dataset in DATASETS:
        for trial in TRIALS:
            for split in SPLITS:
                path = official_dir / "datasets" / dataset / "data" / f"{split}_{trial}.csv"
                rows.append({"dataset": dataset, "trial": trial, "split": split,
                             **split_stats(path)})
    return pd.DataFrame(rows)


def build_report_table(splits: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset: total rows, features, overall % positive, split sizes."""
    # Total size of each trial = train + valid + test (should be the same for all trials).
    per_trial = (splits.groupby(["dataset", "trial"])
                 .agg(rows=("rows", "sum"), positives=("positives", "sum"),
                      features=("features", "first")))
    per_trial["pos_rate_pct"] = 100 * per_trial["positives"] / per_trial["rows"]

    split_rows = (splits.groupby(["dataset", "split"])["rows"].mean()
                  .unstack()[SPLITS].round().astype(int))

    table = per_trial.groupby("dataset").agg(
        rows=("rows", "mean"), features=("features", "first"),
        pos_rate_pct=("pos_rate_pct", "mean"))
    table = table.join(split_rows).loc[DATASETS].reset_index()
    table.insert(1, "name", table["dataset"].map(PRETTY_NAMES))
    table["rows"] = table["rows"].round().astype(int)
    table["pos_rate_pct"] = table["pos_rate_pct"].round(2)
    return table


def sanity_checks(splits: pd.DataFrame) -> list[str]:
    """Return a list of warnings. Empty list = everything looks fine."""
    warnings = []
    for dataset, g in splits.groupby("dataset"):
        if g["features"].nunique() != 1:
            warnings.append(f"{dataset}: feature count differs between files")
        if g.groupby("trial")["rows"].sum().nunique() != 1:
            warnings.append(f"{dataset}: total rows differ between trials")
        if g["missing_values"].sum() > 0:
            warnings.append(f"{dataset}: has {g['missing_values'].sum()} missing values")
        rates = g.groupby("split")["pos_rate_pct"].mean()
        if rates.max() - rates.min() > 0.5:
            warnings.append(f"{dataset}: positive rate differs across splits "
                            f"({rates.round(2).to_dict()}) - splits may not be stratified")
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the datasets and build the dataset table.")
    ap.add_argument("--official_dir", type=Path, default=Path("official"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = collect(args.official_dir)
    table = build_report_table(splits)

    splits.round(4).to_csv(args.out_dir / "dataset_splits.csv", index=False)
    table.to_csv(args.out_dir / "dataset_table.csv", index=False)
    (args.out_dir / "dataset_table.md").write_text(table.to_markdown(index=False) + "\n")

    print(table.to_string(index=False))
    problems = sanity_checks(splits)
    print("\nChecks: " + ("all passed" if not problems else f"{len(problems)} warning(s)"))
    for problem in problems:
        print("  WARNING:", problem)
    print(f"\nSaved to {args.out_dir}/")


if __name__ == "__main__":
    main()