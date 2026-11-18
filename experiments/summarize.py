# F4 + F5 + F6: turn saved runs into report tables (mean ± std) and final charts.
#
# Reads results/official/ (from official_baselines.py) and, optionally,
# a CSV of our own full-pipeline runs (columns: dataset, trial, test_hv).
#
# Run (normal venv, from the repo root):
#   python -m experiments.summarize
#   python -m experiments.summarize --ours results/ours/hv_runs.csv
#
# Saves:
#   results/tables/   main_hv, baselines_hv, crsl (.csv + .md)
#   results/figures/  hv_comparison.png, menus_sr_vs_te.png

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = ["default", "credit", "fraud", "bank"]           # paper's column order
TITLES = {"default": "Default", "credit": "Credit", "fraud": "Fraud", "bank": "Bank"}

# Published numbers: Wen & Lou (KDD '24), test HV with hvc-ss, 500 initial rules.
PAPER_HV = {
    "sr": {"default": (0.540, 0.012), "credit": (0.359, 0.013),     # Table 4
           "fraud": (0.808, 0.030), "bank": (0.593, 0.012)},
    "te": {"default": (0.524, 0.011), "credit": (0.325, 0.022),     # Table 3
           "fraud": (0.798, 0.040), "bank": (0.533, 0.015)},
}
# Table 5: test recall at a minimum precision (mean, std)
PAPER_CRSL = {
    ("default", 0.3): {"crsl": (0.765, 0.025), "pors_sr": (0.854, 0.010)},
    ("default", 0.5): {"crsl": (0.519, 0.012), "pors_sr": (0.557, 0.020)},
    ("default", 0.7): {"crsl": (0.238, 0.038), "pors_sr": (0.260, 0.040)},
    ("credit", 0.3): {"crsl": (0.555, 0.030), "pors_sr": (0.584, 0.013)},
    ("credit", 0.5): {"crsl": (0.269, 0.059), "pors_sr": (0.289, 0.017)},
    ("credit", 0.7): {"crsl": (0.038, 0.021), "pors_sr": (0.041, 0.013)},
    ("fraud", 0.3): {"crsl": (0.840, 0.038), "pors_sr": (0.851, 0.045)},
    ("fraud", 0.5): {"crsl": (0.816, 0.024), "pors_sr": (0.822, 0.035)},
    ("fraud", 0.7): {"crsl": (0.812, 0.032), "pors_sr": (0.814, 0.021)},
    ("bank", 0.3): {"crsl": (0.907, 0.015), "pors_sr": (0.951, 0.007)},
    ("bank", 0.5): {"crsl": (0.630, 0.075), "pors_sr": (0.781, 0.018)},
    ("bank", 0.7): {"crsl": (0.337, 0.044), "pors_sr": (0.264, 0.044)},
}

# Chart style (validated categorical palette, fixed order: blue, orange, aqua)
COLORS = {"Paper": "#2a78d6", "Their code": "#eb6834", "Our code": "#1baf7a",
          "SpectralRules": "#2a78d6", "TreeEns": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if not np.isnan(mean) else "-"


def mean_std(values: pd.Series) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1) and number of trials."""
    v = values.dropna()
    if len(v) == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"\n== {name} ==\n{df.to_string(index=False)}")


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def main_table(hv: pd.DataFrame, ours: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """F5: Dataset | Paper | Their code | Our code | Gap (ours vs their code)."""
    rows, numbers = [], {}
    for d in DATASETS:
        p_mean, p_std = PAPER_HV["sr"][d]
        t_mean, t_std, n = mean_std(hv.query("dataset == @d and pool == 'sr'")["test_hv"])
        o_mean, o_std, _ = (mean_std(ours.query("dataset == @d")["test_hv"])
                            if ours is not None else (np.nan, np.nan, 0))
        gap = 100 * (o_mean - t_mean) / t_mean if not np.isnan(o_mean) else np.nan
        numbers[d] = {"Paper": (p_mean, p_std), "Their code": (t_mean, t_std),
                      "Our code": (o_mean, o_std)}
        rows.append({"Dataset": TITLES[d], "Paper": fmt(p_mean, p_std),
                     "Their code": fmt(t_mean, t_std), "Trials": n,
                     "Their code vs paper": f"{100 * (t_mean - p_mean) / p_mean:+.1f}%" if n else "-",
                     "Our code": fmt(o_mean, o_std),
                     "Gap (ours vs theirs)": f"{gap:+.1f}%" if not np.isnan(gap) else "-",
                     "Why": ""})
    return pd.DataFrame(rows), numbers


def baselines_table(hv: pd.DataFrame) -> pd.DataFrame:
    """E1: does SpectralRules beat the tree-ensemble rule pool? (paper vs their code)"""
    rows = []
    for d in DATASETS:
        sr = mean_std(hv.query("dataset == @d and pool == 'sr'")["test_hv"])
        te = mean_std(hv.query("dataset == @d and pool == 'te'")["test_hv"])
        rows.append({"Dataset": TITLES[d],
                     "SpectralRules (paper)": fmt(*PAPER_HV["sr"][d]),
                     "SpectralRules (their code)": fmt(*sr[:2]),
                     "TreeEns (paper)": fmt(*PAPER_HV["te"][d]),
                     "TreeEns (their code)": fmt(*te[:2]),
                     "SR better?": ("yes" if sr[0] > te[0] else "no") if sr[2] and te[2] else "-"})
    return pd.DataFrame(rows)


def crsl_table(crsl: pd.DataFrame) -> pd.DataFrame:
    """E1: Table 5 style. Recall on test when precision must be >= a floor."""
    rows = []
    for (d, floor), paper in PAPER_CRSL.items():
        g = crsl.query("dataset == @d and prec_floor == @floor")
        c = mean_std(g.query("method == 'crsl'")["test_recall"])
        p = mean_std(g.query("method == 'pors_sr'")["test_recall"])
        rows.append({"Dataset": TITLES[d], "Min precision": floor,
                     "CRSL (paper)": fmt(*paper["crsl"]), "CRSL (their code)": fmt(*c[:2]),
                     "PORS (paper)": fmt(*paper["pors_sr"]), "PORS (their code)": fmt(*p[:2]),
                     "Winner (their code)": ("PORS" if p[0] > c[0] else "CRSL") if c[2] and p[2] else "-"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_hv_bars(numbers: dict, path: Path) -> None:
    """Grouped bars: Paper / Their code / Our code, one group per dataset."""
    series = [s for s in ["Paper", "Their code", "Our code"]
              if any(not np.isnan(numbers[d][s][0]) for d in DATASETS)]
    x = np.arange(len(DATASETS))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, s in enumerate(series):
        means = [numbers[d][s][0] for d in DATASETS]
        stds = [numbers[d][s][1] for d in DATASETS]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, means, width * 0.92, yerr=stds,
               color=COLORS[s], label=s, edgecolor="white", linewidth=1,
               error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
    ax.set_xticks(x, [TITLES[d] for d in DATASETS])
    ax.set_ylabel("Test hypervolume (higher is better)", color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_title("SpectralRules + PORS: hypervolume by dataset (mean ± std, 5 trials)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, ncol=len(series), loc="upper left", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def staircase(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keep only non-dominated points and sort by recall, for a step plot."""
    pts = points.sort_values(["recall", "precision"], ascending=[False, False])
    keep, best_p = [], -1.0
    for r, p in zip(pts["recall"], pts["precision"]):
        if p > best_p:
            keep.append((r, p))
            best_p = p
    keep.sort()
    return np.array([k[0] for k in keep]), np.array([k[1] for k in keep])


def plot_menus(fronts_dir: Path, path: Path, trial: int = 1) -> None:
    """2x2 small multiples: Pareto menu on the test set, SpectralRules vs TreeEns."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    drew_any = False
    for ax, d in zip(axes.flat, DATASETS):
        for pool, label in [("sr", "SpectralRules"), ("te", "TreeEns")]:
            f = fronts_dir / f"{d}_{pool}_{trial}.csv"
            if not f.exists():
                continue
            r, p = staircase(pd.read_csv(f))
            ax.step(r, p, where="post", color=COLORS[label], linewidth=2, label=label)
            drew_any = True
        ax.set_title(TITLES[d], color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Recall", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision", color=MUTED)
    if not drew_any:
        plt.close(fig)
        return
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper right", fontsize=9)
    fig.suptitle(f"Pareto menus on the test set (official PORS, trial {trial})",
                 x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build final tables and charts.")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--ours", type=Path, default=None,
                    help="CSV with our full-pipeline runs (dataset, trial, test_hv)")
    args = ap.parse_args()

    official = args.results / "official"
    tables, figures = args.results / "tables", args.results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    hv = pd.read_csv(official / "hv_runs.csv")
    ours = pd.read_csv(args.ours) if args.ours else None

    main_df, numbers = main_table(hv, ours)
    save_table(main_df, tables, "main_hv")
    save_table(baselines_table(hv), tables, "baselines_hv")
    if (official / "crsl_runs.csv").exists():
        save_table(crsl_table(pd.read_csv(official / "crsl_runs.csv")), tables, "crsl")

    plot_hv_bars(numbers, figures / "hv_comparison.png")
    plot_menus(official / "fronts", figures / "menus_sr_vs_te.png")
    print(f"\nTables -> {tables}/   Figures -> {figures}/")


if __name__ == "__main__":
    main()