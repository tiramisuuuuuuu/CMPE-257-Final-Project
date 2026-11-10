"""Scoring, Pareto fronts, hypervolume, plots.  Owners: Yanin + Sreya"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .pors import pr, pareto

def points_from_covs(covs, y):
    """covs: list of coverage arrays (one per rule set) -> non-dominated (prec, rec) points."""
    pts = []
    for c in covs:
        p, r = pr(c, y)
        pts.append({"prec": p, "rec": r})
    return [(d["prec"], d["rec"]) for d in pareto(pts)]

def hypervolume(points):
    """Area under the Pareto staircase in (recall, precision) space, ref (0,0). Higher = better."""
    pts = sorted(points, key=lambda p: p[1])
    area, prev_r = 0.0, 0.0
    for i, (p, r) in enumerate(pts):
        best_p = max(q[0] for q in pts[i:])
        area += (r - prev_r) * best_p
        prev_r = r
    return area

def precision_at_recall(points, target):
    ok = [p for p, r in points if r >= target]
    return max(ok) if ok else 0.0

def plot_fronts(fronts, title, path):
    """fronts: {method_name: [(prec, rec), ...]}"""
    plt.figure(figsize=(6, 5))
    for name, pts in fronts.items():
        pts = sorted(pts, key=lambda p: p[1])
        plt.step([r for _, r in pts], [p for p, _ in pts], where="pre", marker="o",
                 ms=3, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.xlim(0, 1.02); plt.ylim(0, 1.02); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

def rule_to_text(rule, names):
    return " AND ".join(names[j] for j in rule)
