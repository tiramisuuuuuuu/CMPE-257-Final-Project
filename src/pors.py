"""PORS: build the Pareto front ("menu") of rule sets.  Owner: Sreya

A rule set fires if ANY rule fires -> coverage of a set = OR of rule coverages.
This is a simple version; add the paper's pruning heuristics on top and
compare runtime/quality against this one.
"""
import itertools
import numpy as np

def pr(cov, y):
    tp = (cov & y).sum(); fp = (cov & ~y).sum()
    return tp / max(tp + fp, 1), tp / max(y.sum(), 1)

def pareto(cands):
    """cands: list of dicts with 'prec','rec'. Keep the non-dominated ones."""
    cands = sorted(cands, key=lambda c: (-c["rec"], -c["prec"]))
    front, best_p = [], -1.0
    for c in cands:
        if c["prec"] > best_p:
            front.append(c); best_p = c["prec"]
    return front

def pors(rule_covs, y, max_rules=10, max_front=200):
    singles = []
    for i, c in enumerate(rule_covs):
        p, r = pr(c, y)
        singles.append({"prec": p, "rec": r, "cov": c, "ids": frozenset([i])})
    front = pareto(singles)
    for _ in range(max_rules - 1):
        new = []
        for s in front:
            for i, rc in enumerate(rule_covs):
                if i in s["ids"]:
                    continue
                u = s["cov"] | rc
                p, r = pr(u, y)
                new.append({"prec": p, "rec": r, "cov": u, "ids": s["ids"] | {i}})
        updated = pareto(front + new)
        if len(updated) > max_front:   # thin the front evenly (don't just cut off one end)
            idx = np.linspace(0, len(updated) - 1, max_front).round().astype(int)
            updated = [updated[i] for i in sorted(set(idx))]
        if {f["ids"] for f in updated} == {f["ids"] for f in front}:
            break
        front = updated
    return front

def brute_force(rule_covs, y, max_pool=15):
    """True Pareto front by trying every subset. Only for small pools!"""
    n = min(len(rule_covs), max_pool)
    cands = []
    for k in range(1, n + 1):
        for ids in itertools.combinations(range(n), k):
            u = np.logical_or.reduce([rule_covs[i] for i in ids])
            p, r = pr(u, y)
            cands.append({"prec": p, "rec": r, "cov": u, "ids": frozenset(ids)})
    return pareto(cands)
