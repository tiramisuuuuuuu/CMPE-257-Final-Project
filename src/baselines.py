"""Baselines that also produce a precision/recall curve.  Owner: Yanin"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .sca import sca, coverage

def cumulative_sets(covs_train, y_train):
    """Sort 'rules' by train precision, add them one by one -> list of rule sets (index lists)."""
    stats = []
    for i, c in enumerate(covs_train):
        tp = (c & y_train).sum(); fp = (c & ~y_train).sum()
        stats.append((tp / max(tp + fp, 1), i))
    order = [i for _, i in sorted(stats, reverse=True)]
    return [order[:k] for k in range(1, len(order) + 1)]

def single_sca_baseline(L_tr, y_tr, beta=1.0, **kw):
    """Plain SCA (one guard), no spectrum."""
    rules = sca(L_tr, y_tr, beta=beta, **kw)
    covs = [coverage(L_tr, r) for r in rules]
    return rules, cumulative_sets(covs, y_tr)

def tree_baseline(L_tr, y_tr, max_depth=4, min_samples_leaf=20, seed=0):
    """Decision tree on the same yes/no features; each leaf = one rule."""
    t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               class_weight="balanced", random_state=seed).fit(L_tr, y_tr)
    leaves_tr = t.apply(L_tr)
    leaf_ids = np.unique(leaves_tr)
    covs = [leaves_tr == l for l in leaf_ids]
    return t, leaf_ids, cumulative_sets(covs, y_tr)

def ripper_baseline(df_tr, y_tr, seed=0):
    """Optional: RIPPER via the wittgenstein package (single point, not a curve)."""
    import wittgenstein as lw
    d = df_tr.copy(); d["_y"] = y_tr.astype(int)
    clf = lw.RIPPER(random_state=seed)
    clf.fit(d, class_feat="_y", pos_class=1)
    return clf
