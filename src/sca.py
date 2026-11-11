"""Sequential Covering Algorithm (one "guard").  Owner: Arthi

Builds rules one at a time. Each rule = AND of literals (column indices of L).
The objective is F-beta: small beta -> cares about precision,
large beta -> cares about recall. NOTE: check the paper's exact objective /
stopping criteria and adjust; document any differences in the report.
"""
import numpy as np

def coverage(L, rule):
    """bool array: which rows a rule fires on (empty rule = all rows)."""
    if len(rule) == 0:
        return np.ones(L.shape[0], dtype=bool)
    return L[:, rule].all(axis=1)

def fbeta(tp, fp, P, beta):
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    b2 = beta ** 2
    return (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)

def grow_rule(L, y, active, beta, max_len=3, min_support=10):
    """Greedily add the literal that most improves F-beta on active rows."""
    rule = []
    cov = active.copy()
    P = (y & active).sum()
    best = -1.0
    for _ in range(max_len):
        Lc, yc = L[cov], y[cov]
        tp = Lc[yc].sum(axis=0)
        fp = Lc[~yc].sum(axis=0)
        s = fbeta(tp, fp, P, beta)
        s[tp < min_support] = -1          # ignore tiny rules
        s[rule] = -1                      # don't reuse a literal
        j = int(np.argmax(s))
        if s[j] <= best:
            break
        best = s[j]
        rule.append(j)
        cov = cov & L[:, j]
    return rule, best

def sca(L, y, beta=1.0, max_rules=20, max_len=3, min_support=10, min_precision=0.0):
    """Run one guard. Covered positives are removed after each rule."""
    active = np.ones(len(y), dtype=bool)
    rules = []
    for _ in range(max_rules):
        if (y & active).sum() < min_support:
            break
        rule, s = grow_rule(L, y, active, beta, max_len, min_support)
        if not rule:
            break
        cov = coverage(L, rule)
        tp = (cov & y).sum(); fp = (cov & ~y).sum()
        if tp / max(tp + fp, 1) < min_precision:
            break
        rules.append(rule)
        active = active & ~cov       
    return rules
