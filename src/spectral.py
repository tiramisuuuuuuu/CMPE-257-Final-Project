"""SpectralRules (paper Alg. 2): run SCA once per beta, pool the rules.  Owner: Arthi

Paper settings (Wen & Lou, KDD 2024, Sec. 4.1 + 5.2):
  B = {0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.40, 0.60, 0.80}
  n = 500 rules total -> each beta makes at most ceil(n / |B|) = 50 rules
  max rule length = 6
"""
import math
from .sca import sca, coverage

PAPER_BETAS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.40, 0.60, 0.80)


def spectral_rules(L, y, betas=PAPER_BETAS, n=500, max_len=6, **sca_kwargs):
    sca_kwargs.pop("max_rules", None)          # rules per beta comes from n, as in the paper
    per_beta = math.ceil(n / len(betas))       # Alg. 2 line 4
    pool, seen = [], set()
    for b in betas:
        for rule in sca(L, y, beta=b, max_rules=per_beta, max_len=max_len, **sca_kwargs):
            key = coverage(L, rule).tobytes()  # drop exact duplicates (same rows covered)
            if key not in seen:
                seen.add(key)
                pool.append(rule)
    return pool
