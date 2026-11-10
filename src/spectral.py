"""SpectralRules: run SCA with a spectrum of objectives, pool the rules.  Owner: Arthi"""
from .sca import sca, coverage

DEFAULT_BETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)   # the "7 guards"

def spectral_rules(L, y, betas=DEFAULT_BETAS, **sca_kwargs):
    pool, seen = [], set()
    for b in betas:
        for rule in sca(L, y, beta=b, **sca_kwargs):
            key = coverage(L, rule).tobytes()     # dedupe rules with identical coverage
            if key not in seen:
                seen.add(key)
                pool.append(rule)
    return pool
