import numpy as np
from .ssf import uniform_sampling, hv_ss, hvc_ss
from .pareto import pareto_optimal


def precision_recall(cov, y):
    """Calculate precision and recall for a combined rule coverage."""

    cov = np.asarray(cov, dtype=bool)
    y = np.asarray(y, dtype=bool)

    tp = np.sum(cov & y)
    fp = np.sum(cov & ~y)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(np.sum(y), 1)

    return precision, recall


def pors(rule_covs, y, max_rounds=50, max_front=200,
         sampling="hvc_ss", k=10):
    """
    Pareto Optimal Rule Subset Search.

    Args:
        rule_covs: List of boolean coverage arrays, one per rule.
        y: Ground-truth boolean labels.
        max_rounds: Maximum number of iterations before force exiting; also caps how many rules may appear in a subset.
        max_front: Maximum number of Pareto solutions sampled per iteration.
        sampling: 'hv_ss' or 'uniform_sampling'.

    Returns:
        List of Pareto-optimal rule subsets.
    """

    rule_covs = [np.asarray(cov, dtype=bool) for cov in rule_covs]
    y = np.asarray(y, dtype=bool)

    n_rules = len(rule_covs)

    if n_rules == 0:
        return []

    # 0. Create singleton subsets for every rule
    rule_subsets = {}

    for index, cov in enumerate(rule_covs):
        precision, recall = precision_recall(cov, y)

        key = frozenset({index})

        rule_subsets[key] = {
            "indices": {index},
            "coverage": cov.copy(),
            "precision": precision,
            "recall": recall,
        }

    # 1. Reduce to Pareto-optimal singleton subsets
    sorted_pareto_front = pareto_optimal(rule_subsets)

    # Used for hvc sampling, ignore if configured for different SSF
    previous_pareto_front = None

    # 2. Iteratively expand the Pareto front
    for _ in range(max_rounds):
        # 3. Sample k subsets from the current Pareto front
        if sampling == "hvc_ss":
            selected_keys = hvc_ss(sorted_pareto_front, previous_pareto_front, k)
            previous_pareto_front = sorted_pareto_front
        elif sampling == "hv_ss":
            selected_keys = hv_ss(sorted_pareto_front, k)
        elif sampling == "uniform_sampling":
            selected_keys = uniform_sampling(sorted_pareto_front, k)
        else:
            raise ValueError(
                "sampling must be 'hv_ss' or 'uniform_sampling'"
            )

        # 4. Expand each selected subset by adding one rule
        candidates = {
            key: value.copy()
            for key, value in rule_subsets.items()
        }

        for key in selected_keys:

            subset = rule_subsets[key]

            for index in range(n_rules):

                if index in subset["indices"]:
                    continue

                new_indices = subset["indices"] | {index}
                new_key = frozenset(new_indices)

                # Skip subsets that already exist
                if new_key in candidates:
                    continue

                # Combine rule coverage using logical OR
                new_cov = subset["coverage"] | rule_covs[index]

                precision, recall = precision_recall(new_cov, y)

                candidates[new_key] = {
                    "indices": new_indices,
                    "coverage": new_cov,
                    "precision": precision,
                    "recall": recall,
                }

        # 5. Reevaluate all candidates and retain Pareto-optimal ones
        previous_keys = set(rule_subsets.keys())

        rule_subsets = candidates

        sorted_pareto_front = pareto_optimal(rule_subsets)

        current_keys = set(rule_subsets.keys())

        # 6. Stop if the Pareto front has not changed
        if current_keys == previous_keys:
            break

    # Return an array containing the rule subsets and their precision & reccall
    return [
        {
            "ids": sorted(value["indices"]),
            "prec": value["precision"],
            "rec": value["recall"],
        }
        for value in sorted(
            rule_subsets.values(),
            key=lambda x: (-x["precision"], x["recall"])
        )
    ]
