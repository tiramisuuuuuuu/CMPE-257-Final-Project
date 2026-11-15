from paretoset import paretoset
import pandas as pd

def pareto_optimal(rule_subsets):
    # extract objective values while preserving original keys
    objectives = pd.DataFrame.from_dict(
        {
            key: {
                "precision": value["precision"],
                "recall": value["recall"],
            }
            for key, value in rule_subsets.items()
        },
        orient="index"
    )

    objectives.index.name = "rule_subset"
    objectives = objectives.reset_index()

    mask = paretoset(
        objectives[["precision", "recall"]],
        sense=["max", "max"]
    )

    pareto_df = objectives[mask].sort_values(
        by=["precision", "recall"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # extract Pareto-optimal keys
    pareto_keys = set(pareto_df["rule_subset"])

    # modify original dictionary in place
    for key in list(rule_subsets.keys()):
        if key not in pareto_keys:
            del rule_subsets[key]

    return pareto_df


def precision_recall(cov, y):
    tp = (cov & y).sum(); fp = (cov & ~y).sum()
    return tp / max(tp + fp, 1), tp / max(y.sum(), 1)


def pors(rule_covs, y, max_rules=10, max_front=200):
    # 0. create singleton sets for every rule
    rule_subsets = {
        frozenset({index}): {
            "indices": {index},
            "coverage": cov,
            "precision": precision_recall(cov, y[index])[0],
            "recall": precision_recall(cov, y[index])[1],
        }
        for index, cov in enumerate(rule_covs)
    }


    # 1. reduce sets to only include pareto optimal sets, sets which are not dominated in both precision and recall by any other set
    sorted_pareto_front = pareto_optimal(rule_subsets)

    # 3. use ssf to sample only k sets from all of our sets

    # 4. expand those k sets by adding one rule from our initial pool of rules, test every single rule on the k subsets

    # 5. reevaluate all sets and keep only pareto optimal

    # 6. if pareto front hasn't changed exit, otherwise repeat 3-6

    """ example output: [
            {"ids": [2,5], "prec": 0.2, "rec": 0.75},
            {"ids": [1,2], "prec": 0.5, "rec": 0.63},
        ]
    """
    return []
