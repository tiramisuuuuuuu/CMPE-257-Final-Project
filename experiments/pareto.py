from paretoset import paretoset
import pandas as pd
import numpy as np

def pareto_optimal(rule_subsets):
    """Keep only Pareto-optimal subsets, modifying rule_subsets in place."""

    if not rule_subsets:
        return pd.DataFrame(columns=["rule_subset", "precision", "recall"])

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

    # Remove non-Pareto-optimal subsets from original dictionary
    pareto_keys = set(pareto_df["rule_subset"])

    for key in list(rule_subsets.keys()):
        if key not in pareto_keys:
            del rule_subsets[key]

    return pareto_df



def pareto_optimal_np(solutions):
    """
    Keep only Pareto-optimal solutions in a NumPy array.

    Maximizes precision and recall.
    Sorts by precision descending, then recall ascending.

    Args:
        solutions: NumPy array with columns [precision, recall].

    Returns:
        Filtered NumPy array.
    """

    if solutions.size == 0:
        return solutions

    mask = paretoset(
        solutions[:, :2],
        sense=["max", "max"]
    )

    solutions = solutions[mask]

    order = np.lexsort((solutions[:, 1], -solutions[:, 0]))
    solutions = solutions[order]

    return solutions
