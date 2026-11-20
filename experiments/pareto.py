from paretoset import paretoset
import pandas as pd

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



def pareto_optimal_df(df):
    """
    Keep only Pareto-optimal solutions, modifying the DataFrame in place.

    Expects precision and recall columns, maximizing both.
    """

    if df.empty:
        return df

    mask = paretoset(
        df[["precision", "recall"]],
        sense=["max", "max"]
    )

    df.drop(index=df.index[~mask], inplace=True)

    df.sort_values(
        by=["precision", "recall"],
        ascending=[False, True],
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

    return df
