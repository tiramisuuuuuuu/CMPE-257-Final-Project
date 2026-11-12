def pors():
    # 0. create singleton sets for every rule

    # 1. reduce sets to only include pareto optimal sets, sets which are not dominated in both precision and recall by any other set

    # 3. use ssf to sample only k sets from all of our sets

    # 4. expand those k sets by adding one rule from our initial pool of rules

    # 5. reevaluate all sets and keep only pareto optimal

    # 6. if pareto front hasn't changed exit, otherwise repeat 3-6

    """ example output: [
            {"ids": [2,5], "prec": 0.2, "rec": 0.75},
            {"ids": [1,2], "prec": 0.5, "rec": 0.63},
        ]
    """
    return []
