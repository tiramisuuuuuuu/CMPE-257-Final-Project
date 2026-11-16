# The paper tests 6 SSF methods, each one may choose different sets from the Pareto front to explore / expand

import numpy as np

def uniform_sampling(front, k):
    """
    Select k approximately equally spaced solutions along the
    precision-recall Pareto front.

    Assumes front is sorted by descending precision.
    Returns the selected rule_subset keys.
    """

    if front.empty or k <= 0:
        return []

    if k >= len(front):
        return front["rule_subset"].tolist()

    points = front[["precision", "recall"]].to_numpy()

    # 1. Calculate Euclidean distance between consecutive points
    distances = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    # 2. Calculate cumulative distance along the front
    cumulative = np.concatenate(([0], np.cumsum(distances)))

    total_distance = cumulative[-1]

    if total_distance == 0:
        return front["rule_subset"].iloc[:k].tolist()

    # 3. Calculate k equally spaced target distances
    targets = np.linspace(0, total_distance, k)

    # 4. Select the closest unused point to each target
    selected_indices = []
    used = set()

    for target in targets:

        differences = np.abs(cumulative - target)

        # Exclude previously selected points
        for idx in used:
            differences[idx] = np.inf

        closest = int(np.argmin(differences))

        selected_indices.append(closest)
        used.add(closest)

    return front.iloc[selected_indices]["rule_subset"].tolist()


def _hypervolume(points, reference=(0.0, 0.0)):
    """
    Calculate exact 2D hypervolume for maximization objectives.

    The dominated region is measured relative to the reference point.
    """

    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return 0.0

    ref_x, ref_y = reference

    # Ignore points outside the reference region
    points = points[
        (points[:, 0] > ref_x) &
        (points[:, 1] > ref_y)
    ]

    if len(points) == 0:
        return 0.0

    # Sort by ascending recall (~ descending precision)
    points = points[np.argsort(points[:, 0])]

    area = 0.0

    # Integrate the union of dominated rectangles
    for i, (x, y) in enumerate(points):
        prev_x = (
            points[i - 1, 0]
            if i - 1 >= 0
            else 0
        )

        area += (x - prev_x) * (y - ref_y)

    return area


def hv_ss(front, k):
    """
    Greedy hypervolume subset selection.

    Selects k solutions by repeatedly choosing the solution
    with the largest marginal hypervolume contribution.

    Returns the selected rule_subset keys.
    """

    if front.empty or k <= 0:
        return []

    if k >= len(front):
        return front["rule_subset"].tolist()

    points = front[["recall", "precision"]].to_numpy()
    keys = front["rule_subset"].tolist()

    selected = []
    remaining = list(range(len(points)))

    # Reference point for recall and precision
    reference = (0.0, 0.0)

    # 1. Calculate initial hypervolume
    current_hv = 0.0

    # 2. Greedily select the largest marginal contribution
    for _ in range(k):

        best_idx = None
        best_contribution = -np.inf

        for idx in remaining:

            trial_indices = selected + [idx]

            trial_hv = _hypervolume(
                points[trial_indices],
                reference
            )

            contribution = trial_hv - current_hv

            if contribution > best_contribution:
                best_contribution = contribution
                best_idx = idx

        if best_idx is None:
            break

        # Add the best candidate
        selected.append(best_idx)
        remaining.remove(best_idx)

        current_hv = _hypervolume(
            points[selected],
            reference
        )

    return [keys[idx] for idx in selected]


def igd(front, k):
    """
    Greedy IGD subset selection.

    Selects k solutions that minimize Inverted Generational Distance.

    Returns the selected rule_subset keys.
    """

    if front.empty or k <= 0:
        return []

    if k >= len(front):
        return front["rule_subset"].tolist()

    # Objectives: x = recall, y = precision
    points = front[["recall", "precision"]].to_numpy()
    keys = front["rule_subset"].tolist()

    # 1. Calculate Euclidean distances between all pairs
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :],
        axis=2
    )

    # 2. Track minimum distance from each reference point to any selected solution
    min_distances = np.full(len(points), np.inf)

    selected = []
    remaining = list(range(len(points)))

    # 3. Greedily select k solutions
    for _ in range(k):

        best_idx = None
        best_igd = np.inf

        for idx in remaining:

            # Calculate distances after adding this candidate
            new_distances = np.minimum(
                min_distances,
                distances[:, idx]
            )

            # Calculate resulting IGD
            igd = np.mean(new_distances)

            if igd < best_igd:
                best_igd = igd
                best_idx = idx

        if best_idx is None:
            break

        # 4. Add the best candidate
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update minimum distances
        min_distances = np.minimum(
            min_distances,
            distances[:, best_idx]
        )

    return [keys[idx] for idx in selected]


def igd_plus(front, k):
    """
    Greedy IGD+ subset selection.

    Selects k solutions that minimize Inverted Generational
    Distance Plus.

    Assumes precision and recall are maximization objectives.

    Returns the selected rule_subset keys.
    """

    if front.empty or k <= 0:
        return []

    if k >= len(front):
        return front["rule_subset"].tolist()

    # Objectives: x = recall, y = precision
    points = front[["recall", "precision"]].to_numpy()
    keys = front["rule_subset"].tolist()

    # 1. Calculate directional IGD+ distances
    differences = points[:, None, :] - points[None, :, :]

    distances = np.linalg.norm(
        np.maximum(differences, 0),
        axis=2
    )

    # 2. Track minimum IGD+ distance for each reference point
    min_distances = np.full(len(points), np.inf)

    selected = []
    remaining = list(range(len(points)))

    # 3. Greedily select k solutions
    for _ in range(k):

        best_idx = None
        best_igd_plus = np.inf

        for idx in remaining:

            # Calculate distances after adding this candidate
            new_distances = np.minimum(
                min_distances,
                distances[:, idx]
            )

            # Calculate resulting IGD+
            igd_plus = np.mean(new_distances)

            if igd_plus < best_igd_plus:
                best_igd_plus = igd_plus
                best_idx = idx

        if best_idx is None:
            break

        # 4. Add the best candidate
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update minimum distances
        min_distances = np.minimum(
            min_distances,
            distances[:, best_idx]
        )

    return [keys[idx] for idx in selected]
