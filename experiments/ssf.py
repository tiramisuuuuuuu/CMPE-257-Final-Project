# The paper tests 6 SSF methods, each one may choose different sets from the Pareto front to explore / expand

def uniform_sampling(front, k):
    # input pareto front should already be sorted. Since pareto front contains non-dominated solutions, sorting solutions by descending precision would have implicitly sorted recall in ascending order

    # calculate euclidean distance from first solution to last

    # divide total distance by k to find the ideal step size between the selected k solutions

    # find the ideal distances of solutions, by multiplying step size by i, where 0<=i<=k

    # iterate every data point and find how close it is from each ideal distance

    # iterate ideal distances and choose the data point which is the closest and then mark it as "used"

    return []


def hv_ss(front, k):
    # calculate hypervolume indicator for every set

    # sort sets by hypervolume

    # greedily select the first set, with largest hv

    # for the next k-1 sets, try to select the one with largest hv. 
    # Check the largest hv set - recalculate its hv contribution relative to the already selected set. If this new hv is no longer the max, then try the next set instead

    return []
