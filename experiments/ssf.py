# The paper tests 6 SSF methods, each one may choose different sets from the Pareto front to explore / expand

def uniform_sampling(front, k):
    # input pareto front should already be sorted. Since pareto front contains non-dominated solutions, sorting solutions by descending precision would have implicitly sorted recall in ascending order

    # calculate euclidean distance from first solution to last

    # divide total distance by k to find the ideal step size between the selected k solutions

    # find the ideal distances of solutions, by multiplying step size by i, where 0<=i<=k

    # iterate every data point and find how close it is from each ideal distance

    # iterate ideal distances and choose the data point which is the closest and then mark it as "used"

    return [] # return which sets to expand
