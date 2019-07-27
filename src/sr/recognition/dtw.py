import numpy as np
from math import isinf


def dtw(x, y, dist_fun, transitions, variance=None, beam=np.inf):
    """
    :param x: an input
    :param y: a template
    :param dist_fun: function used to calculate distance between two nodes.
    :param transitions: transition matrix, where transitions[i,j] represents the cost for transiting from jth to ith
        state.
    :param variance: covariance matrix of the segments in the templates.
    :param beam: beam size of pruning.
    :return: costs: cost matrix.
            path: reversed path (from end to start). [[x_n,y_n],...,[x_2,y_2],[x_1,y_1]]
    """
    # initialize cost matrix
    col_count = len(x)
    row_count = len(y)
    costs = np.full((row_count, col_count), np.inf)
    path_matrix = np.full((row_count, col_count, 2), np.inf, dtype=np.int)

    j = 0
    while True:
        if j == col_count:
            break
        i = 0
        while True:
            if i == row_count:
                break
            if i == 0 and j == 0:
                if variance is None:
                    costs[0, 0] = dist_fun(x[j], y[i])
                else:
                    costs[0, 0] = dist_fun(x[j], y[i], variance[i])
                i += 1
                continue

            prev_costs = []
            from_pts = []
            for origin in range(transitions.shape[1]):
                # cost is set to -1 if it is excluded from beam search
                if costs[origin, j - 1] == -1:
                    # set it back to np.inf so that it is more convenient to be recognized by pathfinder
                    costs[origin, j - 1] = np.inf
                else:
                    prev_costs.append(transitions[i, origin] + costs[origin, j - 1])
                    from_pts.append([origin, j - 1])

            min_i = np.argmin(prev_costs)
            origin = from_pts[min_i]
            path_matrix[i, j] = origin
            if variance is None:
                curr_cost = prev_costs[min_i] + dist_fun(x[j], y[i])
            else:
                curr_cost = prev_costs[min_i] + dist_fun(x[j], y[i], variance[i])
            costs[i, j] = min(costs[i, j], curr_cost)
            i += 1
        if not isinf(beam):
            sort_idx = np.argsort(costs[:, j].flatten())
            # cost is set to -1 if it is excluded from beam search
            exclude_idx = sort_idx[beam:]
            for i in exclude_idx:
                if not isinf(costs[i, j]):
                    costs[i, j] = -1
        j += 1
    # find the path
    i = row_count - 1
    j = col_count - 1
    path = []
    while i != 0 or j != 0:
        i, j = path_matrix[i, j]
        path.append([i, j])
    return costs, np.array(path)


def dtw1(x, states, transitions, beam=np.inf):
    """
    :param x: an input
    :param states: a list of states (GMM object).
    :param transitions: transition matrix, where transitions[i,j] represents the cost for transiting from jth to ith
        state.
    :param beam: beam size of pruning.
    :return: costs: cost matrix.
            path: reversed path (from end to start). [[x_n,y_n],...,[x_2,y_2],[x_1,y_1]]
    """
    # initialize cost matrix
    col_count = len(x)
    row_count = len(states)
    costs = np.full((row_count, col_count), np.inf)
    path_matrix = np.full((row_count, col_count, 2), np.inf, dtype=np.int)

    j = 0
    while True:
        if j == col_count:
            break
        i = 0
        while True:
            if i == row_count:
                break
            if i == 0 and j == 0:
                costs[0, 0] = states[i].evaluate(x[j])
                i += 1
                continue

            prev_costs = []
            from_pts = []
            for origin in range(transitions.shape[1]):
                # cost is set to -1 if it is excluded from beam search
                if costs[origin, j - 1] == -1:
                    # set it back to np.inf so that it is more convenient to be recognized by pathfinder
                    costs[origin, j - 1] = np.inf
                else:
                    prev_costs.append(transitions[i, origin] + costs[origin, j - 1])
                    from_pts.append([origin, j - 1])

            min_i = np.argmin(prev_costs)
            origin = from_pts[min_i]
            path_matrix[i, j] = origin
            curr_cost = prev_costs[min_i] + states[i].evaluate(x[j])
            costs[i, j] = min(costs[i, j], curr_cost)
            i += 1
        if not isinf(beam):
            sort_idx = np.argsort(costs[:, j].flatten())
            # cost is set to -1 if it is excluded from beam search
            exclude_idx = sort_idx[beam:]
            for i in exclude_idx:
                if not isinf(costs[i, j]):
                    costs[i, j] = -1
        j += 1
    # find the path
    i = row_count - 1
    j = col_count - 1
    path = []
    while i != 0 or j != 0:
        i, j = path_matrix[i, j]
        path.append([i, j])
    return costs, np.array(path)
