# -*- coding: utf-8 -*-
from .hmm_state import NES
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
    assert col_count > 1 and row_count > 1
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


def decode_hmm_states(x, states, transitions, end_points=None):
    """
    :param end_points: TODO: add docstring
    :param x: an input
    :param states: a list of hmm states (derived from class HMMStates).
    :param transitions: transition matrix, where `transitions[i,j]` represents the cost for transiting from jth to ith
        state.
    :return: costs: cost matrix.
            path: reversed path (from end to start) in format `[[x_n, y_n], ... , [x_2, y_2] ,[x_1, y_1]]`.
    """
    # initialize cost matrix
    col_count = len(x)
    row_count = len(states)
    costs = np.full((row_count, col_count), np.inf)
    path_matrix = np.full((row_count, col_count, 2), np.inf, dtype=np.int)

    for c in range(col_count):
        for r in range(row_count):
            if r == 0 and c == 0:
                costs[0, 0] = states[r].evaluate(x[c])
                continue

            prev_costs = []
            from_pts = []
            for origin in range(transitions.shape[1]):
                if isinf(transitions[r, origin]):
                    continue
                # if current state or the state that transits to current state is a non-emitting states
                if type(states[origin]) == NES or type(states[r]) == NES:
                    prev_costs.append(transitions[r, origin] + costs[origin, c])
                    from_pts.append([origin, c])
                else:
                    prev_costs.append(transitions[r, origin] + costs[origin, c - 1])
                    from_pts.append([origin, c - 1])

            if len(prev_costs) == 0:
                continue
            min_i = np.argmin(prev_costs)
            origin_point = from_pts[min_i]
            if origin_point == [r, c]:
                raise NameError("FUCKED")
            path_matrix[r, c] = origin_point
            curr_cost = prev_costs[min_i] + states[r].evaluate(x[c])
            costs[r, c] = min(costs[r, c], curr_cost)

    if end_points is None:
        end_points = [[row_count - 1, col_count - 1]]
    # find the best path with lowest cost
    best_path_cost = np.inf
    best_end = []
    for end in end_points:
        if best_path_cost >= costs[end[0], end[1]]:
            best_path_cost = costs[end[0], end[1]]
            best_end = end

    i = best_end[0]
    j = best_end[1]
    if isinf(costs[i, j]):
        import warnings
        warnings.warn("decode_hmm_states: Cannot find a path when decoding sequence")

    path = []
    while j != 0:
        i, j = path_matrix[i, j]
        path.append([i, j])
    return costs, np.array(path)
