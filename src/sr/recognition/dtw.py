# -*- coding: utf-8 -*-
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


def sentence_viterbi(x, model_graph):
    # TODO: allow arbitrary jumps between gmm states depending on the hmm model
    """
    :param x: input features.
    :param model_graph: TODO: add docstring
    :return: costs: cost matrix
            matched_sentence: a sequence of index of models that best matches the input.

    """
    nodes = model_graph.nodes

    # initialize cost matrix
    n_cols = len(x)
    n_rows = len(nodes)
    costs = np.full((n_rows, n_cols), np.inf)
    costs[0, 0] = 0

    path_matrix = np.full((n_rows, n_cols, 2), np.inf, dtype=np.int)

    # the end of a sentence
    end_nodes = model_graph.get_ends()

    # fill cost matrix
    for c in range(n_cols):
        for r in range(0, n_rows):
            if r == 0 and c == 0:
                continue
            subcosts = []
            from_points = []

            if model_graph[r].model_index == 'NES':
                # the cost of non-emitting state is 0
                node_dist = 0
            else:
                node_dist = model_graph[r].val.evaluate(x[c])

            origins, transition_cost = model_graph.get_origins(model_graph[r])

            for o in origins:
                origin_idx = o.node_index
                if model_graph[r].model_index == 'NES':
                    subcosts.append(costs[origin_idx, c] + node_dist)
                    from_points.append([origin_idx, c])
                elif c > 0:
                    subcosts.append(transition_cost + costs[origin_idx, c - 1] + node_dist)
                    from_points.append([origin_idx, c - 1])

            # if there is no node that can get to current position, simply skip it.
            if len(subcosts) == 0:
                continue
            # remember path and cost
            min_idx = np.argmin(subcosts)
            path_matrix[r, c] = from_points[min_idx]
            costs[r, c] = subcosts[min_idx]

    # find the sentence which has the min cost
    end_node_costs = [costs[n.node_index, -1] for n in end_nodes]
    min_idx = np.argmin(end_node_costs)
    best_end_idx = end_nodes[min_idx].node_index
    best_cost = costs[best_end_idx, -1]

    # find the matched string
    c = n_cols - 1
    r = best_end_idx
    matched_sentence = [nodes[r].model_index]
    while 1:
        if c < 1:
            break
        r, c = path_matrix[r, c]
        if r != 0:
            matched_sentence.append(nodes[r].model_index)

    return best_cost, matched_sentence[::-1]
