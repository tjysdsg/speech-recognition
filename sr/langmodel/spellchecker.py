# -*- coding: utf-8 -*-
from .lextree import *
import numpy as np
import copy


def get_nodes(node_list, lexnode):
    node_list.append(lexnode)
    if len(lexnode.children) == 0:
        return
    for c in lexnode.children:
        get_nodes(node_list, c)


def text_viterbi(x, lextree, dist_fun=lambda *args: int(args[0] != args[1])):
    """
    :param x: a string
    :param lextree: lexical tree as a linked list, type LexNode
    :param dist_fun: function used to calculate distance between two nodes.
    :return: costs: cost matrix, this is probably useless on most occasions, since the flattened tree make it difficult
        to tell which row represent which character in which string.
            matched_string: the string in the dictionary that best matches the input string.

    FIXME: a lot of cache misses when lextree is huge
    """
    x = '*' + copy.copy(x)
    # TODO: be able to adjust different types of costs
    deletion_cost = 1.0
    insertion_cost = 1.0
    match_cost = 0.0
    space_cost = 0.0
    loop_cost = 1.0

    assert type(lextree) is LexNode
    nodes = []
    get_nodes(nodes, lextree)
    nodes.append(LexNode(' '))

    # initialize cost matrix
    n_cols = len(x)
    n_rows = len(nodes)
    costs = np.full((n_rows, n_cols), np.inf)
    costs[0, :] = 0

    # path matrix
    # 0 deletion
    # 1 match
    # 2 insertion
    path_matrix = np.zeros((n_rows, n_cols, 2), dtype=np.int)

    # get transitions
    transitions = {}
    n_nodes = len(nodes)
    word_ends = [n_rows - 1]
    for i in range(n_nodes):
        n = nodes[i]
        # remember its index and add a transition from it to space node, if it's the end of a word.
        if n.property == 2:
            word_ends.append(i)
        # add transition if there is any.
        if len(n.children) > 0:
            for child in n.children:
                transitions[nodes.index(child)] = i

    # fill cost matrix
    for c in range(n_cols):
        for r in range(n_rows):
            if r == 0 and c == 0:
                continue
            subcosts = [np.inf for _ in range(4)]
            from_points = [[0, 0] for _ in range(4)]
            node_dist = dist_fun(x[c], nodes[r].val)

            if r in transitions:
                parent_index = transitions[r]
            else:
                parent_index = None

            # if this is a space node
            if r == n_rows - 1:
                subcosts = [node_dist + space_cost + costs[word_end, c - 1] for word_end in word_ends[1:]]
                from_points = [[word_end, c - 1] for word_end in word_ends[1:]]
            elif parent_index == 0:
                if c > 0:
                    # deletion
                    subcosts[0] = node_dist + deletion_cost + costs[r, c - 1]
                    from_points[0] = [r, c - 1]
                if parent_index is not None:
                    # insertion
                    subcosts[2] = node_dist + insertion_cost + costs[transitions[r], c]
                    from_points[2] = [transitions[r], c]
            elif r == 0:
                if c > 0:
                    # calculate loop costs
                    loop_costs = [node_dist + loop_cost + costs[word_end, c - 1] for word_end in word_ends]
                    # find the lowest loop cost
                    loop_min_idx = np.argmin(loop_costs)
                    subcosts[3] = loop_costs[loop_min_idx]
                    # remember coordinates
                    from_points[3] = [word_ends[loop_min_idx], c - 1]
            else:
                if c > 0:
                    # deletion
                    subcosts[0] = node_dist + deletion_cost + costs[r, c - 1]
                    from_points[0] = [r, c - 1]
                if parent_index is not None and c > 0:
                    # match
                    subcosts[1] = node_dist + match_cost + costs[transitions[r], c - 1]
                    from_points[1] = [transitions[r], c - 1]
                if parent_index is not None:
                    # insertion
                    subcosts[2] = node_dist + insertion_cost + costs[transitions[r], c]
                    from_points[2] = [transitions[r], c]

            # remember path and cost
            min_idx = np.argmin(subcosts)
            path_matrix[r, c] = from_points[min_idx]
            costs[r, c] = subcosts[min_idx]

    subcosts = [costs[word_end, n_cols - 1] for word_end in word_ends]
    min_idx = np.argmin(subcosts)
    min_word_end = word_ends[min_idx]
    best_cost = subcosts[min_idx]

    # find the matched string
    c = n_cols - 1
    r = min_word_end
    matched_string = nodes[r].val
    while 1:
        if c == 1:
            break
        r, c = path_matrix[r, c]
        if r != 0:
            matched_string += nodes[r].val
    return best_cost, matched_string[::-1]


class SpellChecker:
    def __init__(self, beam):
        self.dictionary = None
        self.beam = beam

    def fit(self, dictionary):
        self.dictionary = dictionary
        # TODO: fit spellchecker so that it can recognize continuous text.

    def spell_check(self, text):
        # TODO: evaluate input
        pass
