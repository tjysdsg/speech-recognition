# -*- coding: utf-8 -*-
from .decode import dtw, decode_hmm_states
import numpy as np


def calc_variance(data):
    return np.cov(data).diagonal()


def combine_templates(templates, n_temps, n_segments, seg_starts):
    """
    :param templates: a list of templates
    :param n_temps: the number of templates
    :param n_segments: the number of segments the final templates will have
    :param seg_starts: the start points of the segments in the original templates
    :return: the combined templates
    """
    res = np.zeros((n_segments, templates[0].shape[1]))
    vars = np.zeros((n_segments, templates[0].shape[1]))
    segments = segment_data(templates, n_temps, n_segments, seg_starts)
    for s in range(n_segments):
        seg = segments[s]
        res[s] = seg.mean(axis=0)
        vars[s] = calc_variance(seg.T)
    return res, vars


def segment_data(templates, n_temps, n_segments, seg_starts):
    """
    :param templates: a list of templates
    :param n_temps: the number of templates
    :param n_segments: the number of segments the final templates will have
    :param seg_starts: the start points of the segments in the original templates
    :return: the segmented data
    """
    segments = []
    for s in range(n_segments):
        seg = []
        for r in range(n_temps):
            if s == n_segments - 1:
                seg += templates[r][seg_starts[r, s]:].tolist()
            else:
                seg += templates[r][seg_starts[r, s]:seg_starts[r, s + 1]].tolist()
        segments.append(np.array(seg))
    return segments


def calc_transition_costs(templates, seg_lens, max_jump_dist=2):
    """
    Calculate and construct a dictionary used in dtw()
    :param templates: a list of templates
    :param seg_lens: list of length of segments of all templates
    :param max_jump_dist: maximum distance allowed for a transition to jump across, the default value 2 is optimal in
        most occasions.
    :return: the transition cost
    """
    n_segments = seg_lens.shape[1]
    n_temps = len(templates)
    empty_segs = (seg_lens == 0)
    res = np.full((n_segments, n_segments), np.inf)
    for i in range(n_segments):
        jump_dist = 1
        # number of samples which transition to other segments
        if i == n_segments - 1:
            n_jump = 0
        else:
            n_jump = n_temps
        s = i + 1
        # calculate the distance of the jump
        while s < n_segments - 1:
            # if the next segments is not empty, the jump stops here; otherwise increment jump_dist
            if np.sum(empty_segs[:, s + 1]) == 0:
                break
            jump_dist += 1
            if jump_dist > max_jump_dist:
                break
            s += 1
        # number of samples in this segment
        n_all = 0
        for t in range(len(templates)):
            n_all += seg_lens[t, i]
        # number of samples which transition to itself
        n_stay = n_all - n_jump
        p_stay = n_stay / n_all
        p_jump = n_jump / n_all
        if n_jump:
            res[i + jump_dist, i] = -np.log(p_jump)
        res[i, i] = -np.log(p_stay)
    return res


def get_segments_from_path(path, n_segments):
    # count all segments occurrence
    unique, counts = np.unique(path[:, 0], return_counts=True)
    seg_counts = dict(zip([i for i in range(n_segments)], [0 for _ in range(n_segments)]))
    seg_counts.update(dict(zip(unique, counts)))
    counts = np.array(list(seg_counts.values()))
    unique = np.array(list(seg_counts.keys()))
    sort_idx = np.argsort(unique)
    counts = counts[sort_idx]
    seg_starts = np.add.accumulate(counts)[:-1]
    return seg_starts


def skmeans(templates, n_segments, dist_fun=lambda *args: np.linalg.norm(args[0] - args[1]),
            return_segmented_data=False, max_iteration=1000):
    """
    :param templates: templates as a list
    :param n_segments: number of segments in the final template
    :param dist_fun: function to calculate distance of two nodes in the templates
    :param return_segmented_data: if true, return segmented data, and vice versa.
    :param max_iteration: max iteration if not converged
    :return: res: a combined template;
            vars: the variance of all segments;
            transition_costs: cost matrix of transitions;
            if return_segmented_data is set to true, segmented_data is returned, it is a list of all segments.
    """
    assert max_iteration > 0
    n_temps = len(templates)
    seg_lens = np.zeros((n_temps, n_segments + 1), dtype=np.int)
    for r in range(n_temps):
        temp_len = len(templates[r])
        seg_len = temp_len // n_segments
        seg_lens[r, 1:] = seg_len
    seg_starts = np.add.accumulate(seg_lens, axis=1)[:, :-1]
    seg_lens = seg_lens[:, 1:]

    res, vars = combine_templates(templates, n_temps, n_segments, seg_starts)
    for _ in range(max_iteration):
        # do dtw and align templates
        seg_starts = np.zeros((n_temps, n_segments), dtype=np.int)
        transition_costs = calc_transition_costs(templates, seg_lens)
        for r in range(n_temps):
            # if template contains too few mfcc vectors, we cannot find a path in dtw
            if templates[r].shape[0] < 5:
                raise NameError('template is too small, cannot do dtw on it')

            _, path = dtw(templates[r], res, dist_fun, transition_costs)
            seg_starts[r, 1:] = get_segments_from_path(path, n_segments)
        new_res, vars = combine_templates(templates, n_temps, n_segments, seg_starts)
        if np.allclose(res, new_res):
            break
        res = new_res
    if return_segmented_data:
        segmented_data = segment_data(templates, n_temps, n_segments, seg_starts)
        return res, vars, transition_costs, segmented_data
    else:
        return res, vars, transition_costs


def cluster_centroids(data, clusters, k):
    """Return centroids of clusters in data.
    """
    result = np.empty(shape=(k,) + data.shape[1:])
    for i in range(k):
        np.mean(data[clusters == i, :], axis=0, out=result[i])
    return result


def kmeans(data, k, centroids, dist_fun=lambda *args: np.linalg.norm(args[0] - args[1]), max_iteration=1000):
    """Modified k-means algorithm to fit the scenario
    """
    assert k == centroids.shape[0]
    clusters = np.random.randint(0, k, data.shape[0])
    # find the covariance matrix of the new clusters
    cov = []
    for c in range(k):
        segments = data[clusters == c]
        cov.append(calc_variance(segments.T))
    cov = np.array(cov)
    # calculate distance from centroids
    for _ in range(max(max_iteration, 1)):
        dists = np.zeros((data.shape[0], k))
        for i in range(data.shape[0]):
            for c in range(k):
                dists[i, c] = dist_fun(centroids[c, :], data[i, :], cov[0])
        dists = np.array(dists)
        # Index of the closest centroid to each data point.
        clusters = np.argmin(dists, axis=1)
        new_centroids = cluster_centroids(data, clusters, k)

        # check convergence
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters, centroids, cov


def align_gmm_states(templates, gmm_states, transition_costs, n_segments):
    n_temps = len(templates)

    # do dtw and align templates
    seg_starts = np.zeros((n_temps, n_segments), dtype=np.int)
    for r in range(n_temps):
        t = templates[r]
        _, path = decode_hmm_states(t, gmm_states, transition_costs)
        seg_starts[r, 1:] = get_segments_from_path(path, n_segments)
    return segment_data(templates, n_temps, n_segments, seg_starts)
