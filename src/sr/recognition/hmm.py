# -*- coding: utf-8 -*-
from .dtw import dtw, dtw1
from .kmeans import kmeans, skmeans, align_gmm_states
from .gmm import GMM, mahalanobis
import numpy as np


class HMM:
    """
    Attributes
    ----------
    mu: an array of means of all segments

    sigma: an array of variance of all segments
    transitions: transition matrix

    segments: a list of mfcc feature vectors of each segments
    """

    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.mu = None
        self.sigma = None
        self.transitions = None
        self.segments = []
        self.gmm_states = None
        self.use_gmm = True
        self.use_em = True

    def __eq__(self, other):
        if self.use_gmm:
            if not other.use_gmm:
                return False
            if self.n_segments != other.n_segments:
                return False
            for i in range(self.n_segments):
                if self.gmm_states[i] != other.gmm_states[i]:
                    return False
            return True
        else:
            return np.allclose(self.mu, other.mu) and np.allclose(self.sigma, other.sigma)

    def reset(self):
        self.mu = None
        self.sigma = None
        self.transitions = None
        self.segments = []
        self.gmm_states = None

    def __getitem__(self, item):
        if type(item) is int or type(item) is slice:
            return self.gmm_states[item]
        else:
            raise TypeError('The type of index is not supported')

    def fit(self, ys, n_gaussians, use_gmm=True, use_em=True):
        """Fit the HMM model with a list of training data.
        :param use_gmm: TODO: add docstring
        :param use_em: if true, use both k-means and Expectation-Maximization algorithm to fit GMM, otherwise only
            use k-means to do it.
        :param ys: a list of mfcc templates. It cannot be a numpy array since variable length rows in a matrix are
            not supported.
        :param n_gaussians: the number of gaussian distributions. It should be some power of 2, if not it will be
            converted to previous power of 2.
        :return: the trained HMM model.
        """
        self.use_em = use_em
        self.use_gmm = use_gmm
        if use_gmm:
            self.fit_GMM(ys, n_gaussians)
        else:
            self.mu, self.sigma, self.transitions, self.segments = skmeans(ys, self.n_segments,
                                                                           return_segmented_data=True)
        return self

    def _init_gmm(self, n_gaussians):
        self.gmm_states = [GMM(self.mu[i, :], self.sigma[i, :], n_gaussians) for i in range(self.n_segments)]

    def fit_GMM(self, ys, n_gaussians):
        """fit all GMM states in the HMM.
        :param n_gaussians: the number of gaussians.
        """
        print('Doing segmental k-means')
        self.mu, self.sigma, self.transitions, self.segments = skmeans(ys, self.n_segments,
                                                                       return_segmented_data=True)
        self._init_gmm(n_gaussians)

        # train each GMM
        for i in range(len(self.segments)):
            self._fit_GMM(self.segments[i], n_gaussians, i)

        # update segments
        self.segments = align_gmm_states(ys, self.gmm_states, self.transitions, self.n_segments)

    def _fit_GMM(self, data, n_gaussians, seg_i):
        """fit a single GMM state. Only for internal use.
        :param data: the data only for this state.
        :param n_gaussians: the number of gaussians.
        :param seg_i: the index of this segment/state.
        :return: a GMM object.
        """
        n_splits = int(np.log(n_gaussians))
        assert n_splits > 0

        centroids = np.array([self.mu[seg_i, :]])
        weights = np.full(n_gaussians, 1 / data.shape[0])
        for i in range(n_splits):
            lcentroids = centroids * 0.9
            hcentroids = centroids * 1.1
            centroids = np.concatenate([lcentroids, hcentroids], axis=0)
            clusters, centroids, variance = kmeans(data, 2 ** (i + 1), centroids, dist_fun=mahalanobis)

            # calculate and update weights of distributions
            cs, c_counts = np.unique(clusters, return_counts=True)
            for c in cs:
                weights[c] = c_counts[c] / data.shape[0]

            # update model parameters
            self.gmm_states[seg_i].update_models(centroids, variance, weights[:2 ** (i + 1)])
            # Expectation-maximization
            if self.use_em:
                self.gmm_states[seg_i].em(data, 2 ** (i + 1))

    def evaluate(self, x):
        """
        :param x: input data.
        :return: cost/score/probability.
        """
        if self.use_gmm:
            costs, _ = dtw1(x, self.gmm_states, self.transitions)
        else:
            costs, _ = dtw(x, self.mu, mahalanobis, self.transitions, self.sigma)
        return costs[-1, -1]
