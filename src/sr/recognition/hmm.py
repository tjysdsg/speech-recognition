# -*- coding: utf-8 -*-
from .dtw import *
from .kmeans import *
import numpy as np
import copy
import scipy


class MultivariateNormal:
    def __init__(self, mean, cov):
        self.mean = mean
        self._cov = cov
        if len(self.cov.shape) == 1:
            cov = np.diag(self.cov)
        else:
            cov = self.cov
        self.inv_cov = np.linalg.inv(cov)

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, val):
        self._cov = val
        if len(self.cov.shape) == 1:
            cov = np.diag(self.cov)
        else:
            cov = self.cov
        self.inv_cov = np.linalg.inv(cov)

    @cov.deleter
    def cov(self):
        del self.cov

    def pdf(self, x):
        size = x.shape[0]
        if size == self.mean.shape[0]:
            det = np.prod(self.cov)
            norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.sqrt(det))
            x_mu = x - self.mean
            result = np.exp(-0.5 * (x_mu.dot(self.inv_cov).dot(x_mu.T)))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")


def dist(v1, v2, variance):
    D = len(variance)
    m = (v1 - v2)
    return 0.5 * np.log((2 * np.pi) ** D * np.prod(variance)) + 0.5 * np.sum(m / variance * m)


class GMM:
    def __init__(self, mu, sigma, n_gaussians):
        self.n_gaussians = n_gaussians
        self.w = np.full(n_gaussians, 1 / n_gaussians)
        self.dists = [MultivariateNormal(mean=mu, cov=sigma) for _ in range(n_gaussians)]
        self.mu_old = np.tile(mu, (n_gaussians, 1))
        self.sigma_old = np.tile(sigma, (n_gaussians, 1))
        self.w_old = np.full(n_gaussians, 1 / n_gaussians)

    def evaluate(self, x, return_neg_log_likelihood=True):
        res = [self.dists[i].pdf(x) * self.w[i] for i in range(self.n_gaussians)]
        res = np.array(res)
        if return_neg_log_likelihood:
            return -np.log(res.sum())
        else:
            return res

    def em(self, data, n_gaussians, max_iteration=10000):
        for iter in range(max_iteration):
            p = np.zeros((data.shape[0], n_gaussians))
            mu = np.zeros((data.shape[1], n_gaussians))
            sigma = np.zeros((data.shape[1], n_gaussians))
            for i in range(data.shape[0]):
                p[i, :] = self.evaluate(data[i, :], return_neg_log_likelihood=False)[:n_gaussians]

            p_sum = np.sum(p, axis=1).reshape((p.shape[0], 1))
            # avoid divide by 0
            p_sum[p_sum == 0] = 10 ** (-5)
            p /= p_sum
            p_sum = np.sum(p, axis=0)
            # avoid divide by 0
            p_sum[p_sum == 0] = 10 ** (-5)
            for c in range(n_gaussians):
                mu[:, c] = np.sum(data * p[:, [c]], axis=0)
                mu[:, c] /= p_sum[c]
                # sigma
                sub2 = (data - mu[:, c]) ** 2
                sigma[:, c] = np.sum(sub2 * p[:, [c]], axis=0)
                sigma[:, c] /= p_sum[c]

            mu = mu.T
            sigma = sigma.T
            # update w and mu
            w = p.mean(axis=0).T
            self.update_models(mu, sigma, w)
            if np.allclose(mu, self.mu_old[:n_gaussians, :]) and \
                    np.allclose(sigma, self.sigma_old[:n_gaussians, :]) and \
                    np.allclose(w, self.w_old[:n_gaussians]):
                print("EM converged at iteration:", iter)
                break
            else:
                print("EM iteration:", str(iter), end="\r")
                self.mu_old[:n_gaussians, :] = mu
                self.sigma_old[:n_gaussians, :] = sigma
                self.w_old[:n_gaussians] = w

    def update_models(self, mus, sigmas, weights):
        self.w[:mus.shape[0]] = weights
        for m in range(mus.shape[0]):
            self.dists[m].mean = mus[m, :]
            self.dists[m].cov = sigmas[m, :]

    def __eq__(self, other):
        res = True
        res = res and self.n_gaussians == other.n_gaussians and np.allclose(self.w, other.w)
        for i in range(self.n_gaussians):
            res = res and np.allclose(self.dists[i].mean, other.dists[i].mean) and np.allclose(self.dists[i].cov,
                                                                                               other.dists[i].cov)
        return res


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
            clusters, centroids, variance = kmeans(data, 2 ** (i + 1), centroids, dist_fun=dist)

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
            costs, _ = dtw(x, self.mu, dist, self.transitions, self.sigma)
        return costs[-1, -1]
