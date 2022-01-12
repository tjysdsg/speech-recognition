# -*- coding: utf-8 -*-
import numpy as np


class MultivariateNormal:
    """
    Multivariate normal (gaussian) distribution using the diagonal of its covariance matrix.
    """

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


def mahalanobis(v1, v2, variance):
    """
    Mahalanobis distance function. FIXME consider moving similar math utility function to a seperate folder?
    :param v1: the first 1d array.
    :param v2: the second 1d array.
    :param variance: 1d array, variance.
    :return: scalar, the mahalanobis distance.
    """
    D = len(variance)
    m = (v1 - v2)
    return 0.5 * np.log((2 * np.pi) ** D * np.prod(variance)) + 0.5 * np.sum(m / variance * m)


class HMMState:
    """
    Base class for all HMM states.
    """

    def __init__(self):
        import uuid
        self.id = uuid.uuid4().int
        self.parent = None

    def evaluate(self, x):
        raise NotImplemented()

    def __eq__(self, other):
        raise NotImplemented()

    def __hash__(self):
        return hash(self.id)


class NES(HMMState):
    """
    Non-emitting state.
    Mainly used for type() comparison, and it has no other specific functionality.
    """

    def __init__(self):
        super().__init__()

    def evaluate(self, x):
        return 0

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class GMM(HMMState):
    """
    Gaussian Mixture Model, using Expectation-Maximization algorithm to update parameters.
    """

    def __init__(self, mu, sigma, n_gaussians):
        super().__init__()
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
                print("EM iteration:", str(iter), end="\r", flush=True)
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

    def __len__(self):
        return self.n_gaussians

    def __hash__(self):
        return hash(self.id)
