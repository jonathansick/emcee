#!/usr/bin/env python
# encoding: utf-8
"""
A vanilla Metropolis-Hastings sampler

"""

from __future__ import division

__all__ = ['MHSampler']

import numpy as np

from sampler import Sampler

class MHSampler(Sampler):
    """
    The most basic possible Metropolis-Hastings style MCMC sampler for comparison

    Parameters
    ----------
    cov : numpy.ndarray (dim, dim)
        The covariance matrix to use for the proposal distribution.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    args : list, optional
        A list of arguments for lnprobfn.

    """
    def __init__(self, cov, *args, **kwargs):
        super(MHSampler, self).__init__(*args, **kwargs)
        self.cov = cov

    def do_reset(self):
        self.chain = np.empty((self.dim, 0))

    def sample(self, p0, lnprob=None, randomstate=None, storechain=True,
            iterations=1):
        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = self.lnprob(p)

        # resize chain
        if storechain:
            self.chain = np.concatenate((self.chain,
                    np.zeros((self.dim, iterations))), axis=1)

        for i in xrange(iterations):
            self.iterations += 1

            # proposal
            q = self._random.multivariate_normal(p, self.cov)
            newlnprob = self.lnprob(q)
            diff = newlnprob-lnprob

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()

            if diff > 0:
                p = q
                self.chain[:,i] = p
                lnprob = newlnprob
                self.naccepted += 1
            yield p, lnprob, self.random_state

if __name__ == '__main__':
    ndim = 10
    N = 100

    mean = np.zeros(ndim)
    cov  = 0.5-np.random.rand(ndim*ndim).reshape((ndim,ndim))
    cov  = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov  = np.dot(cov,cov)
    icov = np.linalg.inv(cov)
    p0   = np.random.randn(ndim)

    def lnprob_gaussian(x, icov):
        return -np.dot(x,np.dot(icov,x))/2.0

    sampler = MHSampler(cov, ndim, lnprob_gaussian, args=[icov])
    for i in sampler.sample(p0, iterations=1000):
        pass

    print sampler.acceptance_fraction

