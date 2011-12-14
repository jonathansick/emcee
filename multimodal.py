#!/usr/bin/env python
# encoding: utf-8
"""
A really hard multi-modal test case

"""

import matplotlib.pyplot as pl

import numpy as np

from pyest.ensemble import EnsembleSampler

def lnprob_gaussian(x, mu, icov, logdet):
    d = x-mu
    return -np.dot(d,np.dot(icov,d))/2.0 - 0.5*logdet*len(x)*np.log(2*np.pi)

def multi_modal():
    nwalkers = 100
    ndim = 2
    K = 3

    means = np.array([10*np.random.rand(ndim) for k in range(K)])
    covs = []
    invs = []
    lndets = []
    for k in range(K):
        cov  = 5*(0.5-np.random.rand(ndim*ndim).reshape((ndim,ndim)))
        cov  = np.triu(cov)
        cov += cov.T - np.diag(cov.diagonal())
        cov  = np.dot(cov, cov)
        print cov
        icov = np.linalg.inv(cov)
        covs.append(cov)
        invs.append(icov)
        lndets.append(np.linalg.slogdet(cov)[1])

    p0   = [10*np.random.rand(ndim) for i in xrange(nwalkers)]

    def lnprob(p):
        ls = [lnprob_gaussian(p, means[k], invs[k], lndets[k]) for k in range(K)]
        return np.sum(ls)

    sampler = EnsembleSampler(nwalkers, ndim, lnprob)
    for i in sampler.sample(p0, iterations=100):
        pass

    print np.mean(sampler.acceptance_fraction)
    chain = sampler.flatchain

    pl.plot(chain[:,0], chain[:,1], '.k')
    pl.plot(means[:,0], means[:,1], 'or')
    pl.show()

if __name__ == '__main__':
    multi_modal()

