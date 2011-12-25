#!/usr/bin/env python
# encoding: utf-8
"""
The experiments outlined in document/pyest.pdf

"""

import numpy as np
np.random.seed(5)

import pyest

def lnprob_gaussian(x, icov):
    return -np.dot(x,np.dot(icov,x))/2.0

def ndgaussian():
    nwalkers = 100
    ndim     = 50

    N = 1e3
    resample = 1

    cov  = 0.5-np.random.rand(ndim*ndim).reshape((ndim,ndim))
    cov  = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov  = np.dot(cov, cov)
    icov = np.linalg.inv(cov)
    p0   = [0.1*np.random.randn(ndim) for i in xrange(nwalkers)]

    # Metropolis-Hastings
    print "Metropolis-Hastings"
    sampler = pyest.MHSampler(cov, ndim, lnprob_gaussian, args=[icov])
    for i in sampler.sample(np.array(p0[0]), iterations=N, resample=resample):
        pass
    try:
        print "acor time: ", sampler.acor
    except Exception as e:
        print "acor failed: ", e

    # Stretch move
    print "Stretch move"
    sampler = pyest.DualEnsembleSampler(nwalkers, ndim, lnprob_gaussian, args=[icov])
    for i in sampler.sample(np.array(p0), iterations=N, resample=resample):
        pass
    try:
        print "acor time: ", np.mean(sampler.acor)
    except Exception as e:
        print "acor failed: ", e

    # Single Gaussian proposal
    print "Single Gaussian proposal"
    sampler = pyest.DualEnsembleSampler(nwalkers, ndim, lnprob_gaussian, args=[icov],
            ensemble_type=pyest.SingleGaussianEnsemble)
    for i in sampler.sample(np.array(p0), iterations=N, resample=resample):
        pass
    try:
        print "acor time: ", np.mean(sampler.acor)
    except Exception as e:
        print "acor failed: ", e

    # Mixture of Gaussians proposal
    print "Mixture of Gaussians proposal"
    sampler = pyest.DualEnsembleSampler(nwalkers, ndim, lnprob_gaussian, args=[icov],
            ensemble_type=pyest.MOGEnsemble, ensemble_args={'K': 1})
    for i in sampler.sample(np.array(p0), iterations=N, resample=resample):
        pass
    try:
        print "acor time: ", np.mean(sampler.acor)
    except Exception as e:
        print "acor failed: ", e

    # Mixture of Gaussians proposal
    print "Affine Invariant Mixture of Gaussians proposal"
    sampler = pyest.DualEnsembleSampler(nwalkers, ndim, lnprob_gaussian, args=[icov],
            ensemble_type=pyest.AIMOGEnsemble, ensemble_args={'K': 1})
    for i in sampler.sample(np.array(p0), iterations=N, resample=resample):
        pass
    try:
        print "acor time: ", np.mean(sampler.acor)
    except Exception as e:
        print "acor failed: ", e

if __name__ == '__main__':
    ndgaussian()

