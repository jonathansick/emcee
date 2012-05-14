#!/usr/bin/env python
# encoding: utf-8
"""
Demonstrates the use of MapScale worker pools for multiprocessing with emcee.
The sampling problem is based on quickstart.py -- sampling of a 50D Guassian.

2012-05-14 - Created by Jonathan Sick
"""

import numpy as np
import emcee
from mapscale.mapscale import Processor


def main():
    # We'll sample a 50-dimensional Gaussian...
    ndim = 50
    # ...with randomly chosen mean position...
    means = np.random.rand(ndim)
    # ...and a positive definite, non-trivial covariance matrix.
    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    # Invert the covariance matrix first.
    icov = np.linalg.inv(cov)

    # We'll sample with 250 walkers.
    nwalkers = 250

    # Choose an initial set of positions for the walkers.
    p0 = [np.random.rand(ndim) for i in xrange(nwalkers)]

    # Set up the probability function
    lnprob = Posterior(means, icov)

    # Set up MapScale processor. This will boot up 8 local work servers.
    lnprobMapper = Processor(lnprob, 8)

    # Set up the EnsembleSampler to use our lnprobMapper.
    # Note that don't pass in lnprob itself; we've baked the probability
    # function into our mapper.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, None,
            lnprob_mapper=lnprobMapper)

    # Run 500 steps as a burn-in.
    pos, prob, state = sampler.run_mcmc(p0, 100)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for 2000
    # steps.
    sampler.run_mcmc(pos, 1000, rstate0=state)

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 100-dimensional
    # vector.
    print "Mean acceptance fraction:", np.mean(sampler.acceptance_fraction)

    # Tell the lnprobMapper to shutdown the work servers.
    lnprobMapper.shutdown()


class Posterior(object):
    """Example posterior function"""
    def __init__(self, mu, icov):
        super(Posterior, self).__init__()
        self.mu = mu
        self.icov = icov

    def setup(self):
        """Include setup code here. This is called once in every process."""
        pass

    def cleanup(self):
        """Include shutdown code here. Use this to clean up the environment
        (delete temp work directories, etc.)
        """
        pass

    def __call__(self, x):
        """The probability call with parameter set `x`."""
        diff = x - self.mu
        return -np.dot(diff, np.dot(self.icov, diff)) / 2.0

if __name__ == '__main__':
    main()
