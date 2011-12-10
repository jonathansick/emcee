# encoding: utf-8
"""


"""

from __future__ import division

__all__ = ['Ensemble', 'EnsembleSampler']

import numpy as np

import acor

from sampler import Sampler

class Ensemble(object):
    def __init__(self):
        self.positions = None

    def propose_position(self, ensemble):
        """
        Propose a new position for another ensemble given the current positions

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble that will be advanced.

        """
        pass

class EnsembleSampler(Sampler):
    """
    The base ensemble sampler based on Goodman & Weare

    Parameters
    ----------
    k : int
        The number of walkers.

    dim : int
        The dimension of the parameter space.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    ensemble_type : Ensemble, optional
        The type of ensemble to use.

    args : list, optional
        A list of arguments for lnprobfn.

    Notes
    -----
    The 'chain' member of this object has the shape: (k, nlinks, dim) where
    'nlinks' is the number of steps taken by the chain and 'k' is the number of
    walkers.  Use the 'flatchain' property to get the chain flattened to
    (nlinks, dim).

    """
    def __init__(self, k, *args, **kwargs):
        self.k = k
        self._ens = kwargs.pop('ensemble_type', Ensemble)
        self.a = kwargs.pop('a', 2.0)

        super(EnsembleSampler, self).__init__(*args, **kwargs)

    def do_reset(self):
        self.naccepted = np.zeros(self.k)
        # initialize 2 ensembles
        self._ensembles = [self._ens(), self._ens()]
        self._chain  = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

    def sample(self, p0, lnprob=None, randomstate=None, storechain=True, resample=1,
            iterations=1):
        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = np.array([self.get_lnprob(p[k]) for k in range(self.k)])

        # resize chain
        if storechain:
            N = int(iterations/resample)
            self._chain = np.concatenate((self._chain,
                    np.zeros((self.k, N, self.dim))), axis=1)
            self._lnprob = np.concatenate((self._lnprob, np.zeros((self.k, N))),
                    axis=1)

        i0 = self.iterations
        for i in xrange(int(iterations)):
            self.iterations += 1

            for k in xrange(self.k):
                # proposal
                zz = ((self.a-1.)*self._random.rand()+1)**2./self.a
                rint = self._random.randint(self.k-1)
                if rint >= k:
                    rint += 1
                q = p[rint] - zz*(p[rint]-p[k])

                newlnprob = self.get_lnprob(q)
                diff = (len(q) - 1.) * np.log(zz) + newlnprob - lnprob[k]
                accept = (diff > np.log(self._random.rand()))

                if accept:
                    p[k] = q
                    lnprob[k] = newlnprob
                    self.naccepted[k] += 1

                if storechain and i%resample == 0:
                    ind = i0 + int(i/resample)
                    self._chain[k,ind,:] = p[k]
                    self._lnprob[k,ind]  = lnprob[k]

            # heavy duty iterator action going on right here
            yield p, lnprob, self.random_state

    @property
    def flatchain(self):
        """
        Return the chain links flattened along the walker axis

        """
        s = self._chain.shape
        return self._chain.reshape(s[0]*s[1], s[2])

    @property
    def acor(self):
        s = self._chain.shape
        t = np.zeros((s[0], s[2]))
        for i in range(s[0]):
            t[i,:] = acor.acor(self._chain[i].T)[0]
        return t

if __name__ == '__main__':
    import pylab as pl
    ndim = 10
    N = 100

    mean = np.zeros(ndim)
    cov  = 0.5-np.random.rand(ndim*ndim).reshape((ndim,ndim))
    cov  = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov  = np.dot(cov,cov)
    icov = np.linalg.inv(cov)
    p0   = np.random.randn(ndim*N).reshape(N, ndim)

    def lnprob_gaussian(x, icov):
        return -np.dot(x,np.dot(icov,x))/2.0

    sampler = EnsembleSampler(100, ndim, lnprob_gaussian, args=[icov])
    for i in sampler.sample(p0, iterations=1e3, resample=10):
        pass

    print sampler.acceptance_fraction
    pl.plot(sampler.lnprobability[0,:])
    pl.figure()
    pl.plot(sampler.chain[0,:,:])

    pl.show()

