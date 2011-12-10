# encoding: utf-8
"""


"""

from __future__ import division

__all__ = ['Ensemble', 'EnsembleSampler', 'DualEnsembleSampler']

import numpy as np

import acor

from sampler import Sampler

class Ensemble(object):
    def __init__(self, sampler):
        self._sampler = sampler

    def get_lnprob(self, pos=None):
        if pos is None:
            p = self.pos
        else:
            p = pos

        if self._sampler.pool is not None:
            M = self._sampler.pool.map
        else:
            M = map
        lnprob = np.array(M(self._sampler.get_lnprob, [p[i]
                    for i in range(len(p))]))

        return lnprob

    def propose_position(self, ensemble):
        """
        Propose a new position for another ensemble given the current positions

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble that will be advanced.

        """
        s = np.atleast_2d(ensemble.pos)
        Ns = len(s)
        c = np.atleast_2d(self.pos)
        Nc = len(c)

        zz = ((self._sampler.a-1.)*self._sampler._random.rand(Ns)+1)**2./self._sampler.a
        rint = self._sampler._random.randint(Nc, size=(Ns,))

        # propose new walker position and calculate the lnprobability
        q = c[rint] - zz[:,np.newaxis]*(c[rint]-s)
        newlnprob = ensemble.get_lnprob(q)

        lnpdiff = (self._sampler.dim - 1.) * np.log(zz) + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self._sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept

class SingleGaussianEnsemble(Ensemble):
    def propose_position(self, ensemble):
        """
        Propose a new position for another ensemble given the current positions

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble that will be advanced.

        """
        s = np.atleast_2d(ensemble.pos)
        Ns = len(s)
        c = np.atleast_2d(self.pos)
        Nc = len(c)

        mu  = np.mean(c, axis=0)
        cov = np.cov(c, rowvar=0)

        # propose new walker position and calculate the lnprobability
        q = self._sampler._random.multivariate_normal(mu,cov,size=Ns)
        newlnprob = ensemble.get_lnprob(q)

        diff = q-mu
        newQ = -0.5*np.dot(diff, np.linalg.solve(cov, diff.T)).diagonal()
        diff = s-mu
        oldQ = -0.5*np.dot(diff, np.linalg.solve(cov, diff.T)).diagonal()

        lnpdiff = oldQ - newQ + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self._sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept

class EnsembleSampler(Sampler):
    """
    The vanilla ensemble sampler based on Goodman & Weare

    Parameters
    ----------
    k : int
        The number of walkers.

    dim : int
        The dimension of the parameter space.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    a : float, optional
        The Goodman & Weare "a" scale parameter. (default: 2.0)

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
        self.a = kwargs.pop('a', 2.0)
        super(EnsembleSampler, self).__init__(*args, **kwargs)

    def do_reset(self):
        self.naccepted = np.zeros(self.k)
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
        s = self.chain.shape
        return self.chain.reshape(s[0]*s[1], s[2])

    @property
    def acor(self):
        s = self.chain.shape
        t = np.zeros((s[0], s[2]))
        for i in range(s[0]):
            t[i,:] = acor.acor(self.chain[i].T)[0]
        return t

class DualEnsembleSampler(EnsembleSampler):
    """
    A generalized Ensemble sampler that uses 2 ensembles

    Parameters
    ----------
    k : int
        The number of walkers. Must be a multiple of 2.

    dim : int
        The dimension of the parameter space.

    lnprobfn : callable
        A function that computes the probability of a particular point in phase
        space.  Will be called as lnprobfn(p, *args)

    a : float, optional
        The Goodman & Weare "a" scale parameter. (default: 2.0)

    ensemble_type : callable, optional
        The constructor for the Ensemble type that will be used. (default:
        Ensemble)

    args : list, optional
        A list of arguments for lnprobfn.

    """
    def __init__(self, *args, **kwargs):
        self.ensemble_type = kwargs.pop('ensemble_type', Ensemble)
        self.pool = kwargs.pop('pool', None)

        super(DualEnsembleSampler, self).__init__(*args, **kwargs)
        assert self.k%2 == 0

    def do_reset(self):
        self.ensembles = [self.ensemble_type(self), self.ensemble_type(self)]

        self.naccepted = np.zeros(self.k)
        self._chain  = np.empty((self.k, 0, self.dim))
        self._lnprob = np.empty((self.k, 0))

    def sample(self, p0, lnprob=None, randomstate=None, storechain=True, resample=1,
            iterations=1):
        self.random_state = randomstate

        p = np.array(p0)
        halfk = int(self.k/2)
        self.ensembles[0].pos = p[:halfk]
        self.ensembles[1].pos = p[halfk:]
        if lnprob is not None:
            self.ensembles[0].lnprob = lnprob[:halfk]
            self.ensembles[1].lnprob = lnprob[halfk:]
        else:
            lnprob = np.zeros(self.k)
            for k, ens in enumerate(self.ensembles):
                ens.lnprob = ens.get_lnprob()
                lnprob[halfk*k:halfk*(k+1)] = ens.lnprob

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

            for k, ens in enumerate(self.ensembles):
                q, newlnprob, accept = self.ensembles[(k+1)%2].propose_position(ens)
                fullaccept = np.zeros(self.k,dtype=bool)
                fullaccept[halfk*k:halfk*(k+1)] = accept
                if np.any(accept):
                    lnprob[fullaccept] = newlnprob[accept]
                    p[fullaccept] = q[accept]

                    # update ensemble too
                    ens.pos[accept] = q[accept]
                    ens.lnprob[accept] = newlnprob[accept]

                    self.naccepted[fullaccept] += 1

            if storechain and i%resample == 0:
                ind = i0 + int(i/resample)
                self._chain[:,ind,:] = p
                self._lnprob[:,ind]  = lnprob

            # heavy duty iterator action going on right here
            yield p, lnprob, self.random_state

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

    sampler = DualEnsembleSampler(100, ndim, lnprob_gaussian, args=[icov])
    for i in sampler.sample(p0, iterations=1e3, resample=10):
        pass

    print sampler.acceptance_fraction
    pl.plot(sampler.lnprobability[0,:])
    pl.figure()
    pl.plot(sampler.chain[0,:,:])

    pl.show()

