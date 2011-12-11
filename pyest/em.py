# encoding: utf-8
"""


"""

from __future__ import division

__all__ = ['EMEnsemble']

import numpy as np

import mixtures
from ensemble import Ensemble

class EMEnsemble(Ensemble):
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 10)
        super(EMEnsemble, self).__init__(*args, **kwargs)

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

        mixture = mixtures.MixtureModel(self.K, c)
        mixture.run_kmeans(verbose=False)
        mixture.run_em(verbose=False, regularization=1e-10, maxiter=2)

        q = mixture.sample(Ns)
        newlnprob = self.get_lnprob(q)

        newQ = mixture.lnprob(q)
        oldQ = mixture.lnprob(s)

        lnpdiff = oldQ - newQ + newlnprob - ensemble.lnprob
        accept = (lnpdiff > np.log(self._sampler._random.rand(len(lnpdiff))))

        return q, newlnprob, accept

