#!/usr/bin/env python
# encoding: utf-8
"""
Rosenbrock... whatever.

"""

from __future__ import division

__all__ = ['']

import pylab as pl
import numpy as np
from pyest import DualEnsembleSampler, EMEnsemble
from matplotlib.patches import Ellipse


def lnposterior(p):
    return -(100*(p[1]-p[0]*p[0])**2+(1-p[0])**2)/20.0

nwalkers = 100
p0 = np.array([-8,-10])+np.array([16,70])*np.random.rand(nwalkers*2).reshape(nwalkers,2)

# sampler = DualEnsembleSampler(nwalkers, 2, lnposterior,
#                 ensemble_type=EMEnsemble, ensemble_args={'K': 10})

sampler = DualEnsembleSampler(nwalkers, 2, lnposterior)

if False:
    es = []
    cs = ['r', 'b']

    line1,  = pl.plot(p0[:len(p0)/2,0], p0[:len(p0)/2,1], '.%s'%(cs[0]), zorder=-2, alpha=0.3)
    line2,  = pl.plot(p0[len(p0)/2:,0], p0[len(p0)/2:,1], '.%s'%(cs[1]), zorder=-2, alpha=0.3)

    pl.xlim([-8, 8])
    pl.ylim([-10,60])
    pl.draw()

    for i, (pos,lnprob,state) in enumerate(sampler.sample(p0,None,None,
                                            iterations=100)):
        if sampler.ensemble_type == EMEnsemble:
            if len(es) > 0:
                [e.remove() for e in es]
            es = []

            for j, ensemble in enumerate(sampler.ensembles):
                for k in range(ensemble.K):
                    x,y = ensemble.mixture.means[0,k],ensemble.mixture.means[1, k]
                    U,S,V = np.linalg.svd(ensemble.mixture._cov[k])
                    theta = np.degrees(np.arctan2(U[1,0], U[0,0]))
                    es.append(Ellipse(xy=[x,y], width=2*np.sqrt(S[0]),
                        height=2*np.sqrt(S[1]), angle=theta,
                        facecolor='none', edgecolor=cs[j],lw=2))
                    ax = pl.gca()
                    ax.add_patch(es[-1])

        line1.set_xdata(pos[:len(pos)/2,0])
        line1.set_ydata(pos[:len(pos)/2,1])
        line2.set_xdata(pos[len(pos)/2:,0])
        line2.set_ydata(pos[len(pos)/2:,1])
        pl.draw()
else:
    for pos,lnprob,state in sampler.sample(p0,None,None, iterations=100000, resample=1000):
        if sampler.iterations % 1000 == 0:
            print sampler.iterations

flatchain = sampler.flatchain
acor = sampler.acor
print np.mean(acor, axis=0)

pl.xlim([-8, 8])
pl.ylim([-10,60])

pl.plot(flatchain[0,:], flatchain[1,:], '.k')
pl.savefig("ensemble.png")


