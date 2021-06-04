"""
Main module of the bild inference package
"""

import numpy as np

from . import mcmc

def main(traj, model, MCMCconfig,
         priorscheme=None, sampler=None,
         return_='nothing', # or 'traj', 'dict'
         return_mcmcrun=False,
         ):
    """
    How to run this scheme on one trajectory
    """
    if priorscheme is None:
        priorscheme = mcmc.GeometricPriorScheme(model.nStates)
    if sampler is None:
        sampler = mcmc.FullMCMC()

    sampler.setup(traj, model, priorscheme)
    sampler.configure(**MCMCconfig)
    myrun = sampler.run()

    out = {'valid' : len(myrun.logLs) < sampler.config['iterations']}
    (out['loopingtrace'], out['prior_params']), _ = myrun.best_sample_logL()
    if return_mcmcrun:
        out['mcmcrun'] = myrun

    if return_ == 'dict':
        return out
    else:
        traj.meta['bild'] = out
        if return_ == 'traj':
            return traj
        else:
            return None
