"""
This module implements an analysis method to check stationarity.

Still under construction.

The basic idea is to test the null hypothesis "the trajectories are generated
from a process with stationary Gaussian increments". Under this hypothesis, we know that snippets of the displacement trajectory should follow a multivariate Gaussian, whose covariance matrix we can calculate from the MSD. We can thus calculate a :math:`\chi^2` statistic for each of these snippets, and check that they follow the expected :math:`\chi^2` distribution.
"""

import numpy as np
from matplotlib import pyplot as plt

import scipy.linalg
import scipy.stats

from tracklib import Trajectory, TaggedSet
from .p2 import MSD

def chi2vsMSD(dataset, n=10, msd=None):
    """
    Calculate snippet-wise chi2 scores assuming the given MSD.

    The actual scores will be written to ``traj.meta['chi2scores']`` for each `!traj`
    in `!dataset`. This function then returns just the number of degrees of
    freedom that the reference chi2 distribution should have.
    
    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the trajectories to use
    n : int
        the window size / length of the snippets to look at
    msd : (T,) array-like
        the MSD to use for defining the reference Gaussian process. If omitted,
        this will be the empirical `MSD` of the data, as given by
        ``MSD(dataset)``.

    Returns
    -------
    dof : int
        number of degrees of freedom for reference distribution.

    See also
    --------
    summary_plot, tracklib.analysis.msd.MSD
    """
    d = dataset.map_unique(lambda traj : traj.d)

    if msd is None: # pragma: no cover
        msd = MSD(dataset)

    msd = np.insert(msd, 0, msd[1])
    corr = 0.5*(msd[2:] + msd[:-2] - 2*msd[1:-1])
    corr = corr / d # Correct for the fact that MSD is summed over dimensions
    maG = scipy.linalg.inv(scipy.linalg.toeplitz(corr[:n]))

    for traj in dataset:
        if traj.N == 1:
            displacements = traj.diff()
        elif traj.N == 2:
            displacements = traj.relative().diff()
        else: # pragma: no cover
            raise ValueError("Don't know what to do with trajectories with N = {}", traj.N)

        def chi2score(i):
            snip = displacements[i:(i+n)]
            return np.sum(np.diag( snip.T @ maG @ snip ))

        traj.meta['chi2scores'] = np.array([chi2score(i) for i in range(len(traj)-n)])

    return n*d

def summary_plot(dataset, dof=None, p=0.05, ax=None, **kwargs):
    """
    Produce a summary plot of chi2 scores.

    Histogram the chi-square scores calculated with `chi2vsMSD`. This assumes
    that each trajectory has a metadata field `!'chi2scores'`. Optionally also
    plots significance thresholds and/or the expected chi2 distribution.

    Any keyword arguments not listed below will be forwarded to `!plt.hist`.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the data to use
    dof : int
        degrees of freedom for the reference chi2 distribution. If omitted, no
        reference will be shown.
    p : float in (0, 1)
        significance level at which to draw cutoffs. Set to ``None`` to prevent
        plotting of significance cutoffs.
    ax : handle
        the axes handle into which to plot. Will be set to plt.gca() if
        omitted.

    See also
    --------
    chi2vsMSD
    """
    if ax is None:
        ax = plt.gca()

    scores = []
    for traj in dataset:
        scores += traj.meta['chi2scores'].tolist()
    scores = np.array(scores)

    preferences = {
            'bins' : 'auto',
            'density' : True,
            'histtype' : 'step'
        }
    for key in preferences.keys():
        if not key in kwargs.keys():
            kwargs[key] = preferences[key]

    ax.hist(scores[~np.isnan(scores)], **kwargs)

    if dof is not None:
        xplot = np.linspace(0, np.nanmax(scores), 1000)
        ax.plot(xplot, np.insert(scipy.stats.chi2.pdf(xplot[1:], dof), 0, 0), color='red', label='expected chi2')

        if p is not None:
            thres = scipy.stats.chi2.ppf(p, dof)
            ax.axvline(x=thres, color='magenta')
            thres = scipy.stats.chi2.isf(p, dof)
            ax.axvline(x=thres, color='magenta')
