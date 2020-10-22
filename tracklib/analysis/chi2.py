import numpy as np
from matplotlib import pyplot as plt

import scipy.linalg
import scipy.stats

from tracklib import Trajectory, TaggedList

def chi2vsMSD(dataset, n=10, msd=None):
    """
    Calculate the snippet-wise chi-square score for trajectories vs. a given
    MSD. This can be used to check for statistical homogeneity, within single
    trajectories and across the dataset.
    
    Input
    -----
    dataset : TaggedList of Trajectory
        the trajectories to use
    n : integer
        the window size / length of the snippets to look at
    msd : array-like, longer than n
        the MSD to use for defining the reference Gaussian process. If omitted,
        this will be the empirical MSD of the data, as given by MSD(dataset).
        Note: we assume that this MSD is for trajectories like the one in the
        dataset, i.e. if you want to specify diffusive motion in d-dimensions,
        use 2*d*D*t, not 2*D*t.

    Output
    ------
    The calculated chi-square scores will be written to the trajectory metadata
    as 'chi2scores'.
    This function returns the #dof for the reference chi2 distribution (for use
    with scipy.stats.chi2)
    """
    d = dataset.getHom('d')

    if msd is None:
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
        else:
            raise ValueError("Don't know what to do with trajectories with N = {}", traj.N)

        def chi2score(i):
            snip = displacements[i:(i+n)]
            return np.sum(np.diag( snip.T @ maG @ snip ))

        traj.meta['chi2scores'] = np.array([chi2score(i) for i in range(len(traj)-n)])

    return n*d

def summary_plot(dataset, dof=None, p=0.05, ax=None, **kwargs):
    """
    Histogram the chi-square scores calculated with chi2vsMSD(). This assumes
    that each trajectory has a metadata field 'chi2scores'. Optionally also
    plots significance thresholds and/or the expected chi2 distribution.

    Input
    -----
    dataset : TaggedList of Trajectory
        the data to use
    dof : integer
        degrees of freedom for the reference chi2 distribution. If omitted, no
        reference will be shown.
    p : float in (0, 1)
        significance level at which to draw cutoffs. Can be set to None to
        prevent the plotting of these lines.
        default: 0.05
    ax : the axes handle into which to plot.
        default: plt.gca()
    Additional kwargs will be forwarded to plt.hist() for plotting
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
        ax.plot(xplot, scipy.stats.chi2.pdf(xplot, dof), color='red', label='expected chi2')

        if p is not None:
            thres = scipy.stats.chi2.ppf(p, dof)
            ax.axvline(x=thres, color='magenta')
            thres = scipy.stats.chi2.isf(p, dof)
            ax.axvline(x=thres, color='magenta')
