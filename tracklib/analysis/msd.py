import numpy as np

import scipy.optimize

from tracklib import Trajectory, TaggedList

def MSD(dataset, giveN=False, memo=True):
    """
    Calculate ensemble MSD for the given dataset

    Input
    -----
    dataset : TaggedList (possibly with some selection set)
        a list of Trajectory for which to calculate an ensemble MSD
    giveN : bool
        whether to return the sample size for each MSD data point
    memo : bool
        whether to use the memoization of Trajectory.msd()
        default: True

    Output
    ------
    if giveN:
        a tuple (msd, N) of (T,) arrays containing MSD and sample size
        respectively
    if not giveN:
        only msd, i.e. a (T,) array.

    Notes
    -----
    Corresponding to python's 0-based indexing, msd[0] = 0, such that
    msd[dt] is the MSD at a time lag of dt frames.
    """
    msdNs = [traj.msd(giveN=True, memo=memo) for traj in dataset]

    maxlen = max(len(msdN[0]) for msdN in msdNs)
    emsd = msdNs[0][0]
    npad = [(0, maxlen-len(emsd))] + [(0, 0) for _ in emsd.shape[2:]]
    emsd = np.pad(emsd, npad, constant_values=0)
    eN = np.pad(msdNs[0][1], npad, constant_values=0)
    emsd *= eN

    for msd, N in msdNs[1:]:
        ind = N > 0
        emsd[:len(msd)][ind] += (msd*N)[ind]
        eN[:len(N)][ind] += N[ind]
    emsd /= eN

    if giveN:
        return (emsd, eN)
    else:
        return emsd

def fit_scaling(traj, n=5):
    """
    Fit a powerlaw scaling to the first n data points of the MSD.

    Input
    -----
    traj : Trajectory
        the trajectory whose MSD we are interested in
    n : integer
        how many data points of the MSD to use for fitting

    Output
    ------
    None. The results of the fit are written to traj.meta['MSDscaling']. This
    will be a dict with keys 'alpha', 'logG', 'cov' for respectively the
    exponent, log of the prefactor, covariance matrix from the fit (cov[0, 0]
    is the variance of alpha, cov[1, 1] that for logG).

    Notes
    -----
     - logG is known to be a bad estimator for diffusivities.
     - "first n points of the MSD" means time lags 1 through n.
    """
    nmsd = traj.msd()[1:(n+1)]
    t = np.arange(len(nmsd))+1
    ind = ~np.isnan(nmsd)
    if np.sum(ind) < 2:
        traj.meta['MSDscaling'] = {
                'alpha' : np.nan,
                'logG' : np.nan,
                'cov' : np.nan*np.ones((1, 1)),
                }
    else:
        popt, pcov = scipy.optimize.curve_fit(lambda x, a, b : x*a + b, np.log(t[ind]), np.log(nmsd[ind]))
        traj.meta['MSDscaling'] = {
                'alpha' : popt[0],
                'logG' : popt[1],
                'cov' : pcov,
                }
