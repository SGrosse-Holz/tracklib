"""
Everything to do with MSDs and calculating them.

Apart from immediately returning the most relevant results, the functions in
this module also store data in the Trajectory.meta dict. The fully populated
structure looks as follows (for a Trajectory traj):

traj.meta['MSD'] : np.ndarray
    the actual MSD of the trajectory
traj.meta['MSDmeta'] : dict
    some meta data for the MSD. Contains the following fields:
    'N' : np.ndarray
        the number of sample points for each point of the MSD
    'alpha' : float
        the fitted powerlaw exponent at the beginning of the MSD
    'logG' : float
        logarithm of the prefactor from the same fit as 'alpha'
    'fit_covariance' : (2, 2) np.ndarray
        covariance matrix of (alpha, logG) from the fit
"""

import numpy as np

import scipy.optimize

from tracklib import Trajectory, TaggedSet

def MSDtraj(traj):
    """
    Calculate/give MSD for a single trajectory.

    The result of the MSD calculation is stored in traj.meta['MSD'] and
    traj.meta['MSDmeta']. This function checks whether the corresponding fields
    exist and if not calculates them, in any case returning their entries at
    the end. So you can either use the return value of this function for
    further processing, or access the corresponding fields in traj.meta.

    Parameters
    ----------
    traj : tracklib.Trajectory
        the trajectory for which to calculate the MSD

    Returns
    -------
    MSD : np.ndarray
        the MSD of the trajectory, where MSD[0] = 0, MSD[1] is lag time of 1
        frame, etc.

    Writes
    ------
    traj.meta['MSD']
        the MSD that is also returned by the function
    traj.meta['MSDmeta']['N']
        for each data point in the MSD, how many points it is averaged over.
        This is mostly important for correct calculation of ensemble means.

    Notes
    -----
    This function expects a single-locus trajectory (N=1) and will raise a
    ValueError otherwise. Preprocess accordingly.

    Usually this function should always be preferred to accessing
    traj.meta['MSD'] directly.
    """
    if not traj.N == 1:
        raise ValueError("Preprocess your trajectory to have N=1")
    
    try:
        return traj.meta['MSD']
    except KeyError:
        traj.meta['MSD'] = np.array([0] + [ \
                np.nanmean(np.sum( (traj.get(slice(i, None), 'N') - traj.get(slice(None, -i), 'N'))**2, axis=1), axis=0) \
                for i in range(1, len(traj))])

        N = np.array([np.sum(~np.any(np.isnan(traj.get(slice(None, None), 'N')), axis=1), axis=0)] + \
                     [np.sum(~np.any(np.isnan(traj.get(slice(i, None), 'N') - traj.get(slice(None, -i), 'N')), axis=1), axis=0) \
                      for i in range(1, len(traj))])
        try:
            traj.meta['MSDmeta']['N'] = N
        except KeyError:
            traj.meta['MSDmeta'] = {'N' : N}
    
    return traj.meta['MSD']

def MSDdataset(dataset, giveN=False):
    """
    Calculate ensemble MSD for the given dataset

    Parameters
    ----------
    dataset : TaggedSet
        a list of Trajectory for which to calculate an ensemble MSD
    giveN : bool, optional
        whether to return the sample size for each MSD data point

    Returns
    -------
    msd / (msd, N) : np.ndarray / tuple of np.ndarray, see `giveN`
        either just the enseble MSD or the MSD and the number of samples for
        each data point

    Notes
    -----
    Corresponding to python's 0-based indexing, msd[0] = 0, such that
    msd[dt] is the MSD at a time lag of dt frames.
    """
    MSDs = [MSDtraj(traj) for traj in dataset]
    Ns = [traj.meta['MSDmeta']['N'] for traj in dataset]

    maxlen = max(len(MSD) for MSD in MSDs)
    eMSD = np.zeros(maxlen)
    eN = np.zeros(maxlen)

    for MSD, N in zip(MSDs, Ns):
        ind = N > 0
        eMSD[:len(MSD)][ind] += (MSD*N)[ind]
        eN[:len(MSD)][ind] += N[ind]
    eMSD /= eN

    if giveN:
        return (eMSD, eN)
    else:
        return eMSD

def MSD(*args, **kwargs):
    """
    Shortcut function to calculate MSDs.

    Will select either MSDtraj or MSDdataset, depending on the type of the
    first argument. Everything is then forwarded to the appropriate function.
    """
    if issubclass(type(args[0]), Trajectory):
        return MSDtraj(*args, **kwargs)
    elif issubclass(type(args[0]), TaggedSet):
        return MSDdataset(*args, **kwargs)
    else:
        raise ValueError("Did not understand first argument, with type {}".format(type(args[0])))

def scaling(traj, n=5):
    """
    Fit a powerlaw scaling to the first n data points of the MSD.

    Parameters
    ----------
    traj : Trajectory
        the trajectory whose MSD we are interested in
    n : int, optional
        how many data points of the MSD to use for fitting

    Returns
    -------
    alpha : float
        the fitted powerlaw scaling. For more details refer to
        traj.meta['MSDmeta']

    Writes
    ------
    traj.meta['MSDmeta']['alpha'] : float
        same as return value
    traj.meta['MSDmeta']['logG'] : float
        logarithm of the MSD prefactor
    traj.meta['MSDmeta']['fit_covariance'] : (2, 2) np.ndarray
        covariance matrix of (alpha, logG) from the fit

    Notes
    -----
    logG is known to be a bad estimator for diffusivities.

    "first n points of the MSD" means time lags 1 through n.
    """
    nMSD = MSDtraj(traj)[1:(n+1)]
    t = np.arange(len(nMSD))+1
    ind = ~np.isnan(nMSD)
    if np.sum(ind) < 2:
        traj.meta['MSDmeta'].update({
                'alpha' : np.nan,
                'logG' : np.nan,
                'fit_covariance' : np.nan*np.ones((2, 2)),
                })
    else:
        popt, pcov = scipy.optimize.curve_fit(lambda x, a, b : x*a + b, np.log(t[ind]), np.log(nMSD[ind]))
        traj.meta['MSDmeta'].update({
                'alpha' : popt[0],
                'logG' : popt[1],
                'fit_covariance' : pcov,
                })

    return traj.meta['MSDmeta']['alpha']
