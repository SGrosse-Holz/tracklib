"""
Everything to do with MSDs and calculating them.

Many results of this module will be stored in `Trajectory.meta` for further
use. For an overview, see :ref:`here <traj_meta_fields_msd>`.
"""

import warnings

import numpy as np

import scipy.optimize
import scipy.interpolate

from tracklib import Trajectory, TaggedSet
from tracklib.util.util import log_derivative

def MSDtraj(traj, TA=True, exponent=2, recalculate=False):
    """
    Calculate/give MSD for a single trajectory.

    The result of the MSD calculation is stored in ``traj.meta['MSD']`` and
    ``traj.meta['MSDmeta']``. This function checks whether the corresponding
    fields exist and if not calculates them. It then returns the value of the
    `!'MSD'` entry. To avoid this behavior, use ``recalculate = True``.

    Parameters
    ----------
    traj : Trajectory
        the trajectory for which to calculate the MSD
    TA : bool, optional
        whether to time-average

    Other Parameters
    ----------------
    exponent : int or float
        the exponent to use. Mostly for debugging
    recalculate : bool
        set to ``True`` to ensure that the MSD is actually calculated, instead
        of reusing an old value

    Returns
    -------
    MSD : np.ndarray
        the MSD of the trajectory, where ``MSD[0] = 0``, ``MSD[1]`` is lag time
        of 1 frame, etc.

    See also
    --------
    MSDdataset, MSD

    Notes
    -----
    This function expects a single-locus trajectory (``N=1``) and will raise a
    `!ValueError` otherwise. Preprocess accordingly.

    This function should always be preferred to accessing ``traj.meta['MSD']``
    directly.

    Explicitly, the ``recalculate`` parameter is equivalent to
    >>> del traj.meta['MSD']
    ... del traj.meta['MSDmeta']
    """
    if recalculate:
        for key in ['MSD', 'MSDmeta']:
            try:
                del traj.meta[key]
            except:
                pass
    
    try:
        return traj.meta['MSD']
    except KeyError:

        if not traj.N == 1: # pragma: no cover
            raise ValueError("Preprocess your trajectory to have N=1")

        if exponent == 1:
            def S(val):
                return val
        elif exponent == 2:
            def S(val):
                return val*val
        else:
            def S(val):
                return val**exponent

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

            if TA:
                traj.meta['MSD'] = np.array([0] + [ \
                        np.nanmean(np.sum( S(traj[i:] - traj[:-i]) , axis=1), axis=0) \
                        for i in range(1, len(traj))])

                N = np.array([np.sum(~np.any(np.isnan(traj[:]), axis=1), axis=0)] + \
                             [np.sum(~np.any(np.isnan(traj[i:] - traj[:-i]), axis=1), axis=0) \
                              for i in range(1, len(traj))])
            else:
                traj.meta['MSD'] = np.array([0] + [ \
                        np.sum( S(traj[0] - traj[i]) ) \
                        for i in range(1, len(traj))])

                N = (~np.isnan(traj.meta['MSD'])).astype(int)

        try:
            traj.meta['MSDmeta']['N'] = N
        except KeyError: # this will happen if 'MSDmeta' key doesn't exist
            traj.meta['MSDmeta'] = {'N' : N}
    
    return traj.meta['MSD']

def MSDdataset(dataset, givevar=False, giveN=False, average_in_logspace=False, **kwargs):
    """
    Calculate ensemble MSD for the given dataset

    Parameters
    ----------
    dataset : TaggedSet
        a list of `Trajectory` for which to calculate an ensemble MSD

    Other Parameters
    ----------------
    givevar : bool, optional
        whether to also return the variance around the mean
    giveN : bool, optional
        whether to return the sample size for each MSD data point
    average_in_logspace : bool, optional
        set to ``True`` to replace the arithmetic with a geometric mean.
    kwargs : keyword arguments
        are all forwarded to forwarded to `MSDtraj`, see that docstring.

    Returns
    -------
    msd : np.ndarray
        the calculated ensemble mean
    var : np.ndarray, optional
        variance around the mean
    N : np.ndarray, optional
        number of data points going into each estimate

    See also
    --------
    MSDtraj, MSD

    Notes
    -----
    Corresponding to python's 0-based indexing, ``msd[0] = 0``, such that
    ``msd[dt]`` is the MSD at a time lag of `!dt` frames.
    """
    MSDs = [MSDtraj(traj, **kwargs) for traj in dataset]
    Ns = [traj.meta['MSDmeta']['N'] for traj in dataset]

    maxlen = max(len(MSD) for MSD in MSDs)
    allMSD = np.empty((len(MSDs), maxlen), dtype=float)
    allMSD[:] = np.nan
    allN = np.zeros((len(Ns), maxlen), dtype=int)
    for i, (msd, N) in enumerate(zip(MSDs, Ns)):
        allMSD[i, :len(msd)] = msd
        allN[i, :len(N)] = N
    allN[np.where(np.isnan(allMSD))] = 0

    if average_in_logspace: # pragma: no cover
        allMSD = np.log(allMSD[:, 1:])
        N0 = np.sum(allN[:, 0])
        allN = allN[:, 1:]

    N = np.sum(allN, axis=0)
    meanN = N / np.sum(allN != 0, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message=r'(invalid value|divide by zero) encountered in true_divide')
        eMSD = np.nansum(allMSD*allN, axis=0) / N
        var = np.nansum((allMSD-eMSD)**2 * allN, axis=0) / (N-meanN)

    if average_in_logspace: # pragma: no cover
        eMSD = np.insert(np.exp(eMSD), 0, 0)
        var = np.insert(np.exp(var), 0, 0)
        N = np.insert(N, 0, N0)

    if givevar and giveN: # pragma: no cover
        return eMSD, var, N
    elif givevar:
        return eMSD, var
    elif giveN:
        return eMSD, N
    else:
        return eMSD

def MSD(*args, **kwargs):
    """
    Shortcut function to calculate MSDs.

    Will select either `MSDtraj` or `MSDdataset`, depending on the type of the
    first argument. Everything is then forwarded to that function.

    See also
    --------
    MSDtraj, MSDdataset, tl.util.util.log_derivative
    """
    if issubclass(type(args[0]), Trajectory):
        return MSDtraj(*args, **kwargs)
    elif issubclass(type(args[0]), TaggedSet):
        return MSDdataset(*args, **kwargs)
    else: # pragma: no cover
        raise ValueError("Did not understand first argument, with type {}".format(type(args[0])))

def dMSD(*args, resampling_density=2, **kwargs):
    """
    Shortcut to calculate time-dependent MSD scaling

    This is a shortcut for ``tracklib.util.util.log_derivative(MSD(<input>))``

    See also
    --------
    tracklib.util.util.log_derivative
    """
    return log_derivative(MSD(*args, **kwargs), resampling_density=resampling_density)

def scaling(traj, n=5):
    """
    Fit a powerlaw scaling to the first `!n` data points of the MSD.

    Results will be written to ``traj.meta['MSDmeta']``, only the exponent
    `!alpha` will be returned.

    Parameters
    ----------
    traj : Trajectory
        the trajectory whose MSD we are interested in
    n : int, optional
        how many data points of the MSD to use for fitting

    Returns
    -------
    alpha : float
        the fitted powerlaw scaling. For more details check
        ``traj.meta['MSDmeta']``

    See also
    --------
    MSDtraj, MSDdataset, MSD

    Notes
    -----
    `!logG` is known to be a bad estimator for diffusivities.

    "first `!n` points of the MSD" means time lags 1 through `!n`.
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
