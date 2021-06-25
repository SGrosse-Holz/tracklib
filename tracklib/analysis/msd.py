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

DEFKEY='MSD'

def MSDtraj(traj, TA=True, recalculate=False, function='SD', writeto=DEFKEY):
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
    recalculate : bool
        set to ``True`` to ensure that the MSD is actually calculated, instead
        of reusing an old value
    function : {'SD', 'D', 'SP'} or callable
        the function to evaluate. This extends the machinery of the `!msd`
        module to arbitrary two-point functions: generically, `!function`
        should be ``fun(traj[m], traj[n]) --> float``, where ``m >= n``, and it
        should be vectorized (i.e. work on numpy arrays and return the
        corresponding arrays). The available presets are
         - "Square Dispacement" ``'SD'`` : ``fun(xm, xn) = (xm-xn)**2``. This
           gives MSD.
         - "Displacement" ``'D'`` : ``fun(xm, xn) = xm-xn``. No squaring,
           mostly for sanity checks. Note peculiarities here though: if
           ``TA=True``, the averaging collapses in a telescope sum such that
           e.g. ``<x(t+1)-x(t)> = (x(T)-x(0))/(T-1)``.
         - "Scalar Product" ``'SP'`` : ``fun(xm, xn) = xm.xn``. Produces
           autocorrelation.
    writeto : hashable or None
        where to store the output of the calculation in the ``traj.meta`` dict.
        Set to ``None`` to return the output dict instead of storing it.
        Defaults to ``'MSD'``.

    Returns
    -------
    dict, optional
        A dict with keys ``'data', 'N'`` giving the averaged data and the count
        of valid data points for each lag time. This is usually written into
        the ``traj.meta`` dict, but returned if ``writeto is None``.

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
    >>> del traj.meta[writeto]
    """
    if recalculate:
        for key in [writeto]:
            try:
                del traj.meta[key]
            except:
                pass
    
    if writeto not in traj.meta.keys():

        if not callable(function):
            if traj.N != 1: # pragma: no cover
                raise ValueError("Preprocess your trajectory to have N=1")

            if function == 'SD':
                def function(xm, xn):
                    return np.sum((xm-xn)**2, axis=-1)
            elif function == 'D':
                def function(xm, xn):
                    return np.sum(xm-xn, axis=-1)
            elif function == 'SP':
                def function(xm, xn):
                    return np.sum(xm*xn, axis=-1)
            else:
                raise ValueError("invalid argument 'function' : {}".format(function))

        if TA:
            data = [function(traj[:], traj[:])]
            data += [function(traj[i:], traj[:-i]) for i in range(1, len(traj))]
        else:
            istart = np.min(np.nonzero(~np.any(np.isnan(traj.data), (0, 2)))[0])
            data = [[function(traj[i], traj[istart])] for i in range(istart, len(traj))]

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

            out = {
                'N' : np.array([np.count_nonzero(~np.isnan(dat)) for dat in data]),
                'data' : np.array([np.nanmean(dat) for dat in data]),
            }

        if writeto is None:
            return out
        else:
            traj.meta[writeto] = out

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
    try:
        msdkey = kwargs['writeto']
    except KeyError:
        msdkey = DEFKEY

    for traj in dataset:
        MSDtraj(traj, **kwargs)

    MSDs = [traj.meta[msdkey]['data'] for traj in dataset]
    Ns = [traj.meta[msdkey]['N'] for traj in dataset]

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
        if givevar:
            var = np.nansum((allMSD-eMSD)**2 * allN, axis=0) / (N-meanN)

    if average_in_logspace: # pragma: no cover
        eMSD = np.insert(np.exp(eMSD), 0, 0)
        if givevar:
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
        MSDtraj(*args, **kwargs)
        if 'writeto' in kwargs:
            writeto = kwargs['writeto']
        else:
            writeto = DEFKEY
        return args[0].meta[writeto]['data']
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
    nMSD = MSD(traj)[1:(n+1)]
    t = np.arange(len(nMSD))+1
    ind = ~np.isnan(nMSD)
    if np.sum(ind) < 2:
        traj.meta['MSD'].update({
                'alpha' : np.nan,
                'logG' : np.nan,
                'fit_covariance' : np.nan*np.ones((2, 2)),
                })
    else:
        popt, pcov = scipy.optimize.curve_fit(lambda x, a, b : x*a + b, np.log(t[ind]), np.log(nMSD[ind]))
        traj.meta['MSD'].update({
                'alpha' : popt[0],
                'logG' : popt[1],
                'fit_covariance' : pcov,
                })

    return traj.meta['MSD']['alpha']
