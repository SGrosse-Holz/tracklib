"""
Calculating two-point functions, like MSD & ACF

Note that all of these (MSD, ACF, VACF, ...) contain the same information. See
also :py:mod:`msdfit <tracklib.analysis.msdfit>`.

Many results of this module will be stored in `Trajectory.meta` for further
use. For an overview, see :ref:`here <traj_meta_fields_msd>`.
"""

import warnings

import numpy as np

from tracklib import Trajectory, TaggedSet

DEFKEY='P2'

def P2traj(traj, TA=True, recalculate=False,
           function=None, preproc=None, postproc=None,
           writeto=DEFKEY,
           ):
    """
    Calculate a two-point function for a single trajectory

    The results are stored in the ``traj.meta[writeto]``. By default, if this
    field exists already, the function aborts. To recalculate and overwrite,
    set ``recalculate = True``.

    Parameters
    ----------
    traj : Trajectory
        the trajectory for which to calculate the MSD
    TA : bool, optional
        whether to time average

    Other Parameters
    ----------------
    recalculate : bool
        set to ``True`` to ensure that the calculation is actually performed
    function : {'SD', 'D', 'SP'} or callable
        the function to evaluate. Should be ``fun(traj[m], traj[n]) -->
        float``, where ``m >= n``, and it should be vectorized (i.e. work on
        numpy arrays and return the corresponding arrays).
    preproc : callable or None
        will be applied to the trajectory before processing. Example: ``lambda
        traj: traj.diff()`` gives the increment trajectory
    postproc : callable or None
        will be applied to the final result, i.e. should take a (T,) array and
        return such. Relevant when accessing this function through
        ``P2dataset``, since in that case this `!postproc` is applied _before_
        averaging. Use case: normalizing correlation functions
    writeto : hashable or None
        where to store the output of the calculation in the ``traj.meta`` dict.
        Set to ``None`` to return the output dict instead of storing it.
        Defaults to ``'P2'``.

    Returns
    -------
    dict, optional
        A dict with keys ``'data', 'N'`` giving the averaged data and the count
        of valid data points for each lag time. This is usually written into
        the ``traj.meta`` dict, but returned if ``writeto is None``.

    See also
    --------
    P2dataset, MSD, tracklib.analysis.msdfit

    Notes
    -----
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

        if callable(preproc):
            proc_traj = preproc(traj)
        else:
            proc_traj = traj

        if TA:
            data = [function(proc_traj[:], proc_traj[:])]
            data += [function(proc_traj[i:], proc_traj[:-i]) for i in range(1, len(proc_traj))]
        else:
            istart = np.min(np.nonzero(~np.any(np.isnan(proc_traj.data), (0, 2)))[0])
            data = [[function(proc_traj[i], proc_traj[istart])] for i in range(istart, len(proc_traj))]

        del proc_traj # just for clarity

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

            out = {
                'N' : np.array([np.count_nonzero(~np.isnan(dat)) for dat in data]),
                'data' : np.array([np.nanmean(dat) for dat in data]),
            }
            if callable(postproc):
                out['data'] = postproc(out['data'])

        if writeto is None:
            return out
        else:
            traj.meta[writeto] = out

def P2dataset(dataset, givevar=False, giveN=False, average_in_logspace=False, **kwargs):
    """
    Ensemble average two-point functions

    Parameters
    ----------
    dataset : TaggedSet
        a list of `Trajectory`

    Other Parameters
    ----------------
    givevar : bool, optional
        whether to also return the variance around the mean
    giveN : bool, optional
        whether to return the sample size for each MSD data point
    average_in_logspace : bool, optional
        set to ``True`` to replace the arithmetic with a geometric mean.
    kwargs : keyword arguments
        are all forwarded to forwarded to `P2traj`, see that docstring.

    Returns
    -------
    eP2 : np.ndarray
        the calculated ensemble mean
    var : np.ndarray, optional
        variance around the mean
    N : np.ndarray, optional
        number of data points going into each estimate

    See also
    --------
    P2traj, MSD, ACF
    """
    try:
        p2key = kwargs['writeto']
    except KeyError:
        p2key = DEFKEY

    for traj in dataset:
        P2traj(traj, **kwargs)

    P2s = [traj.meta[p2key]['data'] for traj in dataset]
    Ns = [traj.meta[p2key]['N'] for traj in dataset]

    maxlen = max(len(P2) for P2 in P2s)
    allP2 = np.empty((len(P2s), maxlen), dtype=float)
    allP2[:] = np.nan
    allN = np.zeros((len(Ns), maxlen), dtype=int)
    for i, (P2, N) in enumerate(zip(P2s, Ns)):
        allP2[i, :len(P2)] = P2
        allN[i, :len(N)] = N
    allN[np.where(np.isnan(allP2))] = 0

    if average_in_logspace: # pragma: no cover
        allP2 = np.log(allP2[:, 1:])
        N0 = np.sum(allN[:, 0])
        allN = allN[:, 1:]

    N = np.sum(allN, axis=0)
    meanN = N / np.sum(allN != 0, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message=r'(invalid value|divide by zero) encountered in true_divide')
        eP2 = np.nansum(allP2*allN, axis=0) / N
        if givevar:
            var = np.nansum((allP2-eP2)**2 * allN, axis=0) / (N-meanN)

    if average_in_logspace: # pragma: no cover
        eP2 = np.insert(np.exp(eP2), 0, 0)
        if givevar:
            var = np.insert(np.exp(var), 0, 0)
        N = np.insert(N, 0, N0)

    if givevar and giveN: # pragma: no cover
        return eP2, var, N
    elif givevar:
        return eP2, var
    elif giveN:
        return eP2, N
    else:
        return eP2

def P2(*args, **kwargs):
    """
    Shortcut function to calculate P2s.

    Will select either `P2traj` or `P2dataset`, depending on the type of the
    first argument. Everything is then forwarded to that function.

    This is mostly just a template for the following functions MSD, ACF, ...

    See also
    --------
    P2traj, P2dataset
    """
    if issubclass(type(args[0]), Trajectory):
        P2traj(*args, **kwargs)
        if 'writeto' in kwargs:
            writeto = kwargs['writeto']
        else:
            writeto = DEFKEY
        return args[0].meta[writeto]['data']
    elif issubclass(type(args[0]), TaggedSet):
        return P2dataset(*args, **kwargs)
    else: # pragma: no cover
        raise ValueError("Did not understand first argument, with type {}".format(type(args[0])))

def MSD(*args, **kwargs):
    """
    Calculate MSD for a `Trajectory` or dataset.

    This is an implementation of `P2`. See `P2traj` and `P2dataset` for
    detailed description of parameters, here we will just list the most
    interesting / relevant ones

    Parameters
    ----------
    in : Trajectory or TaggedSet of Trajectory
        the first argument is the input data, either a single `Trajectory`, or
        a dataset that will be ensemble averaged over
    TA : bool, optional
        whether to time average on (each) single trajectory
    recalculate : bool, optional
        set to ``True`` to ensure that the calculation is actually performed.
        Otherwise, this function might just return precalculated values from
        ``traj.meta['MSD']``.
    giveN : bool, optional
        only applicable when `!in` is a dataset. Whether to also return the
        sample size for each MSD data point.

    Returns
    -------
    msd : np.ndarray
        the calculated MSD. Note that ``msd[i] = MSD(iΔt)``, i.e. ``msd[0] =
        0``.
    N : np.ndarray, optional
        the sample size for each point, see `!giveN`.

    See also
    --------
    P2traj, P2dataset, tracklib.analysis.msdfit
    """
    def SD(xm, xn):
        return np.sum((xm-xn)**2, axis=-1)

    return P2(*args, **kwargs, function=SD, writeto='MSD')

def ACov(*args, **kwargs):
    """
    Calculate autocovariance for a `Trajectory` or dataset.

    This is an implementation of `P2`. See `P2traj` and `P2dataset` for
    detailed description of parameters, here we will just list the most
    interesting / relevant ones

    Parameters
    ----------
    in : Trajectory or TaggedSet of Trajectory
        the first argument is the input data, either a single `Trajectory`, or
        a dataset that will be ensemble averaged over
    TA : bool, optional
        whether to time average on (each) single trajectory
    recalculate : bool, optional
        set to ``True`` to ensure that the calculation is actually performed.
        Otherwise, this function might just return precalculated values from
        ``traj.meta['ACov']``.
    giveN : bool, optional
        only applicable when `!in` is a dataset. Whether to also return the
        sample size for each data point.

    Returns
    -------
    acov : np.ndarray
        the calculated autocovariance
    N : np.ndarray, optional
        the sample size for each point, see `!giveN`

    See also
    --------
    MSD, P2traj, P2dataset, tracklib.analysis.msdfit
    """
    def SP(xm, xn):
        return np.sum(xm*xn, axis=-1)

    return P2(*args, **kwargs, function=SP, writeto='ACov')

def ACorr(*args, **kwargs):
    """
    Calculate autocorrelation for a `Trajectory` or dataset.

    This is an implementation of `P2`. See `P2traj` and `P2dataset` for
    detailed description of parameters, here we will just list the most
    interesting / relevant ones

    Parameters
    ----------
    in : Trajectory or TaggedSet of Trajectory
        the first argument is the input data, either a single `Trajectory`, or
        a dataset that will be ensemble averaged over
    TA : bool, optional
        whether to time average on (each) single trajectory
    recalculate : bool, optional
        set to ``True`` to ensure that the calculation is actually performed.
        Otherwise, this function might just return precalculated values from
        ``traj.meta['ACorr']``.
    giveN : bool, optional
        only applicable when `!in` is a dataset. Whether to also return the
        sample size for each data point.

    Returns
    -------
    acorr : np.ndarray
        the calculated autocorrelation
    N : np.ndarray, optional
        the sample size for each point, see `!giveN`

    See also
    --------
    MSD, P2traj, P2dataset, tracklib.analysis.msdfit
    """
    def SP(xm, xn):
        return np.sum(xm*xn, axis=-1)
    def normalize(data):
        return data / data[0]

    return P2(*args, **kwargs, function=SP, postproc=normalize, writeto='ACorr')

def VACov(*args, **kwargs):
    """
    Calculate velocity autocovariance for a `Trajectory` or dataset.

    This is an implementation of `P2`. See `P2traj` and `P2dataset` for
    detailed description of parameters, here we will just list the most
    interesting / relevant ones

    Parameters
    ----------
    in : Trajectory or TaggedSet of Trajectory
        the first argument is the input data, either a single `Trajectory`, or
        a dataset that will be ensemble averaged over
    TA : bool, optional
        whether to time average on (each) single trajectory
    recalculate : bool, optional
        set to ``True`` to ensure that the calculation is actually performed.
        Otherwise, this function might just return precalculated values from
        ``traj.meta['VACov']``.
    giveN : bool, optional
        only applicable when `!in` is a dataset. Whether to also return the
        sample size for each data point.

    Returns
    -------
    vacov : np.ndarray
        the calculated velocity autocovariance
    N : np.ndarray, optional
        the sample size for each point, see `!giveN`

    See also
    --------
    MSD, P2traj, P2dataset, tracklib.analysis.msdfit
    """
    def SP(xm, xn):
        return np.sum(xm*xn, axis=-1)

    return P2(*args, **kwargs, function=SP, preproc=lambda traj: traj.diff(), writeto='VACov')

def VACorr(*args, **kwargs):
    """
    Calculate velocity autocorrelation for a `Trajectory` or dataset.

    This is an implementation of `P2`. See `P2traj` and `P2dataset` for
    detailed description of parameters, here we will just list the most
    interesting / relevant ones

    Parameters
    ----------
    in : Trajectory or TaggedSet of Trajectory
        the first argument is the input data, either a single `Trajectory`, or
        a dataset that will be ensemble averaged over
    TA : bool, optional
        whether to time average on (each) single trajectory
    recalculate : bool, optional
        set to ``True`` to ensure that the calculation is actually performed.
        Otherwise, this function might just return precalculated values from
        ``traj.meta['VACorr']``.
    giveN : bool, optional
        only applicable when `!in` is a dataset. Whether to also return the
        sample size for each data point.

    Returns
    -------
    vacorr : np.ndarray
        the calculated velocity autocorrelation
    N : np.ndarray, optional
        the sample size for each point, see `!giveN`

    See also
    --------
    MSD, P2traj, P2dataset, tracklib.analysis.msdfit
    """
    def SP(xm, xn):
        return np.sum(xm*xn, axis=-1)
    def normalize(data):
        return data / data[0]

    return P2(*args, **kwargs, function=SP, preproc=lambda traj: traj.diff(), postproc=normalize, writeto='VACorr')
