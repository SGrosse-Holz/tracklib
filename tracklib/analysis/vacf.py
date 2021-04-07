"""
A module for calculating velocity autocorrelations.

Follows the structure of `tracklib.analysis.msd`.
"""

import warnings

import numpy as np

from tracklib import Trajectory, TaggedSet

def VACFtraj(traj, dt=1):
    """
    Calculate/give velocity autocorrelation for a single trajectory.

    The result of the calculation is stored in ``traj.meta['VACF']`` and
    ``traj.meta['VACFmeta']``. This function checks whether the corresponding
    fields exist and if not calculates them. It then returns the value of the
    `!'VACF'` entry.

    Parameters
    ----------
    traj : Trajectory
        the trajectory whose velocity autocorrelation function to calculate
    dt : int, optional
        the time step to use for displacements. If ``dt > 1``, we average over
        the `!dt` realizations of the downsampling.

    Returns
    -------
    VACF : np.ndarray
        the velocity autocorrelation function of the trajectory. This is
        normalized such that ``VACF[0] = 1``.

    See also
    --------
    VACFdataset, VACF

    Notes
    -----
    This function expects a single-locus trajectory (``N=1``) and will raise a
    `!ValueError` otherwise. Preprocess accordingly.

    Usually this function should always be preferred to accessing
    ``traj.meta['VACF']`` directly.
    """
    if not traj.N == 1: # pragma: no cover
        raise ValueError("Preprocess your trajectory to have N=1")

    isknown = False
    try:
        isknown = traj.meta['VACFmeta']['dt'] == dt and 'VACF' in traj.meta.keys()
    except KeyError:
        isknown = False
    
    if not isknown:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

            acf = np.zeros(int(np.ceil(len(traj)/dt - 1)))
            Nacf = np.zeros(len(acf))
            N = np.zeros(len(acf))
            for off in range(dt):
                v = np.diff(traj[off::dt], axis=0)
                myacf = [np.nanmean(np.sum(v**2, axis=1), axis=0)] + [\
                    np.nanmean(np.sum(v[i:]*v[:-i], axis=1), axis=0) \
                    for i in range(1, v.shape[0])]
                myacf = np.array(myacf)/myacf[0]

                myN = [np.sum(~np.any(np.isnan(v), axis=1), axis=0)] + [\
                    np.sum(~np.any(np.isnan(v[i:]*v[:-i]), axis=1), axis=0) \
                    for i in range(1, v.shape[0])]

                acf[:len(myacf)] += myacf
                Nacf[:len(myacf)] += 1
                N[:len(myacf)] += myN

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
            traj.meta['VACF'] = acf / Nacf

        try:
            traj.meta['VACFmeta']['N'] = N
        except KeyError: # this will happen if 'VACFmeta' key doesn't exist
            traj.meta['VACFmeta'] = {'N' : N}
    
    return traj.meta['VACF']

def VACFdataset(dataset, dt=1, givevar=False, giveN=False, average_in_logspace=False):
    """
    Calculate ensemble VACF for the given dataset

    Parameters
    ----------
    dataset : TaggedSet
        a list of `Trajectory` for which to calculate an ensemble VACF
    dt : int, optional
        step size for displacements
    givevar : bool, optional
        whether to also return the variance around the mean
    giveN : bool, optional
        whether to return the sample size for each VACF data point
    average_in_logspace : bool, optional
        if ``True``, the averages used to calculate mean and variance will be
        taken in log-space, i.e. ``exp(mean(log(...)))`` instead of
        ``mean(...)``

    Returns
    -------
    vacf : np.ndarray
        the calculated ensemble mean
    var : np.ndarray, optional
        variance around the mean
    N : np.ndarray, optional
        number of data points going into each estimate

    See also
    --------
    VACFtraj, VACF

    Notes
    -----
    We normalize such that ``vacf[0] = 1``.
    """
    VACFs = [VACFtraj(traj, dt=dt) for traj in dataset]
    Ns = [traj.meta['VACFmeta']['N'] for traj in dataset]

    maxlen = max(len(VACF) for VACF in VACFs)
    allVACF = np.empty((len(VACFs), maxlen), dtype=float)
    allVACF[:] = np.nan
    allN = np.zeros((len(Ns), maxlen), dtype=int)
    for i, (vacf, N) in enumerate(zip(VACFs, Ns)):
        allVACF[i, :len(vacf)] = vacf
        allN[i, :len(N)] = N
    allN[np.where(np.isnan(allVACF))] = 0

    if average_in_logspace: # pragma: no cover
        allVACF = np.log(allVACF[:, 1:])
        N0 = np.sum(allN[:, 0])
        allN = allN[:, 1:]

    N = np.sum(allN, axis=0)
    meanN = N / np.sum(allN != 0, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message=r'(invalid value|divide by zero) encountered in true_divide')
        eVACF = np.nansum(allVACF*allN, axis=0) / N
        var = np.nansum((allVACF-eVACF)**2 * allN, axis=0) / (N-meanN)

    if average_in_logspace: # pragma: no cover
        eVACF = np.insert(np.exp(eVACF), 0, 0)
        var = np.insert(np.exp(var), 0, 0)
        N = np.insert(N, 0, N0)

    if givevar and giveN: # pragma: no cover
        return eVACF, var, N
    elif givevar:
        return eVACF, var
    elif giveN:
        return eVACF, N
    else:
        return eVACF

def VACF(*args, **kwargs):
    """
    Shortcut function to calculate VACFs.

    Will select either `VACFtraj` or `VACFdataset`, depending on the type of the
    first argument. Everything is then forwarded to that function.

    See also
    --------
    VACFtraj, VACFdataset
    """
    if issubclass(type(args[0]), Trajectory):
        return VACFtraj(*args, **kwargs)
    elif issubclass(type(args[0]), TaggedSet):
        return VACFdataset(*args, **kwargs)
    else: # pragma: no cover
        raise ValueError("Did not understand first argument, with type {}".format(type(args[0])))
