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
    if not traj.N == 1:
        raise ValueError("Preprocess your trajectory to have N=1")

    isknown = False
    try:
        isknown = traj.meta['VACFmeta']['dt'] == dt and 'VACF' in traj.meta.keys()
    except KeyError:
        isknown = False
    
    if not isknown:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')

            acf = np.zeros(int(np.ceil(len(traj)/dt - 1) + 1))
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

def VACFdataset(dataset, giveN=False, dt=1):
    """
    Calculate ensemble VACF for the given dataset

    Parameters
    ----------
    dataset : TaggedSet
        a list of `Trajectory` for which to calculate an ensemble VACF
    giveN : bool, optional
        whether to return the sample size for each VACF data point
    dt : int
        the time lag to use for calculating displacements

    Returns
    -------
    vacf / (vacf, N) : np.ndarray / tuple of np.ndarray, see `!giveN`
        either just the enseble VACF or the VACF and the number of samples for
        each data point

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
    eVACF = np.zeros(maxlen)
    eN = np.zeros(maxlen)

    for VACF, N in zip(VACFs, Ns):
        ind = N > 0
        eVACF[:len(VACF)][ind] += (VACF*N)[ind]
        eN[:len(VACF)][ind] += N[ind]

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='invalid value encountered in true_divide')
        eVACF /= eN

    if giveN:
        return (eVACF, eN)
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
    else:
        raise ValueError("Did not understand first argument, with type {}".format(type(args[0])))
