"""
This module provides a useful way to sample from a stationary Gaussian process.

By assuming stationary, Gaussian, mean-zero increments, the process is uniquely
defined by its MSD. Here we use that observation to generate sample traces from
such processes, given the MSD.
"""

import os,sys
from copy import deepcopy

import numpy as np
from scipy.linalg import cholesky, toeplitz

from tracklib import Trajectory, TaggedSet

def sampleMSD(msd, n=1, isCorr=False, subtractMean=True):
    r"""
    Sample traces from a stationary Gaussian process with given MSD.

    Parameters
    ----------
    msd : (N, ) np.array
        the MSD from which to sample. We use the convention that ``msd[t]``
        should be the MSD at time lag ``t``, i.e. ``msd[0] = 0``.
    n : integer, optional
        Number of traces to generate
    isCorr : bool, optional
        whether the input array is MSD (default) or autocorrelation.
    subtractMean : bool
        whether to subtract the mean of each trace. See Notes below.

    Returns
    -------
    traj : (N, n) array
        The generated traces.

    See also
    --------
    dataset

    Notes
    -----
    Strictly speaking, only the ensemble of displacements is well-defined,
    because this is the one we assume steady state for. For the actual traces,
    we can add an arbitrary offset, and the reasonable thing to do here depends
    on the process we are sampling. For a purely diffusive process for example
    it makes sense to have them all start from zero, since there is no steady
    state anyways. If however, we are sampling from an MSD that plateaus at
    long times, we basically also have a steady state for the traces and thus
    the long-run mean converges.  In that case of course it makes sense to have
    this mean be identical for all traces. The option `!subtractMean` does
    exactly this: if ``True``, we subtract the mean from all generated traces,
    i.e. fix the ensemble mean to zero. Note that this is not precisely correct
    for finite-length traces, but probably the best we can do.

    **Algorithm:** from the MSD :math:`\mu(k)` we calculate the increment
    autocorrelation function :math:`\gamma(k)` as

    .. math:: \gamma(k) = \frac{1}{2} \left( \mu(|k+1|) + \mu(|k-1|) - 2\mu(|k|) \right) \,.

    The covariance matrix of the increment process we want to sample from is
    then the Toeplitz matrix with diagonals :math:`\gamma(k)`.
    """
    if not isCorr:
        msd[0] = 0
        msd = np.insert(msd, 0, msd[1])
        corr = 0.5 * (msd[2:] + msd[:-2] - 2*msd[1:-1])
    else: # pragma: no cover
        corr = msd

    try:
        L = cholesky(toeplitz(corr), lower=True)
    except np.linalg.LinAlgError: # pragma: no cover
        vals = np.linalg.eigvalsh(toeplitz(corr))
        raise RuntimeError("Correlation not positive definite. First 5 eigenvalues: {}".format(vals[:5]))
    steps = L @ np.random.normal(size=(len(corr), n))

    trajs = np.insert(np.cumsum(steps, axis=0), 0, 0, axis=0)
    if subtractMean:
        trajs = trajs - np.mean(trajs, axis=0)

    return trajs

def dataset(msd, N=1, Ts=None, d=3, **kwargs):
    """
    Generate a dataset of MSD sampled trajectories.

    All keyword arguments not mentioned below will be forwarded to `sampleMSD`.

    Parameters
    ----------
    msd : (N,) np.ndarray
        the MSD to sample from.
    N : int, optional
        number of particles per trajectory
    Ts : list of int, optional
        list of trajectory lengths, i.e. this determines number and length of
        trajectories. Any ``None`` entry will be replaced by the maximum possible
        value, ``len(msd)``. If there are values bigger than that, raises a
        `!ValueError`. If not specified, will default to 100 trajectories of
        maximum length.
    d : int, optional
        spatial dimension of the trajectories to sample

    Returns
    -------
    `TaggedSet` of `Trajectory`
        the generated data set.

    See also
    --------
    sampleMSD, tracklib.trajectory.Trajectory, tracklib.taggedset.TaggedSet

    Notes
    -----
    The input MSD is assumed to be the goal for the generated trajectories,
    i.e. incorporate the prefactor ``N*d``.
    """
    if Ts is None: # pragma: no cover
        Ts = 100*[None]

    Tmax = len(msd)
    for i, T in enumerate(Ts):
        if T is None:
            Ts[i] = Tmax
        elif T > Tmax: # pragma: no cover
            raise ValueError("Cannot sample trajectory of length {} from MSD of length {}".format(T, Tmax))
        elif T % 1 != 0: # pragma: no cover
            raise ValueError("Found non-integer trajectory length: {} (at index {})".format(T, i))

    # Timing execution of sampleMSD indicates that as long as we have a
    # reasonable distribution of lengths (e.g. exponential with mean 10% of
    # len(msd)), sampling all traces at the same time is faster, even though we
    # might generate a bunch of unused data.
    kwargs['n'] = len(Ts)*N*d
    traces = sampleMSD(msd/(N*d), **kwargs)

    def gen():
        for iT, T in enumerate(Ts):
            mytraces = [traces[:T, ((iT*N + n)*d):((iT*N + n+1)*d)] for n in range(N)]
            yield Trajectory.fromArray(mytraces)

    return TaggedSet(gen(), hasTags=False)

def control(dataset, msd=None):
    """
    Generate a stationary control data set to the one given.

    The control will look exactly like the original in all "meta"-aspects
    (number of trajectories, their length, any meta data, etc.), but the
    trajectories will be sampled from a stationary Gaussian process with the
    given MSD (or the empirical ensemble MSD of the original).

    The mean of each trajectory will be set to coincide with the mean of the
    sister trajectory it is generated from.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the dataset to generate a control for
    msd : (T,) np.ndarray, callable, or None
        the MSD to use for sampling, either directly (i.e. as array) or as
        function that will be evaluated on the needed lag times (where the unit
        of time is one frame). Note that this will be divided by
        (#loci)x(#dimensions) before sampling scalar traces, matching the usual
        notion of MSD of (e.g.) multidimensional trajectories.

    Returns
    -------
    `TaggedSet` of `Trajectory`
        the generated control data set.

    Notes
    -----
    Generation from empirical MSDs does not always work, since they might be
    noisy. In that case, provide a smoothed version as `!msd`.
    """
    if callable(msd):
        maxlen = max([len(traj) for traj in dataset])
        msd = msd(np.arange(maxlen))
    elif msd is None:
        from tracklib.analysis import MSD # bad style :( We cannot do the import at
                                          # import time (i.e. on top), because
                                          # the analysis module depends on the
                                          # models one.
        msd = MSD(dataset)
        del MSD

    N = dataset.map_unique(lambda traj : traj.N)
    d = dataset.map_unique(lambda traj : traj.d)
    msd = msd / (N*d) # reassign!

    def gen():
        for (traj, mytags) in dataset(giveTags=True):
            try:
                traces = sampleMSD(msd[:len(traj)], n=traj.N*traj.d, subtractMean=True)
            except np.linalg.LinAlgError: # pragma: no cover
                raise RuntimeError("Could not generate trajectories from provided (or ensemble) MSD. Try using something cleaner.")
            newdata = np.array([traces[:, (i*traj.d):((i+1)*traj.d)] for i in range(traj.N)])
            newdata += np.nanmean(traj.data, axis=1, keepdims=True)
            newdata[np.isnan(traj.data)] = np.nan

            newmeta = deepcopy(traj.meta)
            # Remove those meta entries that explicitly depend on the data
            for key in ['MSD', 'MSDmeta', 'chi2scores']:
                try:
                    del newmeta[key]
                except KeyError:
                    pass

            yield (Trajectory.fromArray(newdata, **newmeta), deepcopy(mytags))

    return TaggedSet(gen())
