import os,sys
from copy import deepcopy

import numpy as np
from scipy.linalg import cholesky, toeplitz

from tracklib import Trajectory, TaggedList

def sampleMSD(msd, n=1, isCorr=False, subtractMean=True):
    """
    Sample trajectories from a Gaussian process with zero-mean stationary
    increments with given autocorrelation function γ(k). Usually the more
    intuitive quantity is the MSD of the process, so by default we expect the
    MSD μ(k) to be given. They are related by
    ```
        γ(k) = 1/2 * ( μ(|k+1|) + μ(|k-1|) - 2*μ(|k|) ) .
    ```

    Inputs
    ------
    msd : (N, ) array
        The msd from which to sample. The trajectories will have length N-1 and
        msd[0] should be 0, such that generally msd[k] is the MSD for a time
        lag of k*Δt. If isCorr = True, then this is expected to be the
        autocorrelation function already, instead of the MSD. In this case, the
        trajectories will have length N.
    n : integer
        Number of trajectories to generate
        default: 1
    isCorr : whether the input array is MSD or autocorrelation.
        default: False, i.e. input is MSD.
    subtractMean : whether to subtract the mean of each trajectory. See Notes
        below
        default: True

    Output
    ------
    traj : (N, n) array
        The generated trajectories

    Notes
    -----
    To produce a trajectory of length N from an MSD of length N, we assume the
    very last data point of the autocorrelation function to be identical to the
    second to last.
    Strictly speaking, only the ensemble of displacements is well-defined,
    because this is the one we assume steady state for. For the actual
    trajectories, we can add an arbitrary offset, and the reasonable thing to
    do here depends on the process we are sampling. For purely diffusive
    trajectories for example it makes sense to have them all start from zero,
    since there is no steady state anyways. If however, we are sampling from an
    MSD that plateaus at long times, we basically also have a steady state for
    the trajectories and we can reasonably talk about the mean of a trajectory.
    In that case of course it makes sense to keep this mean constant. The
    option subtractMean does exactly this: if True, we subtract the mean from
    all generated trajectories, i.e. fix the ensemble mean to zero.
    """
    if not isCorr:
        msd[0] = 0
        msd = np.insert(msd, 0, msd[1])
        corr = 0.5 * (msd[2:] + msd[:-2] - 2*msd[1:-1])
        corr = np.append(corr, corr[-1])
    else:
        corr = msd

    L = cholesky(toeplitz(corr), lower=True)
    steps = L @ np.random.normal(size=(len(corr), n))

    trajs = np.cumsum(steps, axis=0)
    if subtractMean:
        trajs = trajs - np.mean(trajs, axis=0)

    return trajs

def dataset(msd, N=2, Ts=None, d=3, **kwargs):
    """
    Generate a dataset of MSD sampled trajectories.

    Input
    -----
    msd : 1d numpy.ndarray
        the MSD to sample from.
    N : integer
        number of particles per trajectory
        default: 2
    Ts : list of int
        list of trajectory lengths, i.e. this determines number and length of
        trajectories. Any None entry will be replaced by the maximum possible
        value, len(msd). If there are values bigger than that, raises a
        ValueError.
        default: 100*[None]
    d : integer
        spatial dimension of the trajectories to sample
        default: 3
    other kwargs are forwarded to util.sampleMSD()

    Output
    ------
    A TaggedList of trajectories (with only the trivial tag '_all').
    """
    if Ts is None:
        Ts = 100*[None]

    Tmax = len(msd)
    for i, T in enumerate(Ts):
        if T is None:
            Ts[i] = Tmax
        elif T > Tmax:
            raise ValueError("Cannot sample trajectory of length {} from MSD of length {}".format(T, Tmax))

    # Timing execution of sampleMSD indicates that as long as we have a
    # reasonable distribution of lengths (e.g. exponential with mean 10% of
    # len(msd)), sampling all traces at the same time is faster, even though we
    # might generate a bunch of unused data.
    kwargs['n'] = len(Ts)*N*d
    traces = util.sampleMSD(msd, **kwargs)

    def gen():
        for iT, T in enumerate(Ts):
            mytraces = [traces[:T, ((iT*N + n)*d):((iT*N + n+1)*d)] for n in range(N)]
            yield (Trajectory.fromArray(mytraces), [])

    return TaggedList.generate(gen())

def control(dataset, msd=None, setMean='copy'):
    """
    Generate a sister data set where each trajectory is sampled from a
    stationary Gaussian process with MSD equal to the ensemble mean of the
    given data set or the explicitly given MSD. Note generation from
    experimental data (i.e. the ensemble mean) does not always work, because
    that is noisy. Thus the option to provide a cleaned version.

    The mean of each trajectory will be set to coincide with the mean of the
    sister trajectory it is generated from.

    Input
    -----
    dataset : TaggedList of Trajectory
        the dataset to generate a control for
    msd : (T,) np.ndarray
        the MSD to use for sampling. Note that this will be divided by
        (#loci)x(#dimensions) before sampling scalar traces, matching the usual
        notion of MSD of (e.g.) multidimensional trajectories.

    Output
    ------
    A TaggedList that's an MSD generated sister data set to the input

    Implementation Note
    -------------------
    Should this be merged/unified with MSDdataset?
    """
    if msd is None:
        msd = analysis.MSD(dataset)

    msd /= dataset.getHom('N')*dataset.getHom('d')

    def gen():
        for (traj, mytags) in dataset(giveTags=True):
            try:
                traces = util.sampleMSD(msd[:len(traj)], n=traj.N*traj.d)
            except np.linalg.LinAlgError:
                raise RuntimeError("Could not generate trajectories from provided (or ensemble) MSD. Try to use something cleaner.")
            newdata = np.array([traces[:, (i*traj.d):((i+1)*traj.d)] for i in range(traj.N)])
            newdata += np.mean(traj[:], axis=1, keepdims=True)

            newtraj = Trajectory.fromArray(newdata, **deepcopy(traj.meta))

            if len(traj) != len(newtraj):
                print("traj : {}, newtraj : {} entries".format(len(traj), len(newtraj)))

            yield (newtraj, deepcopy(mytags))

    return TaggedList.generate(gen())
