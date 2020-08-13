import os,sys

import numpy as np
from scipy.linalg import cholesky, toeplitz

def msd(traj, giveN=False, warnd=True):
    """
    MSD calculation for a (T, d, ...) array. Nan-robust, i.e. missing data
    can/should be indicated by np.nan

    Input
    -----
    traj : (T, d, ...) array-like
        the time lapse trajectory to calculate the MSD for. We assume that the
        first dimension is time and the second dimension carries the spatial
        index of the trajectory, the remaining dimensions remain as they are,
        i.e. can be used to calculate MSDs for multiple trajectories at once.
    giveN : boolean
        whether to return msd or (msd, N), N being an array indicating the
        number of data points used for each time lag. This is important for
        example to calculate ensemble averages.
        default: False
    warnd : boolean
        if d (number of spatial dimensions) is greater than 3, there's a high
        probability that there was a mistake, so we throw an error. This
        behavior can be disabled by setting warnd=False.
        default: True

    Output
    ------
    msd : (T, ...) numpy.array
        we use the convention that msd[0] = MSD(Δt=0) = 0, i.e. the time lags
        corresponding to the entries in msd run from 0 to T-1.
    N : (T, ...) numpy.array, optional (controlled by giveN)
        the number of non-nan contributions to any specific time lag.
    """
    traj = np.array(traj)
    if len(traj.shape) < 2:
        # This is a single, 1d trajectory
        traj = np.expand_dims(traj, 1)
    if traj.shape[1] > 3 and warnd:
        raise ValueError("Dimension 1 of trajectory should be the spatial index but was greater than 3. If this is okay, set warnd=False.")

    msd = np.array(np.zeros((1, *traj.shape[2:])).tolist() \
                   + [np.nanmean(np.sum( (traj[i:] - traj[:-i])**2 , axis=1), axis=0) \
                      for i in range(1, len(traj))])
    if giveN:
        N = np.array([np.sum(~np.any(np.isnan(traj), axis=1), axis=0)] + \
                     [np.sum(~np.any(np.isnan(traj[i:] - traj[:-i]), axis=1), axis=0) \
                      for i in range(1, len(traj))])
        return msd, N
    else:
        return msd

def sampleMSD(msd, n=1, isCorr=False, subtractMean=True):
    """
    Sample trajectories from a Gaussian process with zero-mean stationary
    increments with given autocorrelation function. Usually the more intuitive
    quantity is the MSD, so by default we expect the MSD μ(k) to be given. They are
    related by
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
