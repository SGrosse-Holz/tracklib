import os,sys
import collections.abc

import numpy as np
from scipy.linalg import cholesky, toeplitz
import scipy.special

### Has been replaced by analysis.msd.MSDtraj()
# def msd(traj, giveN=False, warnd=True):
#     """
#     MSD calculation for a (T, d, ...) array. Nan-robust, i.e. missing data
#     can/should be indicated by np.nan
# 
#     Input
#     -----
#     traj : (T, d, ...) array-like
#         the time lapse trajectory to calculate the MSD for. We assume that the
#         first dimension is time and the second dimension carries the spatial
#         index of the trajectory, the remaining dimensions remain as they are,
#         i.e. can be used to calculate MSDs for multiple trajectories at once.
#     giveN : boolean
#         whether to return msd or (msd, N), N being an array indicating the
#         number of data points used for each time lag. This is important for
#         example to calculate ensemble averages.
#         default: False
#     warnd : boolean
#         if d (number of spatial dimensions) is greater than 3, there's a high
#         probability that there was a mistake, so we throw an error. This
#         behavior can be disabled by setting warnd=False.
#         default: True
# 
#     Output
#     ------
#     msd : (T, ...) numpy.array
#         we use the convention that msd[0] = MSD(Δt=0) = 0, i.e. the time lags
#         corresponding to the entries in msd run from 0 to T-1.
#     N : (T, ...) numpy.array, optional (controlled by giveN)
#         the number of non-nan contributions to any specific time lag.
#     """
#     traj = np.array(traj)
#     if len(traj.shape) < 2:
#         # This is a single, 1d trajectory
#         traj = np.expand_dims(traj, 1)
#     if traj.shape[1] > 3 and warnd:
#         raise ValueError("Dimension 1 of trajectory should be the spatial index but was greater than 3. If this is okay, set warnd=False.")
# 
#     msd = np.array(np.zeros((1, *traj.shape[2:])).tolist() \
#                    + [np.nanmean(np.sum( (traj[i:] - traj[:-i])**2 , axis=1), axis=0) \
#                       for i in range(1, len(traj))])
#     if giveN:
#         N = np.array([np.sum(~np.any(np.isnan(traj), axis=1), axis=0)] + \
#                      [np.sum(~np.any(np.isnan(traj[i:] - traj[:-i]), axis=1), axis=0) \
#                       for i in range(1, len(traj))])
#         return msd, N
#     else:
#         return msd

#NOTE: sort this a little better
def twoLociRelativeACF(ts, A=1, B=1, d=1):
    """
    A = σ^2 / √κ     (general prefactor)
    B = Δs^2 / 4κ    (tether length)
    """
    # Scipy's implementation of En can only deal with integer n
    def E32(z):
        return 2*np.exp(-z) - 2*np.sqrt(np.pi*z)*scipy.special.erfc(np.sqrt(z))

    if not isinstance(ts, collections.abc.Iterable):
        ts = [ts]

    return np.array([ d*A*( np.sqrt(B) - np.sqrt(t/np.pi)*( 1 - 0.5*E32(B/t) ) ) if t != 0 else d*A*np.sqrt(B) for t in ts])

def twoLociRelativeMSD(ts, *args, **kwargs):
    """
    A = σ^2 / √κ     (general prefactor)
    B = Δs^2 / 4κ       (tether length)
    """
    return 2*(twoLociRelativeACF(0, *args, **kwargs) - twoLociRelativeACF(ts, *args, **kwargs))
