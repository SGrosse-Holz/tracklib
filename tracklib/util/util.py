"""
Random assortment of useful auxiliary stuff
"""
import os,sys
import collections.abc

import numpy as np
from scipy.linalg import cholesky, toeplitz
import scipy.special
import scipy.stats

__all__ = [
    "distribute_noiselevel",
    "log_derivative",
    "KM_survival",
]

def distribute_noiselevel(noise2, pixelsize):
    """
    Get coordinate-wise localization error from scalar noise level

    Parameters
    ----------
    noise2 : float
        the total noise level. In terms of coordinate-wise errors this is
        ``noise2 = Δx^2 + Δy^2 + Δz^2``
    pixelsize : array-like
        linear extension of pixel/voxel in each direction

    Returns
    -------
    localization_error : (d,) np.array
        localization error distributed to the coordinate directions,
        proportionately to the pixel size, i.e. ``Δx``, ``Δy``, ``Δz`` such
        that ``noise2 = Δx^2 + Δy^2 + Δz^2``.
    """
    voxel = np.asarray(pixelsize)
    noise_in_px = np.sqrt(noise2/np.sum(voxel**2))
    return noise_in_px*voxel

def log_derivative(y, x=None, resampling_density=2):
    """
    Calculate loglog-derivative.

    We resample the given data to log-spaced x.

    Parameters
    ----------
    y : array-like
        the function values whose derivative we are interested in
    x : array-like, optional
        the independent variable for the data in y. Will default to
        ``np.arange(len(y))`` (and thus ignore the first data point).
    resampling_density : float, optional
        how tight to space the log-resampled points. A value of 1 corresponds
        to the spacing between the first two data points, higher values
        decrease spacing.

    Returns
    -------
    x : np.array
        the log-resampled abscissa
    dlog : np.array
        the calculated log-derivative
    """
    if x is None:
        x = np.arange(len(y))

    with np.errstate(divide='ignore'):
        x = np.log(x)
        y = np.log(y)
    ind_valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[ind_valid]
    y = y[ind_valid]

    dlogx = (x[1] - x[0])/resampling_density
    xnew = np.arange(np.min(x), np.max(x), dlogx)
    ynew = scipy.interpolate.interp1d(x, y)(xnew)

    return np.exp(xnew[:-1] + dlogx/2), np.diff(ynew)/dlogx

def KM_survival(data, censored, conf=0.95, Tmax=np.inf, S1at=0):
    """
    Kaplan-Meier survival estimator on censored data

    This is the standard survival estimator for right-censored data, i.e. data
    points that are marked as "censored" enter the estimate as ``t_true > t``.

    Parameters
    ----------
    data : (N,) array-like
        individual survival times
    censored : (N,) array-like, boolean
        indicate for each data point whether it is right-censored or not.
    conf : float in (0, 1), optional
        the confidence bounds on the survival curve to calculate
    Tmax : float
        can be used to compute survival only up to some time ``Tmax``.
    S1at : float, optional
        give a natural lower limit for survival times where S = 1.

    Returns
    -------
    (T, 4) array
        the columns of this array are: t, S(t), l(t), u(t), where l and u are
        lower and upper confidence levels respectively.
    """
    data = np.asarray(data)
    censored = np.asarray(censored).astype(bool)

    t = np.unique(data[~censored]) # unique also sorts
    t = t[t <= Tmax]
    S = np.zeros(len(t)+1)
    S[0] = 1
    V = np.zeros(len(t)+1)
    Vsum = 0
    for n, curt in enumerate(t, start=1):
        d_n = np.count_nonzero(data[~censored] == curt)
        N_n = np.count_nonzero(data >= curt)

        S[n] = S[n-1]*(1-d_n/N_n)
        if N_n > d_n:
            Vsum += d_n/(N_n*(N_n-d_n))
            V[n] = np.log(S[n])**(-2)*Vsum
        else:
            Vsum += np.inf
            V[n] = 0

    z = scipy.stats.norm().ppf((1-conf)/2)
    lower = S**(np.exp( z*np.sqrt(V)))
    upper = S**(np.exp(-z*np.sqrt(V)))

    if S1at is not None:
        t = np.insert(t, 0, S1at)
    else:
        S = S[1:]
        lower = lower[1:]
        upper = upper[1:]

    return np.stack([t, S, lower, upper], axis=-1)

## Remember to update __all__ if adding to this
