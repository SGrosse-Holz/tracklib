"""
Random assortment of useful auxiliary stuff
"""
import os,sys
import collections.abc

import numpy as np
from scipy.linalg import cholesky, toeplitz
import scipy.special
import scipy.stats

# This is imported in the util module as *, so don't clutter
__all__ = [
    "distribute_noiselevel",
    "log_derivative",
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
