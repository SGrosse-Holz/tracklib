"""
Currently just a graveyard for small functions, might be deprecated.
"""
import os,sys
import collections.abc

import numpy as np
from scipy.linalg import cholesky, toeplitz
import scipy.special

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
