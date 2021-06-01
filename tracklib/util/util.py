"""
Currently just a graveyard for small functions, might be deprecated.
"""
import os,sys
import collections.abc

import numpy as np
from scipy.linalg import cholesky, toeplitz
import scipy.special

#NOTE: sort this a little better
def _twoLociRelativeACF(ts, A=1, B=1, d=1): # pragma: no cover
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

def _twoLociRelativeMSD(ts, *args, **kwargs): # pragma: no cover
    """
    A = σ^2 / √κ     (general prefactor)
    B = Δs^2 / 4κ       (tether length)
    """
    return 2*(twoLociRelativeACF(0, *args, **kwargs) - twoLociRelativeACF(ts, *args, **kwargs))
####################################################

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

def twoLocusMSD(t, Gamma, tau_c):
    """
    MSD for the distance between two loci on an infinite Rouse polymer

    Parameters
    ----------
    t : iterable
        the times at which to evaluate the MSD
    Gamma : float
        the prefactor for the polymer part (0.5 scaling) of the MSD. Note that
        this parameter should be the prefactor for tracking of one locus, i.e.
        the MSD produced by this function (which is the distance between two
        loci) will have a prefactor of ``2*Gamma``.
    tau_e : float
        the equilibration time. This is the time for which the two asymptotics
        (MSD ~ √t for short times, MSD ~ const at long times) have the same
        value. Note that this is a factor π^3/4 ~ 7.75 greater than the Rouse
        time of the tether between the loci.

    Returns
    -------
    (T,) np.array
        the MSD evaluated at times `!t`
    """
    with np.errstate(divide='ignore'):
        ret = 2 * Gamma * (
                np.sqrt(t)     * ( 1 - np.exp(-tau_c/(np.pi*t)) )
              + np.sqrt(tau_c) * scipy.special.erfc( np.sqrt(tau_e/(np.pi*t)) )
              )
    ret[np.isnan(ret)] = 0 # t = 0 produces the nan
    return ret

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
