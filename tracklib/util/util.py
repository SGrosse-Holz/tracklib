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
