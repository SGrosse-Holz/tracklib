"""
Everything to do with priors
"""

import abc

import numpy as np
import scipy.stats

from .util import Loopingprofile, ParametricFamily

class Prior(metaclass=abc.ABCMeta):
    """
    Abstract base class for priors

    When subclassing, you need to provide the `logpi` method, which should
    return the (log) prior probability for an input `Loopingprofile`. Specifying
    `logpi_vectorized` is optional, if you can speed up the prior calculation
    over an iterable of loopingprofiles.
    """
    @abc.abstractmethod
    def logpi(self, profile):
        """
        log of prior probability

        Parameters
        ----------
        profile : Loopingprofile

        Returns
        -------
        float

        See also
        --------
        logpi_vectorized
        """
        raise NotImplementedError # pragma: no cover

    def logpi_vectorized(self, profiles):
        """
        Evaluate the prior on multiple loopingprofiles

        By default just sequentially evaluates `logpi` on the given
        loopingprofiles, which of course does not give a speedup.

        Parameters
        ----------
        profiles : Sequence (e.g. list) of Loopingprofile

        Returns
        -------
        np.ndarray, dtype=float

        See also
        --------
        logpi
        """
        return np.array([self.logpi(profile) for profile in profiles])

class UniformPrior(Prior):
    r"""
    A uniform prior over loopingprofiles

    This is simply :math:`-N\log(n)` for each `Loopingprofile`, where :math:`n`
    is the number of states and :math:`N` the number of frames.
    """
    def __init__(self, nStates=2):
        self.logn = np.log(nStates)

    def logpi(self, profile):
        return -len(profile)*self.logn

class GeometricPrior(Prior):
    r"""
    A geometric prior over #switches in the `Loopingprofile`

    Writing :math:`\theta` for the `Loopingprofile` and :math:`k(\theta)` for the
    number of switches therein, this is given by

    .. math:: \pi(\theta) = \frac{1}{n}(1+(n-1)q)^{N-1} q^{k(\theta)}\,,

    where :math:`q\in(0, 1]` is the one parameter of this prior, :math:`n` is
    the number of possible states and :math:`N` is the number of (valid) frames
    in the trajectory.

    Parameters
    ----------
    logq : float < 0
        log of the parameter q
    nStates : int, optional
        the number of possible states

    Attributes
    ----------
    logq : float
    n : int
    """
    def __init__(self, logq=0, nStates=2):
        self.logq = logq
        self.n = nStates

        self._log_n = np.log(self.n)
        with np.errstate(under='ignore'):
            self._log_norm_per_dof = np.log(1+np.exp(self.logq)*(self.n-1))

    def logpi(self, profile):
        # optimized
        return profile.count_switches()*self.logq - (len(profile)-1)*self._log_norm_per_dof  - self._log_n

    def logpi_vectorized(self, profiles):
        profiles = np.array([profile[:] for profile in profiles])
        ks = np.count_nonzero(profiles[:, 1:] != profiles[:, :-1], axis=1)
        return ks*self.logq - (profiles.shape[1]-1)*self._log_norm_per_dof - self._log_n

class UniformKPrior_NONORM(Prior):
    """
    A prior that has uniform distribution of #switches
    """
    def __init__(self, nStates=2):
        self.p = 1 - 1/nStates

    def logpi(self, profile):
        k = profile.count_switches()
        return -scipy.stats.binom(n=len(profile)-1, p=self.p).logpmf(k)

class FixedSwitchPrior(Prior):
    """
    Prior for a fixed number of switches
    """
    def __init__(self, K, nStates=2):
        self.K = K
        self.n = nStates
        self._log_norm = self.K*np.log(self.n*(self.n-1))

    def logpi(self, profile):
        k = profile.count_switches()
        if k == self.K:
            N = len(loopingprofile)
            return -np.log(scipy.special.binom(N-1, k)) - self._log_norm
        else:
            return -np.inf

class MaxSwitchPrior(Prior):
    """
    Prior for limiting number of switches
    """
    def __init__(self, K, logq=0, nStates=2):
        self.K = K
        self.logq = logq
        self.q = np.exp(logq)
        self.n = nStates

    def logpi(self, profile):
        k = profile.count_switches()
        if k <= self.K:
            N = len(profile)
            lognorm = np.log( np.sum([self.n*(self.q*(self.n-1))**kk * scipy.special.binom(N-1, kk) for kk in range(self.K+1)]) )
            return k*self.logq - lognorm
        else:
            return -np.inf

class NumIntPrior(Prior):
    """
    Prior designed to reproduce the effects of Chris' numInt scheme
    """
    def __init__(self, numInt, logq=0, nStates=2):
        self.K = numInt - 1
        self.logq = logq
        self.n = nStates

        self._lognorm = self.K * np.log(1 + np.exp(self.logq)*(self.n - 1)) + np.log(self.n)

    def logpi(self, profile):
        k = profile.count_switches()
        if k <= self.K:
            i = np.arange(k)
            N = len(profile)
            return k * self.logq + np.log(np.prod((self.K-i)/(N-1-i))) - self._lognorm
        else:
            return -np.inf
