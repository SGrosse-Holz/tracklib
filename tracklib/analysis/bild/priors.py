"""
Everything to do with priors
"""

import abc

import numpy as np
import scipy.stats

from .util import Loopingtrace, ParametricFamily

class Prior(metaclass=abc.ABCMeta):
    """
    Abstract base class for priors

    When subclassing, you need to provide the `logpi` method, which should
    return the (log) prior probability for an input `Loopingtrace`. Specifying
    `logpi_vectorized` is optional, if you can speed up the prior calculation
    over an iterable of loopingtraces.
    """
    @abc.abstractmethod
    def logpi(self, loopingtrace):
        """
        log of prior probability

        Parameters
        ----------
        loopingtrace : Loopingtrace

        Returns
        -------
        float

        See also
        --------
        logpi_vectorized
        """
        raise NotImplementedError # pragma: no cover

    def logpi_vectorized(self, loopingtraces):
        """
        Evaluate the prior on multiple loopingtraces

        By default just sequentially evaluates `logpi` on the given
        loopingtraces, which of course does not give a speedup.

        Parameters
        ----------
        loopingtraces : Sequence (e.g. list) of Loopingtrace

        Returns
        -------
        np.ndarray, dtype=float

        See also
        --------
        logpi
        """
        return np.array([self.logpi(trace) for trace in loopingtraces])

class UniformPrior(Prior):
    r"""
    A uniform prior over loopingtraces

    This is simply :math:`-N\log(n)` for each `Loopingtrace`, where :math:`n`
    is the number of states and :math:`N` the number of (valid) frames.
    """
    def logpi(self, loopingtrace):
        return -len(loopingtrace)*np.log(loopingtrace.n)

class GeometricPrior(Prior):
    r"""
    A geometric prior over #switches in the `Loopingtrace`

    Writing :math:`\theta` for the `Loopingtrace` and :math:`k(\theta)` for the
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

    def logpi(self, loopingtrace):
        # optimized
        k = np.count_nonzero(loopingtrace.state[1:] != loopingtrace.state[:-1])
        return k*self.logq - (len(loopingtrace)-1)*self._log_norm_per_dof  - self._log_n

    def logpi_vectorized(self, loopingtraces):
        loopingtraces = np.array([trace.state for trace in loopingtraces])
        ks = np.count_nonzero(loopingtraces[:, 1:] != loopingtraces[:, :-1], axis=1)
        return ks*self.logq - (loopingtraces.shape[1]-1)*self._log_norm_per_dof - self._log_n

    @classmethod
    def family(cls, nStates=2):
        """
        Give a `ParametricFamily` for `GeometricPrior`

        Parameters
        ----------
        nStates : int, optional
            the number of states the priors will assume

        Returns
        -------
        ParametricFamily

        See also
        --------
        ParametricFamily
        """
        fam = ParametricFamily((0,), [(None, 0)])
        fam.get = lambda logq : cls(logq, nStates)
        return fam

class UniformKPrior_NONORM(Prior):
    """
    A prior that has uniform distribution of #switches
    """
    def logpi(self, loopingtrace):
        p = 1 - 1/loopingtrace.n
        k = np.count_nonzero(loopingtrace.state[1:] != loopingtrace.state[:-1])
        return -scipy.stats.binom(n=len(loopingtrace)-1, p=p).logpmf(k)

class FixedSwitchPrior(Prior):
    """
    Prior for a fixed number of switches
    """
    def __init__(self, k):
        self.k = k

    def logpi(self, loopingtrace):
        k = np.count_nonzero(np.diff(loopingtrace.state))
        if k == self.k:
            N = len(loopingtrace)
            n = loopingtrace.n
            return -np.log( n*(n-1)**k * scipy.special.binom(N-1, k) )
        else:
            return -np.inf

class MaxSwitchPrior(Prior):
    """
    Prior for limiting number of switches
    """
    def __init__(self, kmax, logq=0, nStates=2):
        self.kmax = kmax
        self.logq = logq
        self.q = np.exp(logq)
        self.n = nStates

    def logpi(self, loopingtrace):
        k = np.count_nonzero(np.diff(loopingtrace.state))
        if k <= self.kmax:
            N = len(loopingtrace)
            lognorm = np.log( np.sum([self.n*(self.q*(self.n-1))**kk * scipy.special.binom(N-1, kk) for kk in range(self.kmax+1)]) )
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

        self._lognorm = + self.K * np.log(1 + np.exp(self.logq)*(self.n - 1)) + np.log(self.n)

    def logpi(self, loopingtrace):
        k = np.count_nonzero(np.diff(loopingtrace.state))
        if k <= self.K:
            i = np.arange(k)
            N = len(loopingtrace)
            return k * self.logq + np.log(np.prod((self.K-i)/(N-1-i))) - self._lognorm
        else:
            return -np.inf

# class NumIntPrior_NONORM(Prior):
#     """
#     Prior designed to reproduce the effects of Chris' numInt scheme
#     """
#     def __init__(self, numInt, logq):
#         self.numInt = numInt
#         self.logq = logq
# 
#     def logpi(self, loopingtrace):
#         k = np.count_nonzero(np.diff(loopingtrace.state))
#         if k < self.numInt:
#             p = 1 - 1/loopingtrace.n
#             return (self.numInt-1 - k)*np.log(len(loopingtrace)-1) + k*self.logq
#             # return -scipy.stats.binom(n=len(loopingtrace)-1, p=p).logpmf(k) + k*self.logq
#         else:
#             return -np.inf
