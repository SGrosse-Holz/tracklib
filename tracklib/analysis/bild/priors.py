"""
Everything to do with priors
"""

import abc

import numpy as np

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
        return -len(loopingtrace.t)*np.log(loopingtrace.n)

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
        return k*self.logq - (len(loopingtrace.t)-1)*self._log_norm_per_dof  - self._log_n

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
