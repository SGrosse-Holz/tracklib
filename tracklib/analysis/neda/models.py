"""
The inference models, and the interface they have to conform to.
"""

import abc

import numpy as np

from tracklib.models import rouse
from .util import Loopingtrace

class Model(metaclass=abc.ABCMeta):
    """
    Abstract base class for inference models

    The most important capability of any model is the likelihood function
    `logL` for a combination of `Loopingtrace` and `Trajectory`. Furthermore, a
    model should provide an initial guess for a good `Loopingtrace`.
    """
    @abc.abstractmethod
    def initial_loopingtrace(self, traj):
        """
        Give a quick guess for a good `Loopingtrace` for a `Trajectory`.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Loopingtrace
        """
        raise NotImplementedError # pragma: no cover

    @abc.abstractmethod
    def logL(self, loopingtrace, traj):
        """
        Calculate (log-)likelihood for (`Loopingtrace`, `Trajectory`) pair.

        Parameters
        ----------
        loopingtrace : Loopingtrace
        traj : Trajectory

        Returns
        -------
        float
            log-likelihood associated with the inputs
        """
        raise NotImplementedError # pragma: no cover

class RouseModel(Model):
    """
    Inference with Rouse models

    This inference model uses a given number of `rouse.Model` instances to
    choose from for each propagation interval. In the default use case this
    switches between a looped and unlooped model, but it could be way more
    general than that, e.g. incorporating different looped states, loop
    positions, numbers of loop, etc.

    Parameters
    ----------
    N, D, k : float
        parameters for the `rouse.Model`. `!N` is the number of monomers, `!D`
        the diffusion constant of a free monomer, `k` the backbone strength.
    looppositions : list of 2-tuples of int, optional
        list of positions of the extra bond. For each entry, a new
        `rouse.Model` instance will be set up. Remember to include an unlooped
        model (if wanted) by including a position like ``(0, 0)``.
    k_extra : float, optional
        the strength of the extra bond. By default equal to `k`
    measurement : "end2end" or (N,) np.ndarray
        which distance to measure. The default setting "end2end" is equivalent
        to specifying a vector ``np.array([-1, 0, ..., 0, 1])``, i.e. measuring
        the distance from the first to the last monomer.

    Attributes
    ----------
    models : list of `rouse.Model`
        the models used for inference

    Notes
    -----
    By default, this model assumes that the difference between the models is
    the position of the extra bond. It is easy to generalize this, by editing
    the `models` attribute after initialization. The only thing to pay
    attention to is that each model needs to have a `!measurement` vector.

    See also
    --------
    Model, rouse.Model
    """
    def __init__(self, N, D, k, looppositions=[(0, 0), (0, -1)], k_extra=None, measurement="end2end"):
        if k_extra is None:
            k_extra = k
        if measurement == "end2end":
            measurement = np.zeros((N,))
            measurement[0]  = -1
            measurement[-1] =  1

        self.models = []
        for loop in looppositions:
            mod = rouse.Model(N, D, k, k_extra, extrabond=loop)
            mod.measurement = measurement
            self.models.append(mod)

    def initial_loopingtrace(self, traj):
        # TODO: come up with a good scheme here
        return Loopingtrace(traj, len(self.models))

    def logL(self, loopingtrace, traj):
        if traj.N == 2: # pragma: no cover
            traj = traj.relative()

        looptrace = loopingtrace.full_valid()
        return np.sum([ \
                rouse.multistate_likelihood(traj[:][:, i],
                                            self.models,
                                            looptrace,
                                            traj.meta['localization_error'][i],
                                           ) \
                for i in range(traj.d)])

class FactorizedModel(Model):
    """
    A simplified model, assuming time scale separation

    This model assumes that each point is sampled from one of a given list of
    distributions, where there is no correlation between the choice of
    distribution for each point. It runs significantly faster than the full
    `RouseModel`, but is of course inaccurate if the Rouse time is longer or
    comparable to the frame rate of the recorded trajectories.

    Parameters
    ----------
    distributions : list of distribution objects
        these will usually be ``scipy.stats.rv_continuous`` objects (e.g.
        Maxwell), but can be pretty arbitrary. The only function they have to
        provide is ``logpdf()``, which should take a scalar or vector of
        distance values and return a corresponding number of outputs.

    Attributes
    ----------
    distributions : list of distribution objects

    Notes
    -----
    Instances of this class memoize trajectories they have seen before. To
    reset the memoization, you can either reinstantiate or clear the cache
    manually:
    
    >>> model = FactorizedModel(model.distributions)
    ... model.clear_memo()

    If using ``scipy.stats.maxwell``, make sure to use it correctly, i.e. you
    have to specify ``scale=...``. Writing ``scipy.stats.maxwell(5)`` instead
    of ``scipy.stats.maxwell(scale=5)`` shifts the distribution instead of
    scaling it and leads to ``-inf`` values in the likelihood, which then screw
    up the MCMC. The classic error to get for this is ``invalid value
    encountered in double_scalars``. This is caused by ``new_logL - cur_logL``
    reading ``- inf + inf`` at the first MCMC iteration, if `logL` returns
    ``-inf``.
    """
    def __init__(self, distributions):
        self.distributions = distributions
        self._known_trajs = dict()

    def _memo(self, traj):
        """
        (internal) memoize `traj`
        """
        if not traj in self._known_trajs:
            with np.errstate(divide='ignore'): # nans in the trajectory raise 'divide by zero in log'
                logL_table = np.array([dist.logpdf(traj.abs()[:][:, 0]) 
                                       for dist in self.distributions
                                       ])
            self._known_trajs[traj] = {'logL_table' : logL_table}

    def clear_memo(self):
        """
        Clear the memoization cache
        """
        self._known_trajs = dict()

    def initial_loopingtrace(self, traj):
        self._memo(traj)
        loopingtrace = Loopingtrace(traj, len(self.distributions))
        loopingtrace.state = np.argmax(self._known_trajs[traj]['logL_table'][:, loopingtrace.t], axis=0)
        return loopingtrace

    def logL(self, loopingtrace, traj):
        self._memo(traj)
        return np.sum(self._known_trajs[traj]['logL_table'][loopingtrace.state, loopingtrace.t])
