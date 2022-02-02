"""
The inference models, and the interface they have to conform to.
"""

import abc
import functools

import numpy as np
import scipy.optimize

from tracklib import Trajectory
from tracklib.models import rouse
from .util import Loopingtrace

class Model(metaclass=abc.ABCMeta):
    """
    Abstract base class for inference models

    The most important capability of any model is the likelihood function
    `logL` for a combination of `Loopingtrace` and `Trajectory`. Furthermore, a
    model should provide an initial guess for a good `Loopingtrace`.

    The method `trajectory_from_loopingtrace` is considered an optional part of
    the interface, since it is not important to the inference, but might come
    in handy when working with a `Model`. So it is recommended but not
    required.
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

    def trajectory_from_loopingtrace(self, loopingtrace, localization_error=0.1, d=3):
        """
        Generate a `Trajectory` from this `Model` and the given `Loopingtrace`

        Parameters
        ----------
        loopingtrace : Loopingtrace
        localization_error : float, optional
            how much Gaussian noise to add to the trajectory.
        d : int, optional
            number of spatial dimensions

        Returns
        -------
        Trajectory
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
    localization_error : float or np.array, optional
        a global value for the localization error. By default, we use the value
        stored in ``traj.meta['localization_error']``, which allows
        trajectory-wise specification of error. But for example for fitting it
        might be useful to have one global setting for localization error, at
        which point it becomes part of the model.

    Attributes
    ----------
    models : list of `rouse.Model`
        the models used for inference
    localization_error : float, array, or None
        if ``None``, use ``traj.meta['localization_error']`` for each
        trajectory ``traj``. If float, assume that value for each dimension of
        the trajectory. If array, should contain values for each dimension
        separately, i.e. ``np.array([Δx, Δy, Δz])``.

    Notes
    -----
    By default, this model assumes that the difference between the models is
    the position of the extra bond. It is easy to generalize this, by editing
    the `models` attribute after initialization. The only thing to pay
    attention to is that each model needs to have a `!measurement` vector.

    The `initial_loopingtrace` for this `Model` is the MLE assuming time scale
    separation. I.e. we calculate the timepoint-wise MLE using the exact steady
    state distributions of each model.

    See also
    --------
    Model, rouse.Model
    """
    def __init__(self, N, D, k,
                 k_extra=None,
                 looppositions=((0, 0), (0, -1)), # no mutable default elements! (i.e. tuple instead of list)
                 measurement="end2end",
                 localization_error=None,
                 ):

        self.localization_error = localization_error

        if k_extra is None:
            k_extra = k
        if str(measurement) == "end2end":
            measurement = np.zeros((N,))
            measurement[0]  = -1
            measurement[-1] =  1

        self.models = []
        for loop in looppositions:
            mod = rouse.Model(N, D, k, k_extra, extrabond=loop)
            mod.measurement = measurement
            self.models.append(mod)

    def initial_loopingtrace(self, traj):
        # We give the MLE assuming time scale separation
        # This is exactly the same procedure as for FactorizedModel, where we
        # utilize the steady state distributions of the individual Rouse
        # models.
        loopingtrace = Loopingtrace.forTrajectory(traj, len(self.models))
        distances = traj.abs()[loopingtrace.t][:, 0]

        ss_variances = np.array([mod.measurement @ mod.steady_state(True)[1] @ mod.measurement \
                                 for mod in self.models])
        ss_likelihoods = -0.5*(distances[:, np.newaxis]**2 / ss_variances[np.newaxis, :] \
                                + traj.d*np.log(2*np.pi*ss_variances)[np.newaxis, :])

        loopingtrace.state = np.argmax(ss_likelihoods, axis=1)
        return loopingtrace

    def logL(self, loopingtrace, traj):
        if traj.N == 2: # pragma: no cover
            traj = traj.relative()

        if self.localization_error is not None:
            if np.isscalar(self.localization_error):
                localization_error = d*[self.localization_error]
            else:
                localization_error = self.localization_error
        else:
            localization_error = traj.meta['localization_error']
        localization_error = np.asarray(localization_error)
        assert localization_error.shape == (traj.d,)


        # if not hasattr(loopingtrace, 'individual_logLs'): # interferes with model fitting
        looptrace = loopingtrace.full_valid()
        logLs = [rouse.multistate_likelihood(traj[:][:, i],
                                             self.models,
                                             looptrace,
                                             localization_error[i],
                                             return_individual_likelihoods=True,
                                            )[1] \
                 for i in range(traj.d)]
        loopingtrace.individual_logLs = np.sum(logLs, axis=0)[loopingtrace.t]

        return np.nansum(loopingtrace.individual_logLs)


    def trajectory_from_loopingtrace(self, loopingtrace, localization_error=None, d=3):
        if localization_error is None:
            if self.localization_error is None:
                raise ValueError("Need to specify either localization_error or model.localization_error")
            else:
                localization_error = self.localization_error
        if np.isscalar(localization_error):
            localization_error = d*[localization_error]
        localization_error = np.asarray(localization_error)
        if localization_error.shape != (d,):
            raise ValueError("Did not understand localization_error")

        arr = np.empty((loopingtrace.T, d))
        arr[:] = np.nan

        cur_mod = self.models[loopingtrace[0]]
        conf = cur_mod.conf_ss(True, d)
        arr[loopingtrace.t[0], :] = cur_mod.measurement @ conf

        for i in range(1, len(loopingtrace)):
            cur_mod = self.models[loopingtrace[i]]
            conf = cur_mod.evolve(conf, True, loopingtrace.t[i]-loopingtrace.t[i-1])
            arr[loopingtrace.t[i], :] = cur_mod.measurement @ conf

        return Trajectory.fromArray(arr + localization_error[np.newaxis, :]*np.random.normal(size=arr.shape),
                                    localization_error=localization_error,
                                    loopingtrace=loopingtrace,
                                    )

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
        distance values and return a corresponding number of outputs. If you
        plan on using `trajectory_from_loopingtrace`, the distributions should
        also have an ``rvs()`` method for sampling.

    Attributes
    ----------
    distributions : list of distribution objects

    Notes
    -----
    This being a heuristical model, we assume that the localization error is
    already incorporated in the `!distributions`, as would be the case if they
    come from experimental data. Therefore, this class ignores the
    ``meta['localization_error']`` field of `Trajectory`.

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
        loopingtrace = Loopingtrace.forTrajectory(traj, len(self.distributions))
        loopingtrace.state = np.argmax(self._known_trajs[traj]['logL_table'][:, loopingtrace.t], axis=0)
        return loopingtrace

    def logL(self, loopingtrace, traj):
        self._memo(traj)
        return np.sum(self._known_trajs[traj]['logL_table'][loopingtrace.state, loopingtrace.t])

    def trajectory_from_loopingtrace(self, loopingtrace, localization_error=0., d=3):
        # Note that the distributions in the model give us only the length, not
        #   the orientation. So we also have to sample unit vectors
        # Furthermore, localization_error should not be added, since
        #   self.distributions already contain it
        arr = np.empty((loopingtrace.T, d))
        arr[:] = np.nan
        magnitudes = np.array([self.distributions[state].rvs() for state in loopingtrace.state])
        vectors = np.random.normal(size=(len(magnitudes), d))
        vectors *= np.expand_dims(magnitudes / np.linalg.norm(vectors, axis=1), 1)
        arr[loopingtrace.t, :] = vectors

        return Trajectory.fromArray(arr,
                                    localization_error=np.array(d*[localization_error]),
                                    loopingtrace=loopingtrace,
                                    )

def _neg_logL_traj(traj, model):
    # For internal use in parallelization
    return -model.logL(traj.meta['loopingtrace'], traj)

def fit(data, modelfamily,
        show_progress=False, assume_notebook_for_progressbar=True,
        map_function=map,
        **kwargs):
    """
    Find the best fit model to a calibration dataset

    Parameters
    ----------
    data : TaggedSet of Trajectory
        the calibration data. Each `Trajectory` should have a `meta
        <Trajectory.meta>` entry ``'loopingtrace'`` indicating the true/assumed
        `Loopingtrace` for this trajectory.
    modelfamily : ParametricFamily of Models
        the family of models to consider.
    show_progress : bool, optional
        set to ``True`` to get progress info. Note that since we do not know
        how many iterations we need for convergence, there is no ETA, just
        elapsed time.
    map_function : map-like callable
        a function to replace the built-in ``map()``, e.g. with a parallel
        version. Will be used as
        ``np.sum(list(map_function(likelihood_given_parameters, data)))``, i.e.
        order does not matter. ``multiprocessing.Pool.imap_unordered`` would be
        a good go-to.
    kwargs : kwargs
        will be forwarded to `!scipy.optimize.minimize`. We use the defaults
        ``method='L-BFGS-B'``, ``maxfun=300``, ``ftol=1e-5``.

    Returns
    -------
    res : fitresult
        the structure returned by `!scipy.optimize.minimize`. The best fit
        parameters are ``res.x``, while their covariance matrix can be obtained
        as ``res.hess_inv.todens()``.

    Examples
    --------
    A good measure for relative uncertainty of the estimate is given by
    ``(√det(Σ) / Π(x))^(1/n)``, i.e. the major axes of the covariance ellipsoid
    over the point estimates, normalized by the dimensionality:

    >>> res = neda.models.fit(data, modelfam)
    ... relative_uncertainty = ( np.sqrt(np.linalg.det(res.hess_inv.todense())) \
    ...                        / np.prod(res.x) )**(1/modelfam.nParams)

    The function being minimized here is the negative log-likelihood of the
    data set, given parameters to the `modelfamily`. Specifically, this
    function is

    >>> def minimization_target(params):
    ...     mappable = functools.partial(_neg_logL_traj, model=modelfamily(*params))
    ...     return np.nansum(list(map_function(mappable, data)))

    See also
    --------
    ParametricFamily

    **Troubleshooting**
     - make sure that the magnitude of parameter values is around one. The
       minimizer (see `!scipy.optimize.minimize`) defaults to a fixed step
       gradient descent, which is useless if parameters are orders of magnitude
       bigger than 1. You can also try to play with the minimizer's options to
       make it use an adaptive step size.
     - make sure units match up. A common mistake is to have a unit mismatch
       between localization error and trajectories (e.g. one in μm and the
       other in nm). If the localization error is too big (here by a factor of
       1000), the fit for `!D` will converge to zero (i.e. ``1e-10``).
     - the ``'hess_inv'`` field returned with ``method='L-BFGS-B'`` might not
       be super reliable, even if the point estimate is pretty good. Check
       initial conditions when using this.
    """
    # Set up progressbar
    if show_progress:
        if assume_notebook_for_progressbar:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        pbar = tqdm()
    else:
        class Nullbar:
            def update(*args): pass
            def close(*args): pass
        pbar = Nullbar()

    # Set up minimization target
    def neg_logL_ds(params):
        mappable = functools.partial(_neg_logL_traj, model=modelfamily(*params))
        out = np.nansum(list(map_function(mappable, data)))
        pbar.update()
        return out

    minimize_kwargs = {
            'method' : 'L-BFGS-B',
            'bounds' : modelfamily.bounds,
            'options' : {'maxfun' : 300, 'ftol' : 1e-5},
            }
    # 'People' might try to override the defaults individually
    if not 'options' in kwargs:
        for key in minimize_kwargs['options']:
            if key in kwargs:
                minimize_kwargs['options'][key] = kwargs[key]
                del kwargs[key]
    minimize_kwargs.update(kwargs)

    res = scipy.optimize.minimize(neg_logL_ds, modelfamily.start_params, **minimize_kwargs)
    pbar.close()
    return res
