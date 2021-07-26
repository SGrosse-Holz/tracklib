"""
The inference models, and the interface they have to conform to.
"""

import abc
import functools

import numpy as np
import scipy.optimize
import scipy.stats

from tracklib import Trajectory
from tracklib.models import rouse
from .util import Loopingtrace

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)

class MultiStateModel(metaclass=abc.ABCMeta):
    """
    Abstract base class for inference models

    The most important capability of any model is the likelihood function
    `logL` for a combination of `Loopingtrace` and `Trajectory`. Furthermore, a
    model should provide an initial guess for a good `Loopingtrace`.

    The method `trajectory_from_loopingtrace` is considered an optional part of
    the interface, since it is not important to the inference, but might come
    in handy when working with a `MultiStateModel`. So it is recommended but not
    required.
    """
    @property
    def nStates(self):
        """
        How many internal states does this model have?
        """
        raise NotImplementedError # pragma: no cover

    @property
    def d(self):
        """
        Spatial dimension
        """
        raise NotImplementedError # pragma: no cover

    def initial_loopingtrace(self, traj):
        """
        Give a quick guess for a good `Loopingtrace` for a `Trajectory`.

        The default implementation gives a random `Loopingtrace`.

        Parameters
        ----------
        traj : Trajectory

        Returns
        -------
        Loopingtrace
        """
        lt = Loopingtrace.forTrajectory(traj, nStates=self.nStates)
        lt.state = np.random.choice(self.nStates, size=lt.state.shape)
        return lt

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

    def trajectory_from_loopingtrace(self, loopingtrace, localization_error):
        """
        Generate a `Trajectory` from this `MultiStateModel` and the given `Loopingtrace`

        Parameters
        ----------
        loopingtrace : Loopingtrace
        localization_error : float, optional
            how much Gaussian noise to add to the trajectory.

        Returns
        -------
        Trajectory
        """
        raise NotImplementedError # pragma: no cover

class MultiStateRouse(MultiStateModel):
    """
    A multi-state Rouse model

    This inference model uses a given number of `rouse.Model` instances to
    choose from for each propagation interval. In the default use case this
    switches between a looped and unlooped model, but it could be way more
    general than that, e.g. incorporating different looped states, loop
    positions, numbers of loops, etc.

    Parameters
    ----------
    N : int
        number of monomers
    D, k : float
        Rouse parameters: 1d diffusion constant of free monomers and backbone
        spring constant
    d : int, optional
        spatial dimension
    looppositions : list of 2-tuples of int, optional
        list of positions of the extra bond. For each entry, a new
        `rouse.Model` instance will be set up. Remember to include an unlooped
        model (if wanted) by including a position like ``(0, 0)``. Each entry
        can alternatively be a 3-tuple, where the 3rd entry then specifies the
        strength of the extra bond relative to the backbone, e.g. ``(0, 5,
        0.5)`` introduces an additional bond between monomers 0 and 5 with
        strength ``0.5*k``. Finally, instead of a single tuple, each bond
        specification can be a list of such tuples if multiple added bonds are
        needed.
    measurement : "end2end" or (N,) np.ndarray
        which distance to measure. The default setting "end2end" is equivalent
        to specifying a vector ``np.array([-1, 0, ..., 0, 1])``, i.e. measuring
        the distance from the first to the last monomer.
    localization_error : float or np.array, optional
        a global value for the localization error. By default, we use the value
        stored in ``traj.meta['localization_error']``, which allows
        trajectory-wise specification of error. But for example for fitting it
        might be useful to have one global setting for localization error, at
        which point it becomes part of the model. Give a scalar value to have
        the same error apply to all dimensions

    Attributes
    ----------
    models : list of `rouse.Model`
        the models used for inference
    measurement : (N,) np.ndarray
        the measurement vector for this model
    localization_error : array or None
        if ``None``, use ``traj.meta['localization_error']`` for each
        trajectory ``traj``.

    Notes
    -----
    The `initial_loopingtrace` for this `MultiStateModel` is the MLE assuming time scale
    separation. I.e. we calculate the timepoint-wise MLE using the exact steady
    state distributions of each model.

    See also
    --------
    MultiStateModel, rouse.Model
    """
    def __init__(self, N, D, k, d=3,
                 looppositions=((0, 0), (0, -1)), # no mutable default parameters!
                                                  # (thus tuple instead of list)
                 measurement="end2end",
                 localization_error=None,
                 ):
        self._d = d

        if str(measurement) == "end2end":
            measurement = np.zeros(N)
            measurement[0]  = -1
            measurement[-1] =  1
        self.measurement = measurement

        if localization_error is not None and np.isscalar(localization_error):
            localization_error = np.array(d*[localization_error])
        self.localization_error = localization_error

        self.models = []
        for loop in looppositions:
            if np.isscalar(loop[0]):
                loop = [loop]
            mod = rouse.Model(N, D, k, d, add_bonds=loop)
            self.models.append(mod)

    @property
    def nStates(self):
        return len(self.models)

    @property
    def d(self):
        return self._d

    def _get_noise(self, traj):
        if self.localization_error is not None:
            return np.asarray(self.localization_error)
        else:
            return np.asarray(traj.meta['localization_error'])

    def initial_loopingtrace(self, traj):
        # We give the MLE assuming time scale separation
        # This is exactly the same procedure as for FactorizedModel, where we
        # utilize the steady state distributions of the individual Rouse
        # models.
        lt = Loopingtrace.forTrajectory(traj, self.nStates)
        noise = self._get_noise(traj)

        Ms = []
        Cs = []
        for mod in self.models:
            M, C = mod.steady_state()
            Ms.append(self.measurement @ M)
            Cs.append(self.measurement @ C @ self.measurement)
        Ms = np.expand_dims(Ms, 0)                # (1, n, d)
        Cs = np.expand_dims(Cs, (0, 2))           # (1, n, 1)
        Cs = Cs + np.expand_dims(noise*noise, (0, 1)) # (1, n, d)

        # assemble (T, n, d) array
        chi2s = (traj[lt.t][:, np.newaxis, :] - Ms)**2 / Cs

        logLs = -0.5*(chi2s + np.log(Cs)) - 0.5*np.log(2*np.pi)*np.ones(chi2s.shape)
        logLs = np.sum(logLs, axis=2) # (T, n)

        lt.state = np.argmax(logLs, axis=1)
        return lt

    def logL(self, loopingtrace, traj):
        localization_error = self._get_noise(traj)

        looptrace = loopingtrace.full_valid()
        T = len(looptrace)

        for model in self.models:
            model.check_dynamics()

        M, C_single = self.models[looptrace[0]].steady_state()
        C = self.d * [C_single]

        logL = np.empty((T,), dtype=float)
        logL[:] = np.nan
        ftraj = np.empty((2, T, self.d), dtype=float)
        ftraj[:] = np.nan
        for t, model_id in enumerate(looptrace):
            # Pick the model for propagation to the next data point
            model = self.models[model_id]

            # Propagate
            M = model.propagate_M(M, check_dynamics=False)
            C = [model.propagate_C(myC, check_dynamics=False) for myC in C]

            # Update
            if not np.any(np.isnan(traj[t])):
#                 M, C, logL[t] = model.update_ensemble_with_observation_nd(
#                                         M, C, traj[t], localization_error, self.measurement)
                Ms = []
                Cs = []
                logLs = []
                for d in range(self.d):
                    l, m, c = model.Kalman_update_1d(
                                    M[:, d], C[d],
                                    traj[t][d], localization_error[d],
                                    self.measurement)
                    logLs.append(l)
                    Ms.append(m)
                    Cs.append(c)

                M = np.stack(Ms, axis=1)
                C = Cs
                logL[t] = np.sum(logLs)
                ftraj[0, t, :] = self.measurement @ M
                ftraj[1, t, :] = [self.measurement @ c @ self.measurement for c in C]

        loopingtrace.individual_logLs = logL[loopingtrace.t]
        loopingtrace.filtered_trajectory = Trajectory.fromArray(ftraj[0], variances=ftraj[1])
        return np.nansum(loopingtrace.individual_logLs)

    def trajectory_from_loopingtrace(self, loopingtrace, localization_error=None):
        if localization_error is None:
            if self.localization_error is None:
                raise ValueError("Need to specify either localization_error or model.localization_error") # pragma: no cover
            else:
                localization_error = self.localization_error
        if np.isscalar(localization_error):
            localization_error = self.d*[localization_error]
        localization_error = np.asarray(localization_error)
        if localization_error.shape != (self.d,):
            raise ValueError("Did not understand localization_error") # pragma: no cover

        looptrace = loopingtrace.full_valid()

        arr = np.empty((len(looptrace), self.d), dtype=float)
        arr[:] = np.nan

        model = self.models[looptrace[0]]
        conf = model.conf_ss()
        arr[0, :] = self.measurement @ conf

        for i in range(1, len(looptrace)):
            model = self.models[looptrace[i]]
            conf = model.evolve(conf)
            arr[i, :] = self.measurement @ conf

        data = np.empty(arr.shape, dtype=float)
        data[:] = np.nan
        data[loopingtrace.t, :] = arr[loopingtrace.t, :]

        noise = localization_error[np.newaxis, :]
        return Trajectory.fromArray(data + noise*np.random.normal(size=data.shape),
                                    localization_error=localization_error,
                                    loopingtrace=loopingtrace,
                                    )

    def toFactorized(self):
        """
        Give the corresponding `FactorizedModel`

        This is the model that simply calculates likelihoods from the steady
        state probabilities of each of the individual states.

        Returns
        -------
        FactorizedModel
        """
        distributions = []
        for mod in self.models:
            _, C = mod.steady_state()
            s2 = self.measurement @ C @ self.measurement + np.sum(self.localization_error**2)/self.d
            distributions.append(scipy.stats.maxwell(scale=np.sqrt(s2)))

        return FactorizedModel(distributions, d=self.d)

class FactorizedModel(MultiStateModel):
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

    Examples
    --------
    Experimentally measured distributions can be used straightforwardly using
    ``scipy.stats.gaussian_kde``: assuming we have measured ensembles of
    distances ``dists_i`` for reference states ``i``, we can use

    >>> model = FactorizedModel([scipy.stats.gaussian_kde(dists_0),
    ...                          scipy.stats.gaussian_kde(dists_1),
    ...                          scipy.stats.gaussian_kde(dists_1)])

    """
    def __init__(self, distributions, d=3):
        self.distributions = distributions
        self._d = d
        self._known_trajs = dict()

    @property
    def nStates(self):
        return len(self.distributions)

    @property
    def d(self):
        return self._d

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
        loopingtrace = Loopingtrace.forTrajectory(traj, self.nStates)
        loopingtrace.state = np.argmax(self._known_trajs[traj]['logL_table'][:, loopingtrace.t], axis=0)
        return loopingtrace

    def logL(self, loopingtrace, traj):
        self._memo(traj)
        return np.sum(self._known_trajs[traj]['logL_table'][loopingtrace.state, loopingtrace.t])

    def trajectory_from_loopingtrace(self, loopingtrace, localization_error=0.):
        # Note that the distributions in the model give us only the length, not
        #   the orientation. So we also have to sample unit vectors
        # Furthermore, localization_error should not be added, since
        #   self.distributions already contain it. It will be written to the
        #   meta entry though!
        arr = np.empty((loopingtrace.T, self.d))
        arr[:] = np.nan
        magnitudes = np.array([self.distributions[state].rvs() for state in loopingtrace.state])
        vectors = np.random.normal(size=(len(magnitudes), self.d))
        vectors *= np.expand_dims(magnitudes / np.linalg.norm(vectors, axis=1), 1)
        arr[loopingtrace.t, :] = vectors

        return Trajectory.fromArray(arr,
                                    localization_error=np.array(self.d*[localization_error]),
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
    if show_progress: # pragma: no cover
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
