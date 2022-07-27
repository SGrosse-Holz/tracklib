"""
Core implementation of msdfit

See also
--------
msdfit
"""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools

from tqdm.auto import tqdm

import numpy as np
from scipy import linalg, optimize, stats

from tracklib.models import rouse
from tracklib.util import parallel

__all__ = [
    "MSDfun",
    "ds_logL",
    "Fit",
    "Profiler",
]

# Verbosity rules: 0 = no output, 1 = warnings only, 2 = informational, 3 = debug info
verbosity = 1
def vprint(v, *args, **kwargs):
    if verbosity >= v: # pragma: no cover
        print("[msdfit]", (v-1)*'--', *args, **kwargs)
        
################## Covariance in terms of MSD #################################

def MSDfun(fun):
    """
    Decorator for MSD functions

    This is a decorator to use when implementing `params2msdm
    <Fit.params2msdm>` in `Fit`. It takes over some of the generic polishing.
    It assumes that the decorated function has the signature
    ``function(np.array) --> np.array`` and

    - ensures that the argument is cast to an array if necessary (such that you
      can then also call ``msd(5)`` instead of ``msd(np.array([5]))``
    - ensures that ``dt > 0`` by taking an absolute value and setting
      ``msd[dt==0] = 0`` without calling the wrapped function. You can thus
      ignore the ``dt == 0`` case in implementing an MSD function. Note that
      ``msd[dt==0] = 0`` should always be true. However, e.g. in the case of
      localization error, we might have ``lim_{Δt-->0} MSD(Δt) = 2σ² != 0``.

    Example
    -------
    >>> # ... Fit subclass implementation ...
    ... 
    ... def params2msdm(self, params):
    ...     D, noise2 = *params
    ... 
    ...     @MSDfun
    ...     def msd(dt, D=D, noise2=noise2):
    ...         # Note above: we hand over the parameters as default arguments,
    ...         # instead of using them as global variables. This is more robust in
    ...         # a few pathological cases, and generally cleaner.
    ...         return 2*D*dt + 2*noise2
    ... 
    ...     return self.d*[(msd, 0)]
    ... 
    ... # ... continue Fit subclass implementation ...

    See also
    --------
    msd2C_fun, Fit
    """
    def wrap(dt):
        # Preproc
        dt = np.abs(np.asarray(dt))
        was_scalar = len(dt.shape) == 0
        if was_scalar:
            dt = np.array([dt])
        
        # Calculate non-zero dts and set zeros
        msd = np.empty(dt.shape)
        ind0 = dt == 0
        msd[~ind0] = fun(dt[~ind0])
        msd[ind0] = 0
        
        # Postproc
        if was_scalar:
            msd = msd[0]
        return msd
    return wrap

def msd2C_ss0(msd, ti):
    """
    0th order steady state covariance from MSD. For internal use.

    Parameters
    ----------
    msd : np.ndarray
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory

    Returns
    -------
    np.ndarray
        the covariance matrix

    See also
    --------
    msd2C_fun
    """
    return 0.5*( msd[-1] - msd[np.abs(ti[:, None] - ti[None, :])] )

def msd2C_ss1(msd, ti):
    """
    1st order steady state covariance from MSD. For internal use.

    Parameters
    ----------
    msd : np.ndarray
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory

    Returns
    -------
    np.ndarray
        the increment covariance matrix

    See also
    --------
    msd2C_fun
    """
    return 0.5*(  msd[np.abs(ti[1:, None] - ti[None,  :-1])] + msd[np.abs(ti[:-1, None] - ti[None, 1:  ])]
                - msd[np.abs(ti[1:, None] - ti[None, 1:  ])] - msd[np.abs(ti[:-1, None] - ti[None,  :-1])] )

def msd2C_fun(msd, ti, ss_order):
    """
    msd2C for MSDs expressed as functions / non-integer times.

    Parameters
    ----------
    msd : callable, use the `MSDfun` decorator
        note that for ``ss_order == 0`` we expect ``msd(np.inf)`` to be
        well-defined.
    ti : np.ndarray, dtype=int
        times at which there are data in the trajectory
    ss_order : {0, 1}
        steady state order. See module documentation.

    Returns
    -------
    np.ndarray
        covariance matrix

    See also
    --------
    MSDfun, msdfit
    """
    if ss_order == 0:
        return 0.5*( msd(np.inf) - msd(ti[:, None] - ti[None, :]) )
    elif ss_order == 1:
        return 0.5*(  msd(ti[1:, None] - ti[None,  :-1]) + msd(ti[:-1, None] - ti[None, 1:  ])
                    - msd(ti[1:, None] - ti[None, 1:  ]) - msd(ti[:-1, None] - ti[None,  :-1]) )
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")

################## Gaussian Process likelihood ###############################

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)
GP_verbosity = 1
def GP_vprint(v, *args, **kwargs):
    if GP_verbosity >= v: # pragma: no cover
        print("[msdfit.GP]", (v-1)*'--', *args, **kwargs)

class BadCovarianceError(RuntimeError):
    pass

def _GP_core_logL(C, x):
    # Implementation notes
    # - (slogdet, solve) is faster than eigendecomposition (~3x)
    with np.errstate(under='ignore'):
        s, logdet = np.linalg.slogdet(C)
    if s <= 0: # pragma: no cover
        # Note that det > 0 does not imply positive definite
        raise BadCovarianceError("Covariance not positive definite. slogdet = ({}, {})".format(s, logdet))
        
    try:
        xCx = x @ linalg.solve(C, x, assume_a='pos')
    except (FloatingPointError, linalg.LinAlgError) as err: # pragma: no cover
        # what's the problematic case that made me insert this?
        # --> can (probably?) happen in numerical edge cases. Should usually be
        #     prevented by `Fit` in the first place
        GP_vprint(3, f"Problem when inverting covariance, even though slogdet = ({s}, {logdet})")
        GP_vprint(3, type(err), err)
        raise BadCovarianceError("Inverting covariance did not work")

    return -0.5*(xCx + logdet) - len(C)*LOG_SQRT_2_PI

def GP_logL(trace, ss_order, msd, mean=0):
    """
    Gaussian process likelihood for a given trace

    Parameters
    ----------
    trace : (T,) np.ndarray
        the data. Should be recorded at constant time lag; missing data are
        indicated with ``np.nan``.
    ss_order : {0, 1}
        steady state order; see module documentation.
    msd : np.ndarray
        the MSD defining the Gaussian process, evaluated up to (at least) ``T =
        len(trace)``.
    mean : float
        the first moment of the Gaussian process. For ``ss_order == 0`` this is
        the mean (i.e. same units as the trajectory), for ``ss_order == 1``
        this is the mean of the increment process, i.e. has units trajectory
        per time.

    Returns
    -------
    float
        the log-likelihood

    See also
    --------
    ds_logL
    """
    ti = np.nonzero(~np.isnan(trace))[0]
    
    if ss_order == 0:
        X = trace[ti] - mean
        C = msd2C_ss0(msd, ti)
    elif ss_order == 1:
        X = np.diff(trace[ti]) - mean*np.diff(ti)
        C = msd2C_ss1(msd, ti)
    else: # pragma: no cover
        raise ValueError(f"Invalid steady state order: {ss_order}")
    
    return _GP_core_logL(C, X)

def _GP_logL_for_parallelization(params):
    # just unpacking arguments
    return GP_logL(*params)

def ds_logL(data, ss_order, msd_ms):
    """
    Gaussian process likelihood on a data set.

    Parameters
    ----------
    data : a `TaggedSet` of `Trajectory`
    ss_order : {0, 1}
        steady state order; see module documentation.
    msd_ms : list of tuples (msd, mean)
        this should be a list with one entry for each spatial dimension of the
        data set. The first entry of each tuple is a function, ideally
        decorated with `MSDfun`. The second should be a float giving the
        mean/drift for the process (often zero).

    Returns
    ------
    float
        the total log-likelihood of the data under the Gaussian process
        specified by the given MSD.

    Notes
    -----
    Parallel-aware (unordered). However, in practice I find that it is usually
    faster to parallelize runs over multiple data sets / with different
    parameters, if possible. In a benchmarking run for the internal
    parallelization in this function I saw no observable benefit of running on
    more than 5 cores (presumably due to overhead in moving data to the
    workers). 

    See also
    --------
    Fit, tracklib.util.parallel
    """
    # msd_ms : list of tuples: d*[(msd, m)]
    # Implementation note: it does *not* make sense to allow a single tuple for
    # msd_ms, because broadcasting to spatial dimensions would require a
    # prefactor of 1/d in front of the component MSDs. It is easier (more
    # readable) to do this in the MSD implementation than keeping track of it
    # here
    d = data.map_unique(lambda traj : traj.d)
    if len(msd_ms) != d: # pragma: no cover
        raise ValueError(f"Dimensionality of MSD ({len(msd_ms)}) != dimensionality of data ({d})")
        
    # Convert msd to array, such that parallelization works
    Tmax = max(map(len, data))
    dt = np.arange(Tmax)
    array_msd_ms = [[msd(dt), m] for msd, m in msd_ms]
    if ss_order == 0:
        for dim, (msd, m) in enumerate(msd_ms):
            array_msd_ms[dim][0] = np.append(array_msd_ms[dim][0], msd(np.inf))

    job_iter = itertools.chain.from_iterable((itertools.product((traj[:][:, dim] for traj in data),
                                                                [ss_order], [msd], [m],
                                                               )
                                              for dim, (msd, m) in enumerate(array_msd_ms)
                                             ))
    
    return np.sum(list(parallel._umap(_GP_logL_for_parallelization, job_iter)))

################## Fit base class definition ###################################

class Fit(metaclass=ABCMeta):
    """
    Abstract base class for MSD fits. Backbone of this module.

    Subclass this to implement fitting of a specific functional form of the
    MSD. See also the existing library of fits in `msdfit.lib`.

    Parameters
    ----------
    data : `TaggedSet` of `Trajectory`
        the data to use. Note that the current selection in the data is saved
        internally

    Attributes
    ----------
    data : `TaggedSet` of `Trajectory`
        the data to use. Note that the current selection in the data is saved
        internally
    d : int
        spatial dimension of trajectories in the data
    T : int
        maximum length of the trajectories, in frames
    ss_order : {0, 1}
        steady state order. Often this will be a fixed constant for a
        particular `Fit` subclass.
    bounds : list of (lb, ub)
        bound for each of the parameters in the fit. This is everything the
        class "knows" about your choice of parameters internally. The length of
        this list is also used internally to count number of parameters, so it
        is important that every parameter has bounds. Use ``np.inf`` and
        ``-np.inf`` for unbounded parameters.
    fix_values : list of (i, fix_to)
        allows to fix some parameter values to constant or values of other
        parameters, e.g. to allow for different behavior in different
        dimensions, but fixing it to be equal by default. See Notes section.
    constraints : list of constraint functions
        allows to specify constraints on the parameters that will be
        implemented as smooth penalty on the likelihood. Can also take care of
        feasibility constraints: by default, there is a constraint checking
        that the covariance matrix given by the current MSD is positive
        definite (otherwise we could not even evaluate the likelihood). See
        Notes section.
    max_penalty : float
        constant cutoff for the penalty mechanism. Infeasible sets of
        parameters (where the likelihood function is not even well-defined)
        will be penalized with this value. For any set of parameters penalized
        with at least this value, likelihood evaluation is skipped and the
        value ``max_penalty`` assigned as value of the minimization target.
        Default value is ``1e10``, there should be little reason to change
        this.
    verbosity : {0, 1, 2, 3}
        controls output during fitting. 0: no output; 1: error messages only;
        2: informational; 3: debugging

    Notes
    -----
    The `!fix_values` mechanism allows to keep some parameters fixed, or
    express them as function of others. ``Fit.fix_values`` is a list of tuples
    ``(i, fix_to)``, where ``i`` is the index of the parameter you want to fix,
    c.f. `!bounds`. ``fix_to`` is either just a constant value, or a function
    ``fix_to(params) --> float``, where ``params`` are the current parameter
    values. Note that in this function you should not rely on any parameters
    that are themselves to be fixed. (It would get impossible to resolve all
    the dependencies).

    `!constraints` are functions with signature ``constraint(params) -->
    float``. The output is interpreted as follows:
    - x <= 0 : infeasible; maximum penalization
    - 0 <= x <= 1 : smooth penalization: ``penalty = exp(1/tan(pi*x))``
    - 1 <= x : feasible; no penalization
    Thus, if e.g. some some function ``fun`` of the parameters should be
    constrained to be positive, you would use ``fun(params)/eps`` as the
    constraint, with ``eps`` some small value setting the tolerance region. If
    there are multiple constraints, always the strongest one is used. For
    infeasible parameters, the likelihood function is not evaluated, but the
    "likelihood" is just set to ``-Fit.max_penalty``.

    Note that there is a default constraint checking positivity of the
    covariance matrix. If your functional form of the MSD is guaranteed to
    satisfy this (e.g. for a physical model), you can remove this constraint
    for performance.

    Upon subclassing, it is highly recommended to initialize the base class
    first thing:

    >>> def SomeFit(Fit):
    ...     def __init__(self, data, *other_args):
    ...         super().__init__(data) # <--- don't forget!

    This class uses ``scipy.optimize.minimize`` to find the MAP parameter
    estimate (or MLE if you leave `logprior` flat). When running the fit, you
    can choose between the simplex (Nelder-Mead) algorithm or gradient descent
    (L-BFGS-B). The latter uses the stopping criterion ``f^k -
    f^{k+1}/max{|f^k|,|f^{k+1}|,1} <= ftol``, which is inappropriate for
    log-likelihoods (which should be optimized to fixed accuracy of O(0.1)
    independent of the absolute value, which might be very large). We therefore
    use Nelder-Mead by default, which does not depend on derivatives and thus
    also has an absolute stopping criterion.

    If this function runs close to a maximum, e.g. in `Profiler` or when using
    successive optimization steps, we can fix the problem with gradient-based
    optimization by removing the large absolute value offset from the
    log-likelihood. This functionality is also exposed to the end user, who can
    overwrite the ``initial_offset`` method to give a non-zero offset together
    with the initial values provided via `initial_params`.
    """
    def __init__(self, data):
        self.data = data
        self.data_selection = data.saveSelection()
        self.d = data.map_unique(lambda traj: traj.d)
        self.T = max(map(len, self.data))

        # Fit properties
        self.ss_order = 0
        self.bounds = [] # List of (lb, ub), to be passed to optimize.minimize. len(bounds) is used to count parameters!
        self.fix_values = [] # will be appended to the list passed to run()

        # Each constraint should be a callable constr(params) -> x. We will apply:
        #   x <= 0                : infeasible. Maximum penalization
        #        0 <= x <= 1      : smooth crossover: np.exp(1/tan(pi*x))
        #                  1 <= x : feasible. No penalization   
        self.constraints = [self.constraint_Cpositive]
        self.max_penalty = 1e10
        
        self.verbosity = 1
        
    def vprint(self, v, *args, **kwargs):
        """
        Prints only if ``self.verbosity >= v``.
        """
        if self.verbosity >= v:
            print("[msdfit.Fit]", (v-1)*'--', *args, **kwargs)
        
    ### To be overwritten / used upon subclassing ###
    
    @abstractmethod
    def params2msdm(self, params):
        """
        Definition of MSD in terms of parameters

        This is the core of the fit definition. It should give a list of tuples
        as required by `ds_logL`

        Parameters
        ----------
        params : np.ndarray
            the current parameters

        Returns
        -------
        list of tuples (msd, m)
            for the definition of `!msd`, use of the `MSDfun` decorator is
            recommended.

        See also
        --------
        ds_logL, MSDfun
        """
        raise NotImplementedError # pragma: no cover

    def logprior(self, params):
        """
        Prior over the parameters

        Use this function if you want to specify a prior over the parameters
        that will be added to the log-likelihood.  Default is a flat prior,
        i.e. ``return 0``.

        Parameters
        ----------
        params : np.ndarray
            the current parameters

        Returns
        -------
        float
        """
        return 0
    
    @abstractmethod
    def initial_params(self):
        """
        Give initial values for the parameters

        You can use ``self.data`` to perform some ad hoc estimation (e.g. from
        the empirical MSD, using `analysis.MSD <tracklib.analysis.p2.MSD>`) or
        just return constants.

        Returns
        -------
        params : np.ndarray
            initial values for the parameters. Should satisfy ``len(params) ==
            len(self.bounds)``.
        """
        raise NotImplementedError # pragma: no cover

    def initial_offset(self):
        """
        Log-likelihood offset associated with initial parameters

        See Notes section of class documentation.

        Returns
        -------
        float

        See also
        --------
        Fit
        """
        return 0
                
    def constraint_Cpositive(self, params):
        """
        Constraint for positive definiteness of covariance matrix

        This can serve as an example of a non-trivial constraint. Note that you
        may not want to use it if positivity is already guaranteed by the
        functional form of your MSD. See also Notes section of class doc.

        Returns
        -------
        float

        Notes
        -----
        This function checks whether the spatial components are identical using
        python's ``is``. So if you are implementing an MSD with identical
        spatial components, you should return the final list as ``self.d*[(msd,
        mean)]``, such that this constraint checks positivity only once.

        See also
        --------
        Fit
        """
        min_ev_okay = 1 - np.cos(np.pi/self.T) # white noise min ev

        scores = []
        done = []
        for msd, _ in self.params2msdm(params):
            if msd not in done:
                min_ev = np.min(np.linalg.eigvalsh(msd2C_fun(msd, np.arange(self.T), self.ss_order)))
                scores.append(min_ev / min_ev_okay)
                done.append(msd)

        return min(scores)
    
    ### General machinery, usually won't need overwriting ###
        
    def _penalty(self, params):
        """
        Gives penalty for given parameters. Internal use.

        Parameters
        ----------
        params : np.ndarray

        Returns
        -------
        float
        """
        x = np.inf
        for constraint in self.constraints:
            x = min(constraint(params), x)
            if x <= 0:
                return -1 # unfeasible

        if x >= 1:
            return 0
        else:
            with np.errstate(over='raise', under='ignore'):
                try:
                    return min(np.exp(1/np.tan(np.pi*x)), self.max_penalty)
                except FloatingPointError:
                    return self.max_penalty

    def expand_fix_values(self, fix_values=None):
        """
        Preprocessing for fixed parameters. Mostly internal use.

        This function concatenates the internal ``self.fix_values`` and the
        given ``fix_values``, and makes sure that they all conform to the
        format ``(i, function)``, i.e. it converts fixed values given as
        constants to (trivial) functions.

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            values to fix, beyond what's already in ``self.fix_values``

        Returns
        -------
        fix_values : list
            same as input, plus internal ``self.fix_values`` and constants
            resolved

        See also
        --------
        Fit.fix_values, get_value_fixer
        """
        # Converts fix_values to useable format (i, function)
        # Also appends self.fix_values
        if fix_values is None:
            fix_values = []
        fix_values = fix_values + self.fix_values
        
        ifix = [i for i, _ in fix_values]
        _, ind = np.unique(ifix, return_index=True)
        full = []
        for i in ind:
            ip, fun = fix_values[i]
            if not callable(fun):
                fun = lambda x, val=fun : val # `fun` is (initially) of course a misnomer in this case

            full.append((ip, fun))
        return full

    def number_of_fit_parameters(self, fix_values=None):
        """
        Get the number of independent fit parameters

        The number of independent fit parameters is the full number of
        parameters (``len(params)``) minus the number of parameters that are
        fixed via the ``fix_values`` mechanism. This function is provided for
        convenience.

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            should be the same as was / will be handed to `run`

        Returns
        -------
        int
            the number of fit parameters
        """
        n_fixed = len(self.expand_fix_values(fix_values)) # also ensures uniqueness of indices
        return len(self.bounds) - n_fixed
                
    def get_value_fixer(self, fix_values=None):
        """
        Create a function that resolves fixed parameters

        Parameters
        ----------
        fix_values : list of tuples (i, fix_to), optional
            values to fix, beyond what's already in ``self.fix_values``

        Returns
        -------
        fixer : callable
            a function with signature ``fixer(params) --> params``, where in
            the output array all parameters fixes are applied.
        """
        fix_values = self.expand_fix_values(fix_values)

        n_params = len(self.bounds)
        is_fixed = np.zeros(n_params, dtype=bool)
        for i, _ in fix_values:
            is_fixed[i] = True
        
        def value_fixer(params, fix_values=fix_values, is_fixed=is_fixed, n_params=n_params):
            if len(params) == n_params:
                fixed_params = params.copy()
            elif len(params) + np.sum(is_fixed) == n_params:
                fixed_params = np.empty(n_params, dtype=float)
                fixed_params[:] = np.nan
                fixed_params[~is_fixed] = params
            else: # pragma: no cover
                raise RuntimeError(f"Did not understand how to fix up parameters: len(params) = {len(params)}, sum(is_fixed) = {np.sum(is_fixed)}, n_params = {n_params}")

            for ip, fixfun in fix_values:
                fixed_params[ip] = fixfun(fixed_params)
            
            return fixed_params

        return value_fixer
    
    def get_min_target(self, offset=0, fix_values=None, do_fixing=True):
        """
        Define the minimization target (negative log-likelihood)

        Parameters
        ----------
        offset : float
            constant to subtract from log-likelihood. See Notes section of
            class doc.
        fix_values : list of tuples (i, fix_to)
            values to fix, beyond what's already in ``self.fix_values``
        do_fixing : bool
            set to ``False`` to prevent the minimization target from resolving
            any of the parameter fixes (by default). Might be useful when
            exploring parameter space.

        Returns
        -------
        min_target : callable
            function with signature ``min_target(params) --> float``.

        Notes
        -----
        The returned ``min_target`` takes additional keyword arguments:

        - ``just_return_full_params`` : bool, ``False`` by default. If
          ``True``, don't calculate the actual target function, just return the
          parameter values after fixing
        - ``do_fixing`` : bool, defaults to `!do_fixing` as given to
          `get_min_target`.
        - fixer, offset : just handed over as arguments for style (scoping)

        See also
        --------
        run
        """
        fixer = self.get_value_fixer(fix_values)
        
        def min_target(params, just_return_full_params=False,
                       do_fixing=do_fixing, fixer=fixer, offset=offset,
                       ):
            if do_fixing:
                params = fixer(params)
            if just_return_full_params:
                return params
            
            penalty = self._penalty(params)
            if penalty < 0: # infeasible
                return self.max_penalty
            else:
                return -ds_logL(self.data,
                                self.ss_order,
                                self.params2msdm(params),
                               ) \
                       - self.logprior(params) \
                       + penalty \
                       - offset
            
        return min_target

    def run(self,
            init_from = None,
            optimization_steps=('simplex',),
            maxfev=None,
            fix_values = None,
            full_output=False,
            show_progress=False,
            verbosity=None,
           ):
        """
        Run the fit

        Parameters
        ----------
        init_from : dict
            initial point for the fit, as a dict with fields ``'params'`` and
            ``'logL'``, like the ones this function returns. If you just want
            to initialize from a certain parameter point, you can set
            ``init_from['logL'] = 0``.
        optimization_steps : tuple
            successive optimization steps to perform. Entries should be
            ``'simplex'`` for Nelder-Mead, ``'gradient'`` for gradient descent,
            or a dict whose entries will be passed to
            ``scipy.optimize.minimize`` as keyword arguments.
        maxfev : int or None
            limit on function evaluations for ``'simplex'`` or ``'gradient'``
            optimization steps
        fix_values : list of tuples (i, fix_to)
            can be used to keep some parameter values fixed or express them as
            function of the other parameters. See class doc for more details.
        full_output : bool
            Set to ``True`` to return the output dict (c.f. Returns) and the
            full output from ``scipy.optimize.minimize`` for each optimization
            step. Otherwise (``full_output == False``, the default) only the
            output dict from the ultimate run is returned.
        show_progress : bool
            display a `!tqdm` progress bar while fitting
        verbosity : {None, 0, 1, 2, 3}
            if not ``None``, overwrites the internal ``self.verbosity`` for
            this run. Use to silence or get more details of what's happening

        Returns
        -------
        dict
            with fields ``'params'``, a complete set of parameters; ``'logL'``,
            the associated value of the likelihood (or posterior, if the prior
            is non-trivial).
        """
        if verbosity is not None:
            tmp = self.verbosity
            self.verbosity = verbosity
            verbosity = tmp
        self.data.restoreSelection(self.data_selection)
        for step in optimization_steps:
            assert type(step) is dict or step in {'simplex', 'gradient'}
        
        # Initial values
        if init_from is None:
            p0 = self.initial_params()
            total_offset = self.initial_offset()
        else:
            p0 = deepcopy(init_from['params'])
            total_offset = -init_from['logL']
        
        # Adjust for fixed values
        ifix = [i for i, _ in self.expand_fix_values(fix_values)]
        bounds = deepcopy(self.bounds)
        for i in sorted(ifix, reverse=True):
            del bounds[i]
            p0[i] = np.nan
        p0 = p0[~np.isnan(p0)]
        
        # Set up progress bar
        bar = tqdm(disable = not show_progress)
        def callback(x):
            bar.update()
        
        # Go!
        all_res = []
        with np.errstate(all='raise'):
            for istep, step in enumerate(optimization_steps):
                if step == 'simplex':
                    options = {'fatol' : 0.1, 'xatol' : 0.01}
                    self.vprint(3, "Note: why is `xatol = 0.01` a good choice? / do we need it?")
                    if maxfev is not None:
                        options['maxfev'] = maxfev
                    kwargs = dict(method = 'Nelder-Mead',
                                  options = options,
                                  bounds = bounds,
                                  callback = callback,
                                 )
                elif step == 'gradient':
                    options = {}
                    if maxfev is not None:
                        options['maxfun'] = maxfev
                    kwargs = dict(method = 'L-BFGS-B',
                                  options = options,
                                  bounds = bounds,
                                  callback = callback,
                                 )
                else:
                    kwargs = dict(callback = callback,
                                  bounds = bounds,
                                 )
                    kwargs.update(step)
                    
                min_target = self.get_min_target(offset=total_offset, fix_values=fix_values)
                try:
                    fitres = optimize.minimize(min_target, p0, **kwargs)
                except BadCovarianceError as err: # pragma: no cover
                    self.vprint(2, "BadCovarianceError:", err)
                    fitres = lambda: None
                    fitres.success = False
                
                if not fitres.success:
                    self.vprint(1, f"Fit (step {istep}: {step}) failed. Here's the result:")
                    self.vprint(1, '\n', fitres)
                    raise RuntimeError("Fit failed at step {:d}: {:s}".format(istep, step))
                else:
                    all_res.append(({'params' : min_target(fitres.x, just_return_full_params=True),
                                     'logL' : -(fitres.fun+total_offset),
                                    }, fitres))
                    p0 = fitres.x
                    total_offset += fitres.fun
                    
        bar.close()
        if verbosity is not None:
            self.verbosity = verbosity
        
        if full_output:
            return all_res
        else:
            return all_res[-1][0]

################## Profiler class definition ###################################

class Profiler():
    """
    Exploration of the posterior after finding the point estimate

    This class provides a top layer on top of `Fit`, enabling more
    comprehensive exploration of the posterior after finding the (MAP) point
    estimate. Generally, it operates in two modes:

    - conditional posterior: wiggle each individual parameter, keeping all
      others fixed to the point estimate values, thus calculating conditional
      posterior values. In parameter space, this is a (multi-dimensional) cross
      along the coordinate axes.
    - profile posterior: instead of simply evaluating the posterior at each
      wiggle, keep the parameter fixed at the new value and optimize all
      others. Thus, in parameter space we are following the ridges of the
      posterior.

    From a Bayesian point of view, beyond conditional and profile posterior,
    the actually interesting quantity is of course the marginal posterior. This
    is best obtained by sampling the posterior by MCMC (see module description
    for an example implementation of such a sampler). Still, conditional or
    profile posterior often can give a useful overview over the shape of the
    posterior. The conditional posterior is also great for setting MCMC step
    sizes. Profile posteriors are of course significantly more expensive than
    conditionals.

    At the end of the day, this class moves along either profile or conditional
    posterior until it drops below a given cutoff, and then gives the lower
    bound, best value, and upper bound for each parameter. Inspired by
    frequentist confidence intervals, we determine the cutoff from a
    "confidence level" 1-α.

    We follow a two-step procedure to find the lower and upper bounds: first,
    we move out from the point estimate with a fixed step size (additively or
    multiplicatively) until the posterior drops below the cutoff. Within the
    thus defined bracket, we then find the actual bound to the desired accuracy
    by bisection. If the first step fails (i.e. the posterior does not drop
    below the cutoff far from the point estimate) the corresponding parameter
    direction is unidentifiable and the bound will be set to infinity
    
    Note that this class will also take care of the initial point estimate, if
    necessary.

    The main point of entry for the user is the `find_MCI` method, that
    performs the sweeps described above. Further, `run_fit` might be useful if
    you just need the point estimate.

    Parameters
    ----------
    fit : Fit
        the `Fit` object to use
    profiling : bool
        whether to run in profiling (``True``) or conditional (``False``) mode.
    conf : float
        the "confidence level" to use for the likelihood ratio cutoff
    conf_tol : float
        acceptable tolerance in the confidence level
    bracket_strategy : 'auto', dict, or list of dict
        specifies the strategy to use during the first step (pushing the
        parameter out to find a proper bracket). For full control, specify a
        list of dicts (one for each parameter) with the structure
        ``dict(multiplicative=(bool), step=(float),
        nonidentifiable_cutoffs=(low, high))``, where ``multiplicative``
        indicates whether propagation is ``param <-- param*step`` or ``param
        <-- param + step``. Correspondingly, ``step`` should be ``>1`` for
        ``multiplicative == True``, and ``step > 0`` for ``multiplicative ==
        False``. Finally, ``nonidentifiability_cutoffs`` specifies how far to
        search before the direction is considered unidentifiable. This is
        measured in "total stepsize" taken in either direction, so e.g. for a
        multiplicative strategy the default is ``nonidentifiability_cutoffs =
        (10, 10)``, meaning we search by a factor of 10 around the point
        estimate in either direction.
        
        Instead of specifying a list of dicts, you can also just give a single
        dict that will be applied to all parameters. Finally, by default this
        argument is set to ``'auto'``, which means that the strategy is
        determined from the bounds in `!fit`. If the lower bound is positive,
        the strategy is multiplicative, else additive.
    bracket_step : float
        global default for the ``'step'`` parameter if ``bracket_strategy ==
        'auto'``. Additive strategies use ``step = point_estimate *
        (bracket_step - 1)``.
    max_fit_runs : int
        an upper bound for how often to re-run the fit (when profiling).
    max_restarts : int
        sometimes the initial fit to find the point estimate might not converge
        properly, such that a better point estimate is found during the
        profiling runs. If that happens, the whole procedure is restarted from
        the new point estimate. This variable provides an upper bound on the
        number of these restarts. See also `!restart_on_better_point_estimate`
        below.
    verbosity : {0, 1, 2, 3}
        controls amount of messages during profiling. 0: nothing; 1: warnings
        only; 2: informational; 3: debugging

    Attributes
    ----------
    fit : Fit
        see Parameters
    min_target_from_fit : callable
        the minimization target of ``self.fit``
    ress : list
        storage of all evaluated parameter points. Each entry corresponds to a
        parameter dimension and contains a list of dicts that were obtained
        while sweeping that parameter. The individual entries are dicts like
        the one returned by `Fit.run`, with fields ``'params'`` and ``'logL'``.
    point_estimate : dict
        the MAP point estimate, a dict like the other points in `!ress`.
    conf, conf_tol : float
        see Parameters
    LR_target : float
        the target decline in the posterior value we're searching for
    LR_interval : [float, float]
        acceptable interval corresponding to `!conf_tol`
    iparam : int
        the index of the parameter currently being sweeped. This is a class
        attribute mainly for readability of the code.
    profiling : bool
        see Parameters
    bracket_strategy, bracket_step : list, float
        see Parameters
    max_fit_runs : int
        see Parameters
    run_count : int
        counts the runs executed so far
    max_restarts_per_parameters : int
        same as `!max_restarts` parameter.
    verbosity : int
        see Parameters
    restart_on_better_point_estimate : bool
        whether to restart upon finding a better point estimate. Might make
        sense to disable if your posterior is rugged
    bar : tqdm progress bar
        the progress bar showing successive fit evaluations

    See also
    --------
    run_fit, find_MCI
    """
    def __init__(self, fit,
                 profiling=True,
                 conf=0.95, conf_tol=0.001,
                 bracket_strategy='auto', # 'auto', dict(multiplicative=(bool),
                                          #              step=(float>1),
                                          #              nonidentifiable_cutoffs=[(low, high)],
                                          #             ),
                                          #         or list of such (one for each parameter)
                 bracket_step=1.2, # global default, but would be overridden by specifying bracket_strategy
                 max_fit_runs=100,
                 max_restarts=10,
                 verbosity=1, # 0: print nothing, 1: print warnings, 2: print everything, 3: debugging
                ):
        self.fit = fit
        self.min_target_from_fit = fit.get_min_target()

        self.ress = [[] for _ in range(len(self.fit.bounds))] # one for each iparam
        self.point_estimate = None
        
        self.conf = conf
        self.conf_tol = conf_tol

        self.iparam = None
        self.profiling = profiling # also sets self.LR_interval and self.LR_target

        self._bracket_strategy_input = bracket_strategy # see expand_bracket_strategy()
        self.bracket_step = bracket_step
        
        self.max_fit_runs = max_fit_runs
        self.run_count = 0
        self.max_restarts_per_parameter = max_restarts
        self.verbosity = verbosity

        self.restart_on_better_point_estimate = True
        
        self.bar = None
        
    ### Internals ###
        
    def vprint(self, verbosity, *args, **kwargs):
        """
        Prints only if ``self.verbosity >= verbosity``.
        """
        if self.verbosity >= verbosity:
            print(f"[msdfit.Profiler @ {self.run_count:d}]", (verbosity-1)*'--', *args, **kwargs)

    @property
    def profiling(self):
        return self._profiling
    
    @profiling.setter
    def profiling(self, val):
        # Make sure to keep LR_* in step with the profiling setting
        self._profiling = val

        if self.profiling:
            dof = 1
        else:
            n_params = len(self.fit.bounds)
            n_fixed = len(self.fit.fix_values)
            dof = n_params - n_fixed
            
        self.LR_interval = [stats.chi2(dof).ppf(self.conf-self.conf_tol)/2,
                            stats.chi2(dof).ppf(self.conf+self.conf_tol)/2]
        self.LR_target = np.mean(self.LR_interval)
        
    
    def expand_bracket_strategy(self):
        """
        Preprocessor for the `!bracket_strategy` attribute
        """
        # We need a point estimate to set up the additive scheme, so this can't be in __init__()
        if self.point_estimate is None: # pragma: no cover
            raise RuntimeError("Cannot set up brackets without point estimate")
            
        if self._bracket_strategy_input == 'auto':
            assert self.bracket_step > 1
            
            self.bracket_strategy = []
            for iparam, bounds in enumerate(self.fit.bounds):
                multiplicative = bounds[0] > 0
                if multiplicative:
                    step = self.bracket_step
                    nonidentifiable_cutoffs = [10, 10]
                else:
                    pe = self.point_estimate['params'][iparam]
                    step = np.abs(pe)*(self.bracket_step - 1)
                    nonidentifiable_cutoffs = 2*[2*np.abs(pe)]
                
                self.bracket_strategy.append({
                    'multiplicative'          : multiplicative,
                    'step'                    : step,
                    'nonidentifiable_cutoffs' : nonidentifiable_cutoffs,
                })
        elif isinstance(self._bracket_strategy_input, dict):
            assert self._bracket_strategy_input['step'] > (1 if self._bracket_strategy_input['multiplicative'] else 0)
            self.bracket_strategy = len(self.fit.bounds)*[self._bracket_strategy_input]
        else:
            assert isinstance(self._bracket_strategy_input, list)
            self.bracket_strategy = self._bracket_strategy_input
        
    class FoundBetterPointEstimate(Exception):
        pass
    
    def restart_if_better_pe_found(fun):
        """
        Decorator taking care of restarts

        We use the `Profiler.FoundBetterPointEstimate` exception to handle
        restarts. So a function decorated with this decorator can just raise
        this exception and will then be restarted properly.
        """
        def decorated_fun(self, *args, **kwargs):
            for restarts in range(self.max_restarts_per_parameter):
                try:
                    return fun(self, *args, **kwargs)
                except Profiler.FoundBetterPointEstimate:
                    self.vprint(1, f"Warning: Found a better point estimate ({self.best_estimate['logL']} > {self.point_estimate['logL']})")
                    self.vprint(1, f"Will restart from there ({self.max_restarts_per_parameter-restarts} remaining)")
                    fit_kw = {}

                    # Some housekeeping
                    self.run_count = 0
                    fit_kw['show_progress'] = self.bar is not None

                    # If we're not calculating a profile likelihood, it does not make
                    # sense to keep the old results, since the parameters are different
                    if not self.profiling:
                        fit_kw['init_from'] = self.best_estimate
                        self.ress = [[] for _ in range(len(self.fit.bounds))]
                        self.point_estimate = None

                    # Get a new point estimate, starting from the better one we found
                    # Note that run_fit() starts from best_estimate if
                    # 'init_from' is not specified
                    self.vprint(2, "Finding new point estimate ...")
                    self.run_fit(**fit_kw)

            # If this loop runs out of restarts, we're pretty screwed overall
            raise RuntimeError("Ran out of restarts after finding a better "
                              f"point estimate (max_restarts = {self.max_restarts_per_parameter})") # pragma: no cover

        return decorated_fun
    
    ### Point estimation ###
        
    @staticmethod
    def likelihood_significantly_greater(res1, res2):
        """
        Helper function

        The threshold is 1e-3. Note that this is an asymmetric operation, i.e.
        there is a regime where neither `!res1` significantly greater `!res2`
        nor the other way round.

        Parameters
        ----------
        res1, res2 : dict
            like the output of `Fit.run`, and the entries of ``self.ress``.

        Returns
        -------
        bool
        """
        return res1['logL'] > res2['logL'] + 1e-3 # smaller differences are irrelevant for likelihoods
            
    @property
    def best_estimate(self):
        """
        The best current estimate

        This should usually be the point estimate, but we might find a better
        one along the way.
        """
        if self.point_estimate is None:
            return None
        else:
            best = self.point_estimate
            for ress in self.ress:
                try:
                    candidate = ress[np.argmax([res['logL'] for res in ress])]
                except ValueError: # argmax([])
                    continue
                if self.likelihood_significantly_greater(candidate, best):
                    best = candidate
            return best
    
    def check_point_estimate_against(self, res):
        """
        Check whether `!res` is better than current point estimate

        Parameters
        ----------
        res : dict
            the evaluated parameter point to check. A dict like the ones in
            ``self.ress``.

        Raises
        ------
        Profiler.FoundBetterPointEstimate
        """
        if (
                self.point_estimate is not None
            and self.likelihood_significantly_greater(res, self.point_estimate)
            and self.restart_on_better_point_estimate
            ):
            raise Profiler.FoundBetterPointEstimate
    
    def run_fit(self, is_new_point_estimate=True,
                **fit_kw,
               ):
        """
        Execute one fit run

        This is used to find the initial point estimate, as well as subsequent
        evaluations of the profile posterior.

        Parameters
        ----------
        is_new_point_estimate : bool
            whether we are looking for a new point estimate or just evaluating
            a profile point. This just affects where the result is stored
            internally
        fit_kw : keyword arguments
            additional parameters for `Fit.run`

        See also
        --------
        find_MCI
        """
        self.run_count += 1
        if self.run_count > self.max_fit_runs:
            raise RuntimeError(f"Ran out of likelihood evaluations (max_fit_runs = {self.max_fit_runs})")
            
        if 'init_from' not in fit_kw:
            fit_kw['init_from'] = self.best_estimate
        
        if self.point_estimate is None and fit_kw['init_from'] is None: # very first fit, so do simplex --> (gradient)
            res = self.fit.run(optimization_steps = ('simplex',),
                               **fit_kw,
                              )
            try: # try to refine
                fit_kw['init_from'] = res
                fit_kw['show_progress'] = False
                res = self.fit.run(optimization_steps = ('gradient',),
                                   **fit_kw,
                                  )
            except RuntimeError: # okay, this didn't work, whatever # pragma: no cover
                self.vprint(2, "Gradient refinement failed. Point estimate might be imprecise. Not a fatal error, resuming operation")

        else: # we're starting from somewhere known, so start out trying to
              # move by gradient, use simplex if that doesn't work
            try:
                res = self.fit.run(optimization_steps = ('gradient',),
                                   verbosity=0,
                                   **fit_kw,
                                  )
            except RuntimeError:
                self.vprint(2, "Gradient fit failed, using simplex")
                res = self.fit.run(optimization_steps = ('simplex',),
                                   **fit_kw,
                                  )
            # At this point, we used to check that the new result is indeed
            # better than the initial point, which was intended as a sanity
            # check. It ended up being problematic: when we are profiling, we
            # initialize to the closest available previous point, which of
            # course will not fulfill the constraint on the parameter of
            # interest. It's associated likelihood (which we would compare
            # against) thus can be better than any possible value that fulfills
            # the constraint. Long story short: no sanity check here.
        
        if is_new_point_estimate:
            self.point_estimate = res
        else:
            self.ress[self.iparam].append(res)
            self.check_point_estimate_against(res)

        if self.bar is not None:
            self.bar.update() # pragma: no cover
        
    ### Sweeping one parameter ###
    
    def find_closest_res(self, val, direction=None):
        """
        Find the closest previously evaluated point

        Parameters
        ----------
        val : float
            new value
        direction : {None, -1, 1}
            search only for existing values that are greater (1) or smaller
            (-1) than the specified one.

        Returns
        -------
        dict
            appropriate point from ``self.ress``

        See also
        --------
        profile_likelihood
        """
        ress = self.ress[self.iparam] + [self.point_estimate]
        
        values = np.array([res['params'][self.iparam] for res in ress])
        if val in values:
            i = np.argmax([res['logL'] for res in ress if res['params'][self.iparam] == val])
            return ress[i]
        
        if self.bracket_strategy[self.iparam]['multiplicative']:
            distances = np.abs([np.log(val/res['params'][self.iparam]) for res in ress])
        else:
            distances = np.abs([val-res['params'][self.iparam] for res in ress])
            
        if direction is not None:
            distances[np.sign(values - val) != direction] = np.inf
            if not np.any(np.isfinite(distances)):
                raise RuntimeError("Did not find any values in specified direction")
            
        min_dist = np.min(distances)
        
        i_candidates = np.nonzero(distances < min_dist+1e-10)[0] # We use bisection, so usually there will be two candidates
        ii_cand = np.argmax([ress[i]['logL'] for i in i_candidates])
        
        return ress[i_candidates[ii_cand]]
    
    def profile_likelihood(self, value, init_from='closest'):
        """
        Evaluate profile (or conditional) likelihood / posterior

        Parameters
        ----------
        value : float
            value of the current parameter of interest (``self.iparam``)
        init_from : dict or 'closest'
            from where to start the optimization (only relevant if
            ``self.profiling``). Set to 'closest' to use
            ``self.find_closest_res``.

        Returns
        -------
        dict
            like in ``self.ress`` (and also stored there)
        """
        if self.profiling:
            if init_from == 'closest':
                init_from = self.find_closest_res(value)

            self.run_fit(init_from = init_from,
                         fix_values = [(self.iparam, value)],
                         is_new_point_estimate = False,
                        )
        else:
            new_params = self.point_estimate['params'].copy()
            new_params[self.iparam] = value
            minus_logL = self.min_target_from_fit(new_params)
                
            if self.bar is not None:
                self.bar.update() # pragma: no cover
            
            self.ress[self.iparam].append({'logL' : -minus_logL, 'params' : new_params})
            self.check_point_estimate_against(self.ress[self.iparam][-1])
            
        return self.ress[self.iparam][-1]['logL']
    
    def iterate_bracket_point(self, x0, pL, direction,
                              step = None,
                             ):
        """
        First step of the procedure ("bracketing")

        In this first step we simply push out from the point estimate to try
        and establish a bracket containing the exact boundary point. See
        `!Profiler.bracket_strategy`. This function implements that push, for
        one parameter in one direction.

        Parameters
        ----------
        x0, pL : float
            the parameter value and associated posterior value to start from.
            Should be from the point estimate
        direction : {-1, 1}
            which direction to go in
        step : float or None
            use to override ``self.bracket_strategy[self.iparam]['step']``,
            which is used by default.

        Returns
        -------
        x, pL : float
            the found bracket end point and associated posterior. If the chosen
            direction turns out to be unidentifiable, ``x`` is set to the
            corresponding bound (might be ``inf``), and ``pL = np.inf``.

        See also
        --------
        initial_bracket_points
        """
        self.expand_bracket_strategy()
        if step is None:
            step = self.bracket_strategy[self.iparam]['step']
            
        bracket_param = 1 if self.bracket_strategy[self.iparam]['multiplicative'] else 0
        p = x0
        pL_thres = self.point_estimate['logL'] - self.LR_target
        while pL > pL_thres:
            self.vprint(3, "bracketing: {:.3f} > {:.3f} @ {}".format(pL, pL_thres, p))
            
            # Check identifiability cutoff
            ib = int((1+direction)/2)
            if bracket_param > self.bracket_strategy[self.iparam]['nonidentifiable_cutoffs'][ib]:
                self.vprint(2, "{} edge of confidence interval is non-identifiable".format('left' if direction == -1 else 'right'))
                p = self.fit.bounds[self.iparam][ib]
                pL = np.inf
                break

            # Update position
            if self.bracket_strategy[self.iparam]['multiplicative']:
                bracket_param *= step
                p = x0 * bracket_param**direction
            else:
                bracket_param += step
                p = x0 + direction*bracket_param

            # Update profile likelihood
            pL = self.profile_likelihood(p)
        else:
            self.vprint(3, "bracketing: {:.3f} < {:.3f} @ {}".format(pL, pL_thres, p))
            
        return p, pL
        
    def initial_bracket_points(self):
        """
        Execute bracketing in both directions

        See also
        --------
        find_bracket_point, find_MCI
        """
        self.expand_bracket_strategy()
        a, a_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction=-1)
        b, b_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction= 1)
        return (a, a_pL), (b, b_pL)
    
    def solve_bisection(self, bracket, bracket_pL):
        """
        Find exact root by recursive bisection

        Parameters
        ----------
        bracket : [float, float]
            the bracket of parameter values containing the root
        bracket_pL : [float, float]
            the posterior values associated with the bracket points

        Returns
        -------
        float
            the root within the bracket, to within the given precision (c.f
            ``self.conf_tolerance``)

        Notes
        -----
        All points evaluated along the way are stored in ``self.ress``
        """
        c = np.mean(bracket)
        c_pL = self.profile_likelihood(c)
        
        a_fun, b_fun = self.point_estimate['logL'] - bracket_pL - self.LR_target
        c_fun = self.point_estimate['logL'] - c_pL - self.LR_target
        i_update = 0 if a_fun*c_fun > 0 else 1
        assert [a_fun, b_fun][1-i_update]*c_fun <= 0
        
        bracket[i_update] = c
        bracket_pL[i_update] = c_pL
        
        if np.all([self.LR_interval[0] < LR < self.LR_interval[1] for LR in self.point_estimate['logL'] - bracket_pL]):
            return np.mean(bracket)
        else:
            return self.solve_bisection(bracket, bracket_pL)
    
    @restart_if_better_pe_found
    def find_single_MCI(self, iparam):
        """
        Run the whole profiling process for one parameter

        Parameters
        ----------
        iparam : int
            index of the parameter to sweep

        Returns
        -------
        m : float
            the point estimate value for this parameter
        ci : np.array([float, float])
            the bounds where the posterior has dropped below the maximum value
            the specified amount (c.f. ``self.conf``)

        See also
        --------
        find_MCI
        """
        self.iparam = iparam
        if self.point_estimate is None:
            raise RuntimeError("Need to have a point estimate before calculating confidence intervals") # pragma: no cover
            
        (a, a_pL), (b, b_pL) = self.initial_bracket_points()
        m = self.point_estimate['params'][self.iparam]
        m_pL = self.point_estimate['logL']
        
        roots = np.array([np.nan, np.nan])
        if a_pL == np.inf:
            roots[0] = a
        if b_pL == np.inf:
            roots[1] = b
        
        for i, (bracket, bracket_pL) in enumerate(zip([(a, m), (m, b)], [(a_pL, m_pL), (m_pL, b_pL)])):
            if np.isnan(roots[i]):
                roots[i] = self.solve_bisection(np.asarray(bracket), np.asarray(bracket_pL))
            if i == 0:
                self.vprint(2, f"Found left edge, now searching right one @ {roots[i]}")
        
        self.vprint(2, f"found CI = {roots} (point estimate = {m}, iparam = {self.iparam})\n")
        return m, roots
    
    def find_MCI(self, iparam='all',
                 show_progress=False,
                ):
        """
        Perform the sweep for multiple parameters

        Parameters
        ----------
        iparam : 'all' or np.ndarray, dtype=int
            the parameters to sweep. If 'all', will sweep all independent
            parameters, i.e. those not fixed via the `Fit.fix_values`
            mechanism.

        Returns
        -------
        mcis : (n, 3) np.ndarray, dtype=float
            for each parameter, in order: point estimate, lower, and upper
            bound.

        See also
        --------
        run_fit
        """
        if show_progress and self.bar is None:
            self.bar = tqdm() # pragma: no cover
            
        if self.point_estimate is None:
            self.vprint(2, "Finding initial point estimate ...")
            self.run_fit(show_progress=show_progress)
        
        self.vprint(2, "initial point estimate: params = {}, logL = {}\n".format(self.point_estimate['params'],
                                                                                 self.point_estimate['logL'],
                                                                                ))
        
        n_params = len(self.point_estimate['params'])
        mcis = np.empty((n_params, 3), dtype=float)
        mcis[:] = np.nan
        
        # check, which parameters to run
        if iparam == 'all':
            fixed = [i for i, _ in self.fit.fix_values]
            iparams = np.array([i for i in range(n_params) if i not in fixed])
        else:
            iparams = np.asarray(iparam)
            if len(iparams.shape) == 0:
                iparams = np.array([iparams])
               
        # keep track of which point estimate was used for which parameters
        used_point_estimates = n_params*[None]
        while True: # limited by max_restarts_per_parameter
            for iparam in iparams:
                # Only run if we did not already sweep from here
                if used_point_estimates[iparam] is not self.point_estimate:
                    self.run_count = 0
                    self.vprint(2, f"starting iparam = {iparam}")
                    m, ci = self.find_single_MCI(iparam)
                    mcis[iparam, :] = m, *ci
                    used_point_estimates[iparam] = self.point_estimate
                
            if ( all([used_point_estimates[iparam] is self.point_estimate for iparam in iparams])
                 or not self.restart_on_better_point_estimate ):
                break
        
        if show_progress and self.bar is not None: # pragma: no cover
            self.bar.close()
            self.bar = None
            
        self.vprint(2, "Done\n")
        
        if len(iparams) == 1:
            return mcis[iparams[0]]
        else:
            return mcis
