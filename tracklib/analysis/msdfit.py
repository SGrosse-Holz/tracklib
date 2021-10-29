"""
Likelihood (i.e. "proper") MSD fitting

In fact, we should talk about autocorrelation fitting, as that is the central
object (since it defines the associated Gaussian Process). However, in the
context of single particle tracking it seems that people prefer MSDs. These are
equivalent viewpoints, as outlined below. This module works with ACFs, but is
well set up for translation to MSDs.

The basic idea is that as soon as we assume steady state, we can immediately
define a covariance operator Σ(t, t') := γ(|t-t'|) from an autocorrelation
function. We can then use this covariance operator to marginalize to time
points where we have data, thus obtain a covariance matrix, and calculate a
likelihood of observing those data.

The above paragraph is vague in that it does not state what autocorrelation
function we are actually talking about, and there are two important versions:
 + if the trajectory itself is sampled from a steady state (e.g. distance of
   points on a polymer) then we use the autocorrelation function of the actual
   data points. The information contained therein is exactly equivalent to the
   MSD + steady state variance. We will call this a steady state of order 0.
   The relation between MSD μ(k) and steady state variance V, and process
   autocorrelation γ(k) is
        μ(k) = 2*( γ(0) - γ(k) ), V = γ(0)
 + in many cases (e.g. sampling a diffusing particle's position) the trajectory
   itself will not come from a steady state process, but its displacements do.
   In this case, which we call steady state of order 1, the autocorrelation
   function that we wish to fit is the one of the displacement process. In this
   case, the information in autocorrelation (of the displacement process) γ(k)
   and MSD μ(k) is exactly the same, and we have
        γ(k) = 1/2*( μ(k+1) - μ(k-1) + 2μ(k) ).
   For inversion note that μ(-1) = μ(1) = γ(0) and μ(0) = 0.
 + theoretically, one could imagine higher order steady states, i.e. the n-th
   order difference of the trajectory comes from a steady state. We trust that
   this case is of minor importance and can thus safely be ignored (for now).
Note the provided functions to convert between (MSD, V) <--> ACF for order 0
and MSD <--> ACF for order 1

At the end of the day, it seems that for steady states of order 0, the natural
description is via the autocorrelation function, while for steady states of
order 1 it is the MSD. Note however that a fair bit of (my) intuition is
associated with MSDs.

This module's main interface are the subclasses of `Fit`, which all implement
different schemes for fitting MSDs/ACFs to a given data set. `ds_logL` provides
the likelihood function for a given ACF on a given data set.

There is a model free approach that fits the MSD with a cubic spline, thus
giving an analytically meaningful MSD (i.e. one that actually defines a
Gaussian process) without having to assume a model. In many use cases, this
even has a (significantly) higher likelihood than the empirical MSD calculated
by `analysis.MSD`. In any case it seems like a better representation, since it
has way fewer parameters. In fact one can apply Bayesian model selection to
find the best number of spline knots.

Other than the model free version, so far we have
 + NPXFit: Noise + Powerlaw + X. This fits a powerlaw to the beginning of the
   MSD and compensates everything at long times with a spline, then adds some
   constant offset for localization error.
 + RouseFit, RouseFitDiscrete: might be moved eventually. Fit two-locus Rouse
   models, either with an infinite, continuous chain, or a model like
   `tracklib.models.Rouse`.
"""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools

import numpy as np
from scipy import linalg, optimize, stats

from tracklib.models import rouse
from tracklib.util import parallel
from tracklib.analysis.p2 import MSD

################## Implementation notes vs. first version #####################
##
##  - we move to a purely MSD based formulation. This means functions like params2msd() should
##    provide MSD(Δt) as a callable, able to handle Δt = inf (for ss_order = 0)
##  - we assume that MSD(Δt) is vectorized (if non-trivial: np.vectorize)
##  - we retain the drift / offset (ss_order = 1 / 0) term. It might come in handy at some point
    ##  - separation of interfaces: we define a `FitType`, taking on the role of the old `Fit` and
    ##    to be thought of as specifying *how* we run a fit (i.e. this is the one getting subclassed
    ##    for specific fit shapes), and a `Fitter`, taking on the role of the old `Profiler`, plus
    ##    running the initial fit for the point estimate
##  - edit: we keep the `Fit` and `Profiler` layout, it just works well
##  - note that profile likelihoods are not local optima, but only optimal in one direction  (well, duh).
##    this means that when selecting the closest point to start from, we should pick one that belongs to
##    the proper profile. We therefore augment `Fit`'s output dict with `iparam`.
##
###############################################################################

# Verbosity rules: 0 = no output, 1 = warnings only, 2 = informational, 3 = debug info
verbosity = 1
def vprint(v, *args, **kwargs):
    if verbosity >= v:
        print("[msdfit]", (v-1)*'--', *args, **kwargs)
        
###############################################################################

def MSDfun(fun):
    # Decorator to make some (vectorized) function R_+ --> R_+ a valid MSD function
    def wrap(dt):
        # Preproc
        dt = np.abs(np.asarray(dt))
        was_scalar = len(dt.shape) == 0
        if was_scalar:
            dt = np.array([dt])
        
        # Calculate
        msd = fun(dt)
        
        # Postproc
        msd[dt == 0] = 0
        if was_scalar:
            msd = msd[0]
        return msd
    return wrap

def msd2C(msd, ti, ss_order):
    if ss_order == 0:
        return 0.5*( msd(np.inf) - msd(ti[:, None] - ti[None, :]) )
    elif ss_order == 1:
        return 0.5*(  msd(ti[1:, None] - ti[None,  :-1]) + msd(ti[:-1, None] - ti[None, 1:  ])
                    - msd(ti[1:, None] - ti[None, 1:  ]) - msd(ti[:-1, None] - ti[None,  :-1]) )
    else:
        raise ValueError(f"Invalid stead state order: {ss_order}")

################## Gaussian Process likelihood ###############################

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)
GP_verbosity = 1
def GP_vprint(v, *args, **kwargs):
    if GP_verbosity >= v:
        print("[msdfit.GP]", (v-1)*'--', *args, **kwargs)

class BadCovarianceError(RuntimeError):
    pass

def _GP_core_logL(C, x):
    # likelihood calculation
    with np.errstate(under='ignore'):
        s, logdet = np.linalg.slogdet(C)
    if s <= 0:
        raise BadCovarianceError("Covariance not positive definite. slogdet = ({}, {})".format(s, logdet))
        
    try:
        xCx = x @ linalg.solve(C, x, assume_a='pos')
    except FloatingPointError as err:
        GP_vprint(3, f"Problem when inverting covariance, even though slogdet = ({s}, {logdet})")
        GP_vprint(3, type(err), err)
        raise BadCovarianceError("Inverting covariance did not work")

    return -0.5*(xCx + logdet) - len(C)*LOG_SQRT_2_PI

def GP_logL(trace, ss_order, msd, mean=0):
    # mean: mean if ss=0, drift if ss=1
    ti = np.nonzero(~np.isnan(trace))[0]
    
    if ss_order == 0:
        X = trace[ti] - mean
    elif ss_order == 1:
        X = np.diff(trace[ti]) - mean*np.diff(ti)
    else:
        raise ValueError(f"Invalid stead state order: {ss_order}")
    
    C = msd2C(msd, ti, ss_order)
    return _GP_core_logL(C, X)

def _GP_logL_for_parallelization(params):
    return GP_logL(*params)

def ds_logL(data, ss_order, msd_ms):
    # msd_ms : tuple (msd, m) or list of such
    d = data.map_unique(lambda traj : traj.d)
    if isinstance(msd_ms, list):
        if len(msd_ms) != d:
            raise ValueError(f"Dimensionality of MSD ({len(msd_ms)}) != dimensionality of data ({d})")
    else:
        msd, m = msd_ms
        if m != 0 and d > 1:
            raise ValueError(f"Cannot use mean (m={m}) when specifying single MSD for {d}-dimensional data")
        msd_ms = [(lambda dt: msd(dt)/d, m) for _ in range(d)]
        
    job_iter = itertools.chain.from_iterable((itertools.product((traj[:][:, dim] for traj in data),
                                                                [ss_order], [msd], [m],
                                                               )
                                              for dim, (msd, m) in enumerate(msd_ms)
                                             ))
    
    return np.sum(list(parallel._umap(_GP_logL_for_parallelization, job_iter)))

################## Fit base class definition ###################################

class Fit(metaclass=ABCMeta):
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
        #        x <= 0 : infeasible. Maximum penalization
        #   0 <= x <= 1 : smooth crossover: np.exp(1/tan(pi*x))
        #   0 <= x      : feasible. No penalization   
        self.constraints = [self.constraint_Cpositive]
        self.max_penalty = 1e10
        
        self.verbosity = 1
        
    def vprint(self, v, *args, **kwargs):
        if self.verbosity >= v:
            print("[msdfit.Fit]", (v-1)*'--', *args, **kwargs)
        
    ### To be overwritten / used upon subclassing ###
    
    @abstractmethod
    def params2msdm(self, params):
        """ Should return msd, m """
        raise NotImplementedError
    
    @abstractmethod
    def initial_params(self):
        raise NotImplementedError

    def initial_offset(self):
        return 0
                
    def constraint_Cpositive(self, params):
        msd, _ = self.params2msdm(params)
        min_ev_okay = 1 - np.cos(np.pi/self.T) # white noise min ev
        min_ev = np.min(np.linalg.eigvalsh(msd2C(msd, np.arange(self.T), self.ss_order)))
        return min_ev / min_ev_okay
    
    ### General machinery, usually won't need overwriting ###
        
    def _penalty(self, params):
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
                except:
                    return self.max_penalty
                
    def get_value_fixer(self, fix_values=()):
        # Set up fixed value machinery and make sure that everything is cast as
        # a function taking the given parameters and calculating the missing
        # one from it. Note note below.
        n_params = len(self.bounds)
        is_fixed = np.zeros(n_params, dtype=bool)
        for i, (ip, val) in enumerate(fix_values):
            is_fixed[ip] = True
            if not callable(val):
                fix_values[i] = (ip, lambda x, val=val : val)
        # Note that the above is pretty hacky: what we want to do is to convert
        # any fixed value into a function that returns that value. Because the
        # function is only evaluated at runtime, the straight-forward approach
        # ``lambda x: val`` does not work, because after all the definitions
        # have run, `val` just has the value of the last iteration. As an
        # example of this effect, note that
        # >>> funs = [lambda : val for val in np.arange(3)]
        # ... for fun in funs: print(fun())
        # prints "2 2 2".
        # We can circumvent this problem by passing the return value as an
        # argument with default value to the function. Default values are
        # evaluated at definition time, so that gives us the list of functions
        # that we want. In terms of the example above:
        # >>> funs = [lambda val=val : val for val in np.arange(3)]
        # ... for fun in funs: print(fun())
        # correctly prints "0 1 2".
        
        def value_fixer(params):
            fixed_params = np.empty(n_params, dtype=float)
            fixed_params[:] = np.nan
            fixed_params[~is_fixed] = params
            for ip, fixfun in fix_values:
                fixed_params[ip] = fixfun(fixed_params)
            
            return fixed_params
        
        return value_fixer
    
    def get_min_target(self, offset=0, fix_values=(), do_fixing=True):
        fixer = self.get_value_fixer(fix_values)
        
        def min_target(params, just_return_full_params=False, do_fixing=do_fixing):
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
                               ) + penalty - offset
            
        return min_target

    def run(self,
            init_from = None, # dict as returned by this function
            optimization_steps=('simplex',),
            maxfev=None,
            fix_values = None, # list of 2-tuples (i, val) to fix params[i] = val
            full_output=False,
            show_progress=False, assume_notebook_for_progress_bar=True,
            verbosity=None, # temporarily overwrite verbosity settings
           ):
        if verbosity is not None:
            tmp = self.verbosity
            self.verbosity = verbosity
            verbosity = tmp
        self.data.restoreSelection(self.data_selection)
        for step in optimization_steps:
            assert type(step) is dict or step in {'simplex', 'gradient'}
        if fix_values is None:
            fix_values = []
        
        # Initial values
        if init_from is None:
            p0 = self.initial_params()
            total_offset = self.initial_offset()
        else:
            p0 = deepcopy(init_from['params'])
            total_offset = -init_from['logL']
        
        # Adjust for fixed values
        fix_values = fix_values + self.fix_values # X = X + Y copies, whereas X += Y would modify the values passed as argument
        ifix = [i for i, _ in fix_values]
        bounds = deepcopy(self.bounds)
        for i in sorted(ifix, reverse=True):
            del bounds[i]
            p0[i] = np.nan
        p0 = p0[~np.isnan(p0)]
        
        # Set up beauty stuff
        if show_progress:
            if assume_notebook_for_progress_bar:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            bar = tqdm()
            del tqdm
        else:
            class Nullbar:
                def update(*args, **kwargs): pass
                def close(*args, **kwargs): pass
            bar = Nullbar()

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
                        options['maxfev'] = maxfev
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
                    
                min_target = self.get_min_target(total_offset, fix_values)
                try:
                    fitres = optimize.minimize(min_target, p0, **kwargs)
                except BadCovarianceError as err:
                    self.vprint(2, "BadCovarianceError:", err)
                    fitres = lambda: None
                    fitres.success = False
                
                if not fitres.success:
                    self.vprint(1, f"Fit (step {istep}: {step}) failed. Here's the result:")
                    self.vprint(1, fitres)
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
        self.ress = [[] for _ in range(len(self.fit.bounds))] # one for each iparam
        self.point_estimate = None
        
        self.iparam = None
        self.profiling = profiling
        self.LR_interval = [stats.chi2(1).ppf(conf-conf_tol)/2,
                            stats.chi2(1).ppf(conf+conf_tol)/2]
        self.LR_target = np.mean(self.LR_interval)
        
        self.bracket_strategy = bracket_strategy # see expand_bracket_strategy()
        self.bracket_step = bracket_step
        
        self.max_fit_runs = max_fit_runs
        self.run_count = 0
        self.max_restarts_per_parameter = max_restarts
        self.verbosity = verbosity
        
        self.bar = None
        
    ### Internals ###
        
    def vprint(self, verbosity, *args, **kwargs):
        if self.verbosity >= verbosity:
            print(f"[msdfit.Profiler @{self.run_count:3d}]", (verbosity-1)*'--', *args, **kwargs)
    
    def expand_bracket_strategy(self):
        # We need a point estimate to set up the additive scheme, so this can't be in __init__()
        if self.point_estimate is None:
            raise RuntimeError("Cannot set up brackets without point estimate")
            
        if self.bracket_strategy == 'auto':
            assert self.bracket_step > 1
            
            self.bracket_strategy = []
            for iparam, bounds in enumerate(self.fit.bounds):
                multiplicative = bounds[0] > 0
                if multiplicative:
                    step = self.bracket_step
                    nonidentifiable_cutoffs = [10, 10]
                else:
                    pe = self.point_estimate['params'][iparam]
                    step = pe*(self.bracket_step - 1)
                    nonidentifiable_cutoffs = 2*[2*np.abs(pe)]
                
                self.bracket_strategy.append({
                    'multiplicative'          : multiplicative,
                    'step'                    : step,
                    'nonidentifiable_cutoffs' : nonidentifiable_cutoffs,
                })
        elif isinstance(self.bracket_strategy, dict):
            assert self.bracket_strategy['step'] > 1
            self.bracket_strategy = len(self.fit.bounds)*[self.bracket_strategy]
        else:
            assert isinstance(self.bracket_strategy, list)
        
    class FoundBetterPointEstimate(Exception):
        pass
    
    def restart_if_better_pe_found(fun):
        def decorated_fun(self, *args, **kwargs):
            restarts = -1
            while restarts < self.max_restarts_per_parameter:
                try:
                    return fun(self, *args, **kwargs)
                except Profiler.FoundBetterPointEstimate:
                    self.vprint(1, ("Warning: Found a better point estimate."
                                    f"Will restart from there. ({self.max_restarts_per_parameter-restarts} remaining)"))
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
                    self.vprint(2, "Finding new point estimate ...")
                    self.run_fit(**fit_kw)
                    restarts += 1

            # If this while loop runs out of restarts, we're pretty screwed overall
            raise StopIteration(f"Ran out of restarts after finding a better point estimate (max_restarts = {self.max_restarts})")
        return decorated_fun
    
    ### Point estimation ###
        
    @staticmethod
    def likelihood_significantly_greater(res1, res2):
        return res1['logL'] > res2['logL'] + 1e-10
            
    @property
    def best_estimate(self):
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
    
    def current_point_estimate_is_worse_than(self, res):
        if self.point_estimate is not None and self.likelihood_significantly_greater(res, self.point_estimate):
            return True
        return False
    
    def run_fit(self, is_new_point_estimate=True,
                **fit_kw,
               ):
        self.run_count += 1
        if self.run_count > self.max_fit_runs:
            raise StopIteration(f"Ran out of likelihood evaluations (max_fit_runs = {self.max_fit_runs})")
            
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
            except RuntimeError: # okay, this didn't work, whatever
                pass
        else: # we're starting from somewhere known, so start out trying to move by gradient, use simplex if that doesn't work
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
        
        if is_new_point_estimate:
            self.point_estimate = res
        else:
            self.ress[self.iparam].append(res)
            if self.current_point_estimate_is_worse_than(res):
                raise Profiler.FoundBetterPointEstimate

        if self.bar is not None:
            self.bar.update()
        
    ### Sweeping one parameter ###
    
    def find_closest_res(self, val, direction=None):
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
            
        min_dist = np.min(distances)
        
        i_candidates = np.nonzero(distances < min_dist+1e-10)[0] # We use bisection, so usually there will be two candidates
        ii_cand = np.argmax([ress[i]['logL'] for i in i_candidates])
        
        return ress[i_candidates[ii_cand]]
    
    def profile_likelihood(self, value, init_from='closest'):
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
            
            minus_logL = self.fit.get_min_target()(new_params)
                
            if self.bar is not None:
                self.bar.update()
            
            self.ress[self.iparam].append({'logL' : -minus_logL, 'params' : new_params})
            if self.current_point_estimate_is_worse_than(self.ress[self.iparam][-1]):
                raise Profiler.FoundBetterPointEstimate
            
        return self.ress[self.iparam][-1]['logL']
    
    def iterate_bracket_point(self, x0, pL, direction,
                              step = None,
                             ):
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
        self.expand_bracket_strategy()
        a, a_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction=-1)
        b, b_pL = self.iterate_bracket_point(self.point_estimate['params'][self.iparam], self.point_estimate['logL'], direction= 1)
        return (a, a_pL), (b, b_pL)
    
    def solve_bisection(self, bracket, bracket_pL):
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
        self.iparam = iparam
        if self.point_estimate is None:
            raise RuntimeError("Need to have a point estimate before calculating confidence intervals")
            
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
                 show_progress=False, assume_notebook_for_progress_bar=True,
                ):
        if show_progress and self.bar is None:
            if assume_notebook_for_progress_bar:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            self.bar = tqdm()
            del tqdm
            
        if self.point_estimate is None:
            self.vprint(2, "Finding initial point estimate ...")
            self.run_fit(show_progress=show_progress)
        old_point_estimate = self.point_estimate
        
        self.vprint(2, "initial point estimate: params = {}, logL = {}\n".format(self.point_estimate['params'],
                                                                                 self.point_estimate['logL'],
                                                                                ))
        
        n_params = len(self.point_estimate['params'])
        mcis = np.empty((n_params, 3), dtype=float)
        mcis[:] = np.nan
        
        if iparam == 'all':
            fixed = [i for i, _ in self.fit.fix_values]
            iparams = np.array([i for i in range(n_params) if i not in fixed])
        else:
            iparams = np.asarray(iparam)
            if len(iparams.shape) == 0:
                iparams = np.array([iparams])
               
        while True: # limited by max_restarts_per_parameter
            for iparam in iparams:
                self.run_count = 0
                self.vprint(2, f"starting iparam = {iparam}")
                m, ci = self.find_single_MCI(iparam)
                mcis[iparam, :] = m, *ci
                
            if self.likelihood_significantly_greater(self.best_estimate, old_point_estimate):
                self.vprint(2, f"Point estimate was updated while we calculated confidence intervals, so restart")
                self.vprint(2, "new best logL = {:.3f} > {:.3f} = old point estimate logL\n".format(self.best_estimate['logL'],
                                                                                                    old_point_estimate['logL']))
            else:
                break
        
        if show_progress and self.bar is not None:
            self.bar.close()
            self.bar = None
            
        self.vprint(2, "Done\n")
        
        if len(iparams) == 1:
            return mcis[iparams[0]]
        else:
            return mcis
