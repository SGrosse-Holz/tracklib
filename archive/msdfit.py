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
from copy import deepcopy
import itertools

import numpy as np
from scipy import linalg, optimize, interpolate, special

from tracklib.models import rouse
from tracklib.util import parallel
from .p2 import MSD

################## Converting between msd, acf, vacf ##########################

def msdss2procacf(msd, ss_var):
    msd[0] = 0 # Otherwise the acf is just wrong
    return ss_var - 0.5*msd

def procacf2msdss(acf):
    return 2*(acf[0] - acf), acf[0]

def msd2stepacf(msd):
    msd[0] = 0 # Otherwise the acf is just wrong
    msd = np.insert(msd, 0, msd[1])
    return 0.5*(msd[2:] + msd[:-2] - 2*msd[1:-1])

def stepacf2msd(acf):
    first_sum = 2*np.insert(np.cumsum(acf, axis=0), 0, 0, axis=0) - acf[0]
    return np.cumsum(first_sum, axis=0) + acf[0]

################## Gaussian Process likelihoods ###############################

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)

class BadCovarianceError(RuntimeError):
    pass

def GP_logL_x(trace, C, m=0):
    """ m, C define the actual process """
    
    # Get valid data points only
    ind = ~np.isnan(trace)
    trace = trace[ind] - m
    
    # Marginalize covariance over missing frames
    # Note: numpy replaces missing entries in boolean index arrays
    # with False, so we don't have to fill them explicitly!
    # Note: numpy says it does this, but actually doesn't
    ind = np.nonzero(ind)[0]
    myC = C[ind, :][:, ind]
    
    # Prepare likelihood calculation
    with np.errstate(under='ignore'):
        s, logdet = np.linalg.slogdet(myC)
    if s <= 0:
        raise BadCovarianceError("Covariance not positive definite. slogdet = ({}, {})".format(s, logdet))
    try:
        # xCx = trace @ linalg.inv(myC) @ trace
        xCx = trace @ linalg.solve(myC, trace, assume_a='pos')
    except:
        raise BadCovarianceError("Inverting covariance did not work")

    return -0.5*(xCx + logdet) - len(myC)*LOG_SQRT_2_PI

def GP_logL_dx(trace, C, m=0):
    """ m, C define the displacement process """
    
    # Get valid data points only
    ind = ~np.isnan(trace)
    trace_clean = trace[ind]
    steps = np.diff(trace_clean) - m
    
    # Adjust covariance for missing frames etc.
    ind = np.nonzero(ind)[0]
    ind -= np.min(ind)
    N1 = np.max(ind) # number of single time steps
    B = np.zeros((len(ind)-1, N1))
    for i in range(len(ind)-1):
        B[i, ind[i]:ind[i+1]] = 1
    
    myC = B @ C[:N1, :N1] @ B.T
    
    # Prepare likelihood calculation
    with np.errstate(under='ignore'):
        s, logdet = np.linalg.slogdet(myC)
    if s <= 0:
        raise BadCovarianceError("Covariance not positive definite. slogdet = ({}, {})".format(s, logdet))
    try:
        # DCD = steps @ linalg.inv(myC) @ steps
        DCD = steps @ linalg.solve(myC, steps, assume_a='pos')
    except:
        raise BadCovarianceError("Inverting covariance did not work")
    
    return -0.5*(DCD + logdet) - len(myC)*LOG_SQRT_2_PI

def _p_logL(params):
    ss_order = params[0]
    if ss_order == 0:
        return GP_logL_x(*params[1:])
    elif ss_order == 1:
        return GP_logL_dx(*params[1:])
    else:
        raise ValueError(f"invalid steady state order: {ss_order}")

def ds_logL(data, ss_order, acf, m=0):
    """ ss_order = 0 if trajectories are steady state, 1 if displacements are steady state """
    d = data.map_unique(lambda traj : traj.d)
        
#     if ss_order == 0:
#         my_logL = GP_logL_x
#     elif ss_order == 1:
#         my_logL = GP_logL_dx
#     else:
#         raise ValueError(f"invalid steady state order: {ss_order}")
#     
#     logLs = np.empty((len(data), d), dtype=float)
#     logLs[:] = np.nan
#     for dim in range(d):
#         if len(acf.shape) == 1:
#             C = linalg.toeplitz(acf/d)
#         else:
#             C = linalg.toeplitz(acf[:, dim])
# 
#         for i, traj in enumerate(data):
#             logLs[i, dim] = my_logL(traj[:][:, dim], C)
# 
#     return np.sum(logLs)

    logL = 0
    for dim in range(d):
        if len(acf.shape) == 1:
            C = linalg.toeplitz(acf/d)
        else:
            C = linalg.toeplitz(acf[:, dim])

        iparams = itertools.product([ss_order], (traj[:][:, dim] for traj in data), [C])
        logL += np.sum(list(parallel._umap(_p_logL, iparams)))

    return logL

################## Fit base class definition ###################################

class Fit:
    def __init__(self, data):
        self.data = data
        self.data_selection = data.saveSelection()
        self.d = data.map_unique(lambda traj: traj.d)
        self.T = max(map(len, self.data))

        # Fit properties
        self.ss_order = 0
        self.bounds = None # List of (lb, ub), to be passed to optimize.minimize
        self.fix_values = [] # will be appended to the list passed to run()

        # Each constraint should be a callable constr(params) -> x. We will apply:
        #        x <= 0 : infeasible. Maximum penalization
        #   0 <= x <= 1 : smooth crossover: np.exp(1/tan(pi*x))
        #   0 <= x      : feasible. No penalization   
        self.constraints = [self.constraint_Cpositive]
        self.max_penalty = 1e10
        
    ### To be overwritten / used upon subclassing ###
        
    def params2acfm(self, params):
        """ Should return acf, m """
        raise NotImplementedError
        
    def initial_params(self):
        raise NotImplementedError

    def initial_offset(self):
        return 0
                
    def constraint_Cpositive(self, params):
        acf, _ = self.params2acfm(params)
        min_ev_okay = 1 - np.cos(np.pi/len(acf)) # white noise min ev
        min_ev = np.min(np.linalg.eigvalsh(linalg.toeplitz(acf)))
        return min_ev / min_ev_okay
    
    ### General machinery, usually won't need overwriting ###
    
    def params2msd(self, params):
        if self.ss_order == 0:
            return procacf2msdss(self.params2acfm(params)[0])
        else:
            return stepacf2msd(self.params2acfm(params)[0])
        
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
    
    def _get_min_target(self, n_params, offset=0, fix_values=()):
        # Set up fixed value machinery and make sure that everything is cast as
        # a function taking the given parameters and calculating the missing
        # one from it. Note note below.
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
        
        def min_target(params, return_full_params=False):
            myparams = np.empty(n_params, dtype=float)
            myparams[:] = np.nan
            myparams[~is_fixed] = params
            for ip, fixfun in fix_values:
                myparams[ip] = fixfun(myparams)

            if return_full_params:
                return myparams
            
            penalty = self._penalty(myparams)
            if penalty < 0: # infeasible
                return self.max_penalty
            else:
                acf, m = self.params2acfm(myparams)
                return -ds_logL(self.data, self.ss_order, acf, m) + penalty - offset
            
        return min_target

    def run(self,
            init_from = None, # dict as returned by this function
            optimization_steps=('simplex',),
            maxfev=None,
            fix_values = [], # list of 2-tuples (i, val) to fix params[i] = val
            full_output=False,
            show_progress=False, assume_notebook_for_progress_bar=True,
            print_on_error=True,
           ):
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
        n_params = len(p0)
        
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
                    
                min_target = self._get_min_target(n_params, total_offset, fix_values)
                fitres = optimize.minimize(min_target, p0, **kwargs)
                
                if not fitres.success:
                    if print_on_error:
                        print(fitres)
                    raise RuntimeError("Fit failed at step {:d}: {:s}".format(istep, step))
                else:
                    all_res.append(({'params' : min_target(fitres.x, return_full_params=True),
                                     'logL' : -(fitres.fun+total_offset),
                                    }, fitres))
                    p0 = fitres.x
                    total_offset += fitres.fun
                    
        bar.close()
        
        if full_output:
            return all_res
        else:
            return all_res[-1][0]

#     def profile_likelihood(self, init_from, iparam, values,
#                            fix_values=(),
#                            full_output=False,
#                            show_progress=False, assume_notebook_for_progress_bar=True,
#                            **run_kw,
#                           ):
#         if show_progress:
#             if assume_notebook_for_progress_bar:
#                 from tqdm.notebook import tqdm
#             else:
#                 from tqdm import tqdm
#             val_iter = tqdm(values)
#             del tqdm
#         else:
#             val_iter = iter(values)
# 
#         best_fits = []
#         for val in val_iter:
#             best_fits.append(self.run(init_from, fix_values=[(iparam, val)]+list(fix_values), **run_kw))
#             
#         logLs = np.array([best_fit['logL'] for best_fit in best_fits])
#         if full_output:
#             return logLs, best_fits
#         else:
#             return logLs

################## Definition of the various subclasses we provide here ########

class SplineFit(Fit):
    def __init__(self, data, ss_order, n,
                 previous_spline_fit_and_result=None,
                ):
        super().__init__(data)
        if n < 2:
            raise ValueError(f"SplineFit with n = {n} < 2 does not make sense")
        self.n = n
        
        self.ss_order = ss_order
        self.bounds = None # Will be done with constraints, since for x we have to ensure monotonicity
                           # and y applies to the whole spline, not just the knots (i.e. parameters)
        self.constraints = [self.constraint_dx,
                            self.constraint_dy,
                            self.constraint_logmsd,
                            self.constraint_Cpositive,
                           ]

        self.prev_fit = previous_spline_fit_and_result # for (alternative) initialization
        
        # Set up
        # Note that in both cases  x is compactified to [0, 1], but by
        # different functions, such that for ss_order = 0 we also have infinity
        # at x = 2.
        if self.ss_order == 0:
            # Fit in (4/π*arctan(log), log) space and add point at infinity,
            # i.e. x = 4/π*arctan(log(∞)) = 2
            self.x_full = np.log(np.arange(1, self.T))
            self.x_full = np.append((4/np.pi)*np.arctan(self.x_full / self.x_full[-1]), 2)
            self.bc_type = ('natural', (1, 0.0))
        elif self.ss_order == 1:
            # Simply fit in log space, with natural boundary conditions (i.e.
            # vanishing second derivative; so we extrapolate as power law).
            self.x_full = np.log(np.arange(1, self.T))
            self.x_full = self.x_full / self.x_full[-1]
            self.bc_type = 'natural'
        else:
            raise ValueError(f"Did not understand ss_order = {ss_order}")
            
    def _params2csp(self, params):
        x = np.array([self.x_full[0], *params[:(self.n-2)], self.x_full[-1]])
        y = params[(self.n-2):]
        return interpolate.CubicSpline(x, y, bc_type=self.bc_type)
        
    def params2acfm(self, params):
        csp = self._params2csp(params)
        if self.ss_order == 0:
            msd = np.insert(np.exp(csp(self.x_full[:-1])), 0, 0)
            ss_var = 0.5*np.exp(params[-1]) # γ(0) = 0.5*μ(∞)
            acf = msdss2procacf(msd, ss_var)
        elif self.ss_order == 1:
            msd = np.insert(np.exp(csp(self.x_full)), 0, 0)
            acf = msd2stepacf(msd)
        else:
            raise ValueError
        return acf, 0
            
    def initial_params(self):
        x_init = np.linspace(self.x_full[0], self.x_full[-1], self.n)

        # If we have a previous fit (e.g. when doing model selection), use that
        # for initialization
        if self.prev_fit is not None:
            fit, res = self.prev_fit
            y_init = fit._params2csp(res['params'])(x_init)
        else:
            # Fit linear (i.e. powerlaw), which is useful in both cases.
            # For ss_order == 0 we will use it as boundary condition,
            # for ss_order == 1 this will be the initial MSD
            e_msd = MSD(self.data)
            t_valid = np.nonzero(~np.isnan(e_msd[1:]))[0]
            (A, B), _ = optimize.curve_fit(lambda x, A, B : A*x + B,
                                           self.x_full[t_valid],
                                           np.log(e_msd[t_valid+1]), # gotta skip msd[0] = 0
                                           p0=(1, 0),
                                           bounds=([0, -np.inf], np.inf),
                                          )
                
            if self.ss_order == 0:
                # interpolate along 2-point spline
                ss_var = np.nanmean(np.concatenate([np.sum(traj[:]**2, axis=1) for traj in self.data]))
                csp = interpolate.CubicSpline(np.array([0, 2]),
                                              np.log(np.array([e_msd[1], 2*ss_var])),
                                              bc_type = ((1, A), (1, 0.)),
                                             )
                y_init = csp(x_init)
            elif self.ss_order == 1:
                y_init = A*x_init + B
            else:
                raise ValueError
            
        return np.array([*x_init[1:-1], *y_init])

    def initial_offset(self):
        if self.prev_fit is None:
            return 0
        else:
            return -self.prev_fit[1]['logL']
        
    def constraint_dx(self, params):
        min_step = 1e-7 # x is compactified to (0, 1)
        x = np.array([self.x_full[0], *params[:(self.n-2)], self.x_full[-1]])
        return np.min(np.diff(x))/min_step
    
    def constraint_dy(self, params):
        # Ensure monotonicity in the MSD. This makes sense intuitively, but is it technically a condition?
        min_step = 1e-7
        y = params[(self.n-2):]
        return np.min(np.diff(y))/min_step
    
    def constraint_logmsd(self, params):
        start_penalizing = 200
        full_penalty = 500
        
        csp = self._params2csp(params)
        return (full_penalty - np.max(np.abs(csp(self.x_full))))/start_penalizing

class NPXFit(Fit): # NPX = Noise + Powerlaw + X (i.e. spline)
    def __init__(self, data, ss_order, n=0):
        super().__init__(data)
        if n == 0 and ss_order == 0:
            raise ValueError("Incompatible assumptions: pure powerlaw (n=0) and trajectory steady state (ss_order=0)")
        self.n = n
        
        # Parameters are (noise2, α, log(Γ), x0, ..., x{n-1}, y1, .., yn)
        # If x == 0 we omit x0. So we always have 2*n spline parameters!
        self.ss_order = ss_order
        self.bounds = 2*[(1e-10, np.inf)] + [(-np.inf, np.inf)] + n*[(0, 2 if ss_order == 0 else 1)] + n*[(-np.inf, np.inf)]
        self.constraints = [self.constraint_dx,
                            self.constraint_dy,
                            self.constraint_logmsd,
                            self.constraint_Cpositive,
                           ]
        
        # Set up
        self.logt_full = np.log(np.arange(1, self.T))
        self.x_full = self.logt_full / self.logt_full[-1]
        if self.ss_order == 0:
            # Fit in 4/π*arctan(log) space and add point at infinity, i.e. x = 4/π*arctan(log(∞)) = 2
            self.x_full = np.append((4/np.pi)*np.arctan(self.x_full), 2)
            self.upper_bc_type = (1, 0.0)
        elif self.ss_order == 1:
            # Simply fit in log space, with natural boundary conditions
            self.upper_bc_type = 'natural'
        else:
            raise ValueError(f"Did not understand ss_order = {ss_order}")
        
    def _first_spline_point(self, x0, A, B):
        if self.ss_order == 0:
            logt0 = np.tan(np.pi/4*x0)*self.logt_full[-1]
            dcdx0 = np.pi/4*A*self.logt_full[-1]/np.cos(np.pi/4*x0)**2
        elif self.ss_order == 1:
            logt0 = x0*self.logt_full[-1]
            dcdx0 = A*self.logt_full[-1]
        else:
            raise ValueError
        y0 = A*logt0 + B
        return x0, logt0, y0, dcdx0
            
    def _params2logmsd(self, params):
        _, A, B = params[:3]
        if self.n == 0 or params[3] >= 1: # Note order of conditions. params[3] only exists if n > 0
            return A*self.logt_full + B
        
        # Set up first spline point
        x0, logt0, y0, dcdx0 = self._first_spline_point(params[3], A, B)
        i0 = np.nonzero(x0 <= self.x_full)[0][0] # exists, because we know that x0 < 1, see above
        
        # Get spline
        x = np.array([*params[3:(self.n+3)], self.x_full[-1]])
        y = np.array([y0, *params[(self.n+3):]])
        csp = interpolate.CubicSpline(x, y, bc_type=((1, dcdx0), self.upper_bc_type))
        
        # Put together MSD
        # for simplicity, we initialize as all powerlaw, then overwrite the top portion
        logmsd = A*self.logt_full + B
        logmsd[i0:] = csp(self.x_full[i0:(-1 if self.ss_order == 0 else None)])
        return logmsd
        
    def params2acfm(self, params):
        logmsd = self._params2logmsd(params)
        if self.ss_order == 0:
            msd = np.insert(np.exp(logmsd), 0, 0)
            ss_var = 0.5*np.exp(params[-1]) # γ(0) = 0.5*μ(∞)
            acf = msdss2procacf(msd + 2*params[0], ss_var + params[0])
        elif self.ss_order == 1:
            msd = np.insert(np.exp(logmsd), 0, 0)
            acf = msd2stepacf(msd + 2*params[0])
        else:
            raise ValueError
        return acf, 0
            
    def initial_params(self):
        # Fit linear (i.e. powerlaw), which is useful in both cases.
        # For ss_order == 0 we will use it as boundary condition,
        # for ss_order == 1 this will be the initial MSD
        e_msd = MSD(self.data)
        i_valid = np.nonzero(~np.isnan(e_msd[1:]))[0]
        (A, B), _ = optimize.curve_fit(lambda x, A, B : A*x + B,
                                       self.logt_full[i_valid],
                                       np.log(e_msd[i_valid+1]), # gotta skip msd[0] = 0
                                       p0=(1, 0),
                                       bounds=([0, -np.inf], np.inf),
                                      )
            
        x0, logt0, y0, dcdx0 = self._first_spline_point(0.5, A, B)
        x_init = np.linspace(x0, self.x_full[-1], self.n+1)
        if self.ss_order == 0:
            # interpolate along 2-point spline
            ss_var = np.nanmean(np.concatenate([np.sum(traj[:]**2, axis=1) for traj in self.data]))
            csp = interpolate.CubicSpline(np.array([x0, 2]),
                                          np.array([y0, np.log(2*ss_var)]),
                                          bc_type = ((1, dcdx0), (1, 0.)),
                                         )
            y_init = csp(x_init)
        elif self.ss_order == 1:
            y_init = A*self.logt_full[-1]*x_init + B
        else:
            raise ValueError
            
        return np.array([e_msd[1], A, B, *x_init[:-1], *y_init[1:]])
        
    def constraint_dx(self, params):
        if self.n == 0:
            return np.inf
        
        min_step = 1e-7 # x is compactified to (0, 1)
        x = np.array([*params[3:(self.n+3)], self.x_full[-1]])
        return np.min(np.diff(x))/min_step
    
    def constraint_dy(self, params):
        # Ensure monotonicity in the MSD. This makes sense intuitively, but is it technically a condition?
        if self.n == 0:
            return np.inf
        
        min_step = 1e-7
        _, _, y0, _ = self._first_spline_point(*params[[3, 1, 2]])
        y = np.array([y0, *params[(self.n+3):]])
        return np.min(np.diff(y))/min_step
    
    def constraint_logmsd(self, params):
        start_penalizing = 200
        full_penalty = 500
        
        logmsd = self._params2logmsd(params)
        return (full_penalty - np.max(np.abs(logmsd)))/start_penalizing

class RouseFit(Fit):
    def __init__(self, data, k=1):
        super().__init__(data)
        self.k = k
        
        # Fit properties
        # Parameters are (noise2, D, L)
        self.ss_order = 0
        self.bounds = 3*self.d*[(1e-10, np.inf)]
        self.constraints = [] # Don't need to check Cpositive, will always be true for Rouse MSDs

        self.fix_values  = [(3*dim+1, lambda x : x[1]) for dim in range(1, self.d)]
        self.fix_values += [(3*dim+2, lambda x : x[2]) for dim in range(1, self.d)]
        
    def params2acfm(self, params):
        acf = np.empty((self.T, self.d), dtype=float)
        acf[:] = np.nan
        dt = np.arange(1, self.T)
        for dim in range(self.d):
            noise2, D, L = params[(3*dim):(3*(dim+1))]
            J = self.d*D*L/self.k
            tau = np.pi*L**2 / (4*self.k)
            
            with np.errstate(under = 'ignore'):
                msd = 2*J*( np.sqrt(dt/tau)*(1-np.exp(-tau/(np.pi*dt))) + special.erfc(np.sqrt(tau/(np.pi*dt))))
            msd = np.insert(msd + 2*noise2, 0, 0)
            
            acf[:, dim] = msdss2procacf(msd, J+noise2)
        return acf, 0
        
    def initial_params(self):
        e_msd = MSD(self.data) / self.d
        J = np.nanmean(np.concatenate([traj[:]**2 for traj in self.data], axis=0))
        G = np.nanmean(e_msd[1:5]/np.sqrt(np.arange(1, 5)))
        tau = (2*J/G)**2

        L = np.sqrt(4*self.k*tau/np.pi)
        D = self.k*J / (self.d*L)

        return np.array(self.d*[e_msd[1]/2, D, L])

class RouseFitDiscrete(Fit):
    def __init__(self, data, w):
        super().__init__(data)
        self.w = w
        
        # Fit properties
        # Parameters are d*[noise2, D, k], where by default we fix Ds and ks to be equal
        self.ss_order = 0
        self.bounds = 3*self.d*[(1e-10, np.inf)] # List of (lb, ub), to be passed to optimize.minimize

# Note: the following loop does not work, because i is local to the list
# comprehension, thus the lambdas will ultimately all return the same value.
#         self.fix_values = [(3*dim+i, lambda x: x[i]) for dim in range(1, self.d) for i in [1, 2]]
        self.fix_values  = [(3*dim+1, lambda x : x[1]) for dim in range(1, self.d)]
        self.fix_values += [(3*dim+2, lambda x : x[2]) for dim in range(1, self.d)]

        self.constraints = [] # Don't need to check Cpositive, will always be true for Rouse MSDs
        
    def params2acfm(self, params):
        acf = np.empty((self.T, self.d), dtype=float)
        for dim in range(self.d):
            noise2, D, k = params[(3*dim):(3*(dim+1))]
            model = rouse.Model(len(self.w), D, k, d=1, setup_dynamics=False)
            acf[:, dim] = model.ss_ACF(np.arange(self.T), w=self.w)
            acf[0, dim] += noise2
        return acf, 0
    
    def initial_params(self):
        e_msd = MSD(self.data) / self.d
        J = np.nanmean(np.concatenate([traj[:]**2 for traj in self.data], axis=0))
        G = np.nanmean(e_msd[1:5]/np.sqrt(np.arange(1, 5)))
        L = np.diff(np.nonzero(self.w)[0])[0]
        
        D = np.pi*G**2*L/(16*J)
        k = np.pi*( G*L / (4*J) )**2

        return np.array(self.d*[e_msd[1]/2, D, k])
