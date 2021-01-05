"""
This module uses the Rouse model to infer looping probabilities from two-locus
trajectories. Most of the conceptual work is due to Christoph Zechner.
"""

from copy import deepcopy

import numpy as np

import scipy.stats

from tracklib.util import mcmc
from tracklib.models import rouse

### Utility stuff ###

def traj_likelihood(traj, model, looptrace=None):
    """
    Apply `rouse.likelihood` to a `Trajectory`

    Parameters
    ----------
    traj : Trajectory
        the trajectory to use. Note that this should have
        ``meta['localization_error']`` set.
    model : tracklib.models.rouse.Model
        the model to use
    looptrace : (traj.T,) np.ndarray, dtype=bool
        the looptrace to use. Can be used to override
        ``traj.meta['looptrace']``, which would be used by default (i.e. if
        ``looptrace is None``). If neither is present, we assume no looping.

    Returns
    -------
    float
        the log-likelihood (simply the sum over the spatial dimensions)

    See also
    --------
    tracklib.models.rouse

    Notes
    -----
    If ``traj.N == 2``, this will evaluate the likelihood for
    ``traj.relative()``.
    """
    if traj.N == 2:
        traj = traj.relative()

    if looptrace is None:
        try:
            looptrace = traj.meta['looptrace']
        except KeyError:
            looptrace = len(traj)*[False]

    return np.sum([
        rouse.likelihood(traj[:][:, i], model, looptrace, traj.meta['localization_error'][i]) \
                for i in range(traj.d)])

### Fitting Rouse parameters to a dataset ###

def fit_RouseParams(data, model_init, unknown_params, fit_in_logspace=False, **kwargs):
    """
    Run gradient descent to find best fit for Rouse parameters

    All keyword arguments will be forwarded to `!scipy.minimize.optimize`.

    Parameters
    ----------
    data : `TaggedSet` of `Trajectory`
        the data to run on. If the ground truth looptrace for any trajectory
        `!traj` is known, it should be in ``traj.meta['looptrace']``. If that
        field is absent, we will assume no looping. Furthermore, this method
        requires the metadata field ``'localization_error'`` for each
        trajectory.
    model_init : `rouse.Model`
        the model to use for fitting. Fixes the parameters that are not fit,
        and provides initial values for those that will be fit.
    unknown_params : str or list of str
        the parameters of `!model_init` that should be fit. Should contain
        names of attributes of a `rouse.Model`. Examples: ``['D', 'k']``,
        ``['D', 'k', 'k_extra']``, ``'k_extra'``.
    fit_in_logspace : bool, str or list of str
        whether / which parameters to fit in log-space.

    Returns
    -------
    res : fitresult
        the structure returned by `!scipy.optimize.minimize`. The best fit
        parameters are ``res.x``, while their covariance matrix can be obtained
        as ``res.hess_inv.todense()``.

    See also
    --------
    traj_likelihood

    Notes
    -----
    Default parameters for minimization: we use ``method = 'L-BFGS-B'``,
    constrain all parameters to be positive (>1e-10) using the ``bounds``
    keyword and set the options ``maxfun = 300``, and ``ftol = 1e-5``.

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
    """
    model = deepcopy(model_init)

    # Unknown parameters
    if isinstance(unknown_params, str):
        unknown_params = [unknown_params]
    elif not isinstance(unknown_params, list):
        raise ValueError("Did not understand type of 'unknown_params' : {} (should be str or list)".format(type(unknown_params)))

    init_params = [getattr(model, pname) for pname in unknown_params]

    # Logspace fitting
    if isinstance(fit_in_logspace, str):
        fit_in_logspace = [fit_in_logspace]
    elif isinstance(fit_in_logspace, bool):
        if fit_in_logspace:
            fit_in_logspace = unknown_params
        else:
            fit_in_logspace = []
    elif not isinstance(fit_in_logspace, str):
        raise ValueError("Did not understand type of 'fit_in_logspace' : {} (should be str, list, or bool)".format(type(fit_in_logspace)))
    
    for i, pname in enumerate(unknown_params):
        if pname in fit_in_logspace:
            init_params[i] = np.log(init_params[i])
            unknown_params[i] = '_log_' + pname

    # Likelihood calculation
    def neg_logL_ds(params):
        for pname, pval in zip(unknown_params, params):
            if pname.startswith('_log_'):
                pname = pname[5:]
                pval = np.exp(pval)
            setattr(model, pname, pval)
        model.check_setup_called(run_if_necessary=True)

        def neg_logL_traj(traj):
            return -traj_likelihood(traj, model)

        # Parallelization?
        return np.nansum(list(map(neg_logL_traj, data)))

    minimize_kwargs = {
            'method' : 'L-BFGS-B',
            'bounds' : tuple((-50, 50) if pname.startswith('_log_') else (1e-10, None) \
                             for pname in unknown_params), # all parameters should be positive
            'options' : {'maxfun' : 300, 'ftol' : 1e-5},
            }
    # 'People' might try to override the defaults individually
    for key in ['maxfun', 'ftol']:
        if key in kwargs.keys():
            minimize_kwargs['options'][key] = kwargs[key]
            del kwargs[key]

    minimize_kwargs.update(kwargs)
    return scipy.optimize.minimize(neg_logL_ds, init_params, **minimize_kwargs)

class LoopSequence:
    """
    Represents a sequence of looping intervals

    The constructor initializes this as a sequence of intervals of equal
    length, with the looping indicator for each one being chosen randomly.
    
    Attributes
    ----------
    t : (n-1,) np.ndarray
        the boundaries of a loop/no-loop region the region ``t[i]---t[i+1]``
        contains all trajectory indices `!ind` with ``t[i]-0.5 <= ind <
        t[i+1]-0.5``
    isLoop : (n,) np.ndarray, dtype=bool
        ``isLoop[i]`` tells us whether the interval ending at ``t[i]`` has a
        loop

    Parameters
    ----------
    T : int
        maximum value for the `t` axis
    numInt : int
        the number of looping intervals to initialize with
    """
    def __init__(self, T, numInt):
        self.T = T
        self.t = np.linspace(0, T, numInt+1)[1:-1]
        self.isLoop = np.array([np.random.rand() > 0.5 for _ in range(numInt)])
        
    @classmethod
    def fromLooptrace(cls, looptrace):
        """
        Initialize a new `LoopSequence` from a looptrace.

        Parameters
        ----------
        looptrace : (T,) np.ndarray
            the looptrace to convert

        Returns
        -------
        LoopSequence
        """
        isLoop = [looptrace[0]]
        t = []
        for i, loop in enumerate(looptrace):
            if loop != isLoop[-1]:
                isLoop.append(loop)
                t.append(i)

        obj = cls(len(looptrace), len(isLoop))
        obj.isLoop = np.array(isLoop)
        obj.t = np.array(t)
        return obj

    def toLooptrace(self):
        """
        Convert to a looptrace.

        Returns
        -------
        (self.T,) np.ndarry, dtype=bool
        """
        looptrace = np.array(self.T*[False])
        t1 = 0
        for t2, isLoop in zip(np.append(self.t, self.T), self.isLoop):
            looptrace[ np.ceil(t1-0.5).astype(int) : np.ceil(t2-0.5).astype(int) ] = isLoop
            t1 = t2
        
        return looptrace

    def numLoops(self):
        """
        Count the number of loops in the sequence

        This counts adjacent intervals with ``self.isLoop[i] == True`` as a
        single loop. An efficient way to count this is to count loop breakings,
        i.e. where ``isLoop[i-1] == True and isLoop[i] == False``.

        Returns
        -------
        int
            the number of loops
        """
        return np.sum(self.isLoop[:-1] * (1-self.isLoop[1:])) + self.isLoop[-1]

    def plottable(self):
        """
        Give arrays that can be used for plotting

        Returns
        -------
        t : np.ndarray
            the time points where the looping status changes, doubled
        loop : np.ndarray
            the looping status at the corresponding time point

        Example
        -------
        For a `LoopSequence` ``seq``, use this as

        >>> from matplotlib import pyplot as plt
        ... plt.plot(*seq.plottable())
        """
        t = np.array([0] + [time for time in self.t for _ in range(2)] + [self.T]) - 0.5
        loop = np.array([float(isLoop) for isLoop in self.isLoop for _ in range(2)])
        return t, loop

### Running the inference on a trajectory ###

def estimate_looping(traj, model, numIntervals, MCMCconfig):
    """
    Run the looping inference on a single trajectory.

    Results are written to corresponding `!traj.meta` fields and the whole
    trajectory object is returned. This way this function can be run in
    parallel on a whole data set.

    Parameters
    ----------
    traj : Trajectory
        the trajectory to run the inference on
    model : tracklib.models.rouse.Model
        the calibrated model to use in the inference
    numIntervals : int
        the number of looping/no looping intervals to assume
    MCMCconfig : dict
        configuration of the MCMC sampler. See `LoopSequenceMCMC.configure`.

    Returns
    -------
    Trajectory
        the input trajectory, with the inference results written to
        ``traj.meta['logL']`` (log-likelihood during the MCMC runs),
        ``traj.meta['pLoop']`` (inferred looping probability), and
        ``traj.meta['loopSequences']`` (the individual loopSequences at each
        MCMC iteration after burn-in).

    See also
    --------
    fit_RouseParams, LoopSequenceMCMC, LoopSequenceMCMC.configure

    Examples
    --------
    Parallelization using `!multiprocessing.Pool`, assuming we have a
    `!TaggedSet` of `!Trajectory` as the input ``data``. Note that for
    parallelization the data will be copied anyways. `!tracklib` is assumed to
    be imported as ``tl``

    >>> from multiprocessing import Pool
    ... from functools import partial
    ...
    ... # Dummy parameters
    ... model = tl.models.Rouse(20, 1, 5, 1)
    ... numIntervals = 10
    ... MCMCconfig = {
    ...         'iterations' : 1000,
    ...         'burn_in' : 500,
    ...     }
    ...
    ... with Pool(20) as mypool:
    ...     data_estimated = tl.TaggedSet(
    ...         mypool.imap_unordered(
    ...             partial(tl.analysis.kli.estimate_looping, model=model, numIntervals=numIntervals, MCMCconfig=MCMCconfig),
    ...             data,
    ...         ),
    ...         hasTags=False,
    ...     )

    Note that you can incorporate a progressbar (using e.g. `!tqdm`) by
    wrapping the generator ``mypool.imap_unordered(...)`` in the above
    expression.
    """
    mc = LoopSequenceMCMC()
    mc.setup(traj, model)
    mc.configure(**MCMCconfig)

    traj.meta['logL'], traj.meta['loopSequences'] = mc.run(LoopSequence(len(traj), numIntervals))
    traj.meta['pLoop'] = np.mean([seq.toLooptrace() for seq in traj.meta['loopSequences']], axis=0)

    return traj

def callLoops(pLoop, threshold=0.5):
    """
    Calculate loop lifetimes

    Parameters
    ----------
    pLoop : array-like
        the input trace. A loop is defined as any point where this is greater
        than `!threshold`.
    threshold : float
        the threshold to use for loop calling

    Returns
    -------
    np.ndarray(dtype=int)
        the start and end points of detected loops, as (N_loop, 2) array. Note
        that the start index might be -1, if the trace starts looped. If
        ``len(pLoop)`` is listed as an end point, the trajectory ended looped.

    Notes
    -----
    This problem is called "run length encoding".
    """
    looptrace = np.pad(np.asarray(pLoop) > threshold, (1, 1), constant_values=False)
    indicator = np.diff(looptrace.astype(int))
    starts = np.where(indicator == 1)[0] - 1 # indices shift by 1 bc of padding
    ends = np.where(indicator == -1)[0] - 1
    return np.array([starts, ends]).T

def looplifes(traj, step=1):
    """
    Get all the loop lifetimes from the individual MCMC steps

    Parameters
    ----------
    traj : Trajectory
        should be processed through `estimate_loops`, such that it has the
        ``'loopSequences'`` meta-field.
    step : int, optional
        how big steps to take through the ``traj.meta['loopSequences']`` list.
        Since successive MCMC steps are correlated, using every single one
        gives mostly redundant data.
    """
    lifelist = []
    for seq in traj.meta['loopSequences']:
        loops = callLoops(seq.toLooptrace().astype(float))
        lifelist += (loops[:, 1] - loops[:, 0]).tolist()

    return lifelist

### The MCMC samplers ###

class LoopSequenceMCMC(mcmc.Sampler):
    """
    An MCMC sampler that uses `LoopSequence` objects as parametrization.

    See also
    --------
    setup, traj_likelihood, tracklib.util.mcmc, tracklib.models.rouse

    Examples
    --------
    Assuming we have a `Trajectory` ``traj`` and a `rouse.Model` ``model``:

    >>> MCMCconfig = {
    ...        'stepsize' : 0.1,
    ...        'iterations' : 100,
    ...        'burn_in' : 50,
    ...        'log_every' : 10,
    ...        'show_progress' : True,
    ...        }
    ... mc = LoopSequenceMCMC()
    ... mc.setup(traj, model, noise=1)
    ... mc.configure(**MCMCconfig)
    ... logL, sequneces = mc.run(LoopSequence(len(traj), 10))

    """
    def setup(self, traj, model):
        """
        Load the data for loop estimation

        Parameters
        ----------
        traj : Trajectory
            the trajectory to run on
        model : tracklib.models.rouse.Model
            the model to use

        See also
        --------
        traj_likelihood, tracklib.models.rouse
        """
        self.traj = traj
        self.model = model

    # Get documentation for these
    # Does this hack have drawbacks?
    configure = mcmc.Sampler.configure
    run = mcmc.Sampler.run
    run.__doc__ = run.__doc__.replace("<user-specified parameter structure>", "LoopSequence")
        
    def propose_update(self, current_sequence):
        "" # Remove the default docstring from mcmc.Sampler; this is an internal function now

        proposed_sequence = deepcopy(current_sequence)
        p_forward = 0.
        p_backward = 0.
        
        # Update a boundary
        ind_up = np.random.randint(len(current_sequence.t))
        if ind_up > 0:
            t1 = current_sequence.t[ind_up-1]
        else:
            t1 = 0
        if ind_up+1 < len(current_sequence.t):
            t2 = current_sequence.t[ind_up+1]
        else:
            t2 = len(self.traj)
        tmid = current_sequence.t[ind_up]
        
        cur_tau = (t2 - tmid) / (t2 - t1)
# Truncated normal is maxent for given mean & variance on fixed interval
        a, b = (0 - cur_tau) / self.stepsize, (1 - cur_tau) / self.stepsize
        prop_tau = scipy.stats.truncnorm.rvs(a, b) * self.stepsize + cur_tau
        p_forward += np.log(scipy.stats.truncnorm.pdf((prop_tau - cur_tau)/self.stepsize, a, b)/self.stepsize)
        a, b = (0 - prop_tau) / self.stepsize, (1 - prop_tau) / self.stepsize
        p_backward += np.log(scipy.stats.truncnorm.pdf((cur_tau - prop_tau)/self.stepsize, a, b)/self.stepsize)
# Beta sucks, because it gets sticky when two values come close
#         betaParam = self.config['betaParam']
#         prop_tau = scipy.stats.beta.rvs(betaParam*cur_tau / (1-cur_tau), betaParam)
#         p_forward += np.log(scipy.stats.beta.pdf(prop_tau, betaParam*cur_tau / (1-cur_tau), betaParam))
#         p_backward += np.log(scipy.stats.beta.pdf(cur_tau, betaParam*prop_tau / (1-prop_tau), betaParam))
        
        proposed_sequence.t[ind_up] = t1 + prop_tau * (t2 - t1)
        
        # Update a loop interval
        ind_up = np.random.randint(len(current_sequence.isLoop))
        if np.random.rand() < 0.5:
            proposed_sequence.isLoop[ind_up] = not proposed_sequence.isLoop[ind_up]
            
        return proposed_sequence, p_forward, p_backward

    def logL(self, sequence):
        "" # Remove the default docstring from mcmc.Sampler; this is an internal function now

        return traj_likelihood(self.traj, self.model, sequence.toLooptrace()) \
                + (sequence.numLoops() - 1)*np.log(1-0.3) # from Christoph's code
    
class LoopTraceMCMC(LoopSequenceMCMC):
    """
    Run MCMC directly on the loop trace.

    See `LoopSequenceMCMC` for usage (just that now of course the
    `!initial_values` argument to `run <tracklib.util.mcmc.Sampler.run>` should
    be a looptrace instead of a `LoopSequence`.

    See also
    --------
    LoopSequenceMCMC, tracklib.util.mcmc.Sampler.run
    """
    def propose_update(self, current_looptrace):
        proposed_looptrace = deepcopy(current_looptrace)
        ind_up = np.random.randint(len(proposed_looptrace))
        proposed_looptrace[ind_up] = not proposed_looptrace[ind_up]
        return proposed_looptrace, 0., 0.
    
    def logL(self, looptrace):
        # Could also do this by reusing LoopSequenceMCMC's logL, but maybe
        # better to have separate control (for example over loop counting)
        return traj_likelihood(self.traj, self.model, looptrace)
