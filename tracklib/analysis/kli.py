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

def fit_RouseParams(data, model_init, unknown_params, **kwargs):
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
    """
    model = deepcopy(model_init)

    if isinstance(unknown_params, str):
        unknown_params = [unknown_params]
    elif not isinstance(unknown_params, list):
        raise ValueError("Did not understand type of 'unknown_params' : {} (should be str or list)".format(type(unknown_params)))
    init_params = [getattr(model, pname) for pname in unknown_params]

    def neg_logL_ds(params):
        for pname, pval in zip(unknown_params, params):
            setattr(model, pname, pval)
        model.check_setup_called(run_if_necessary=True)

        def neg_logL_traj(traj):
            return -traj_likelihood(traj, model)

        # Parallelization?
        return np.sum(list(map(neg_logL_traj, data)))

    minimize_kwargs = {
            'method' : 'L-BFGS-B',
            'bounds' : tuple((1e-10, None) for _ in range(len(init_params))), # all parameters should be positive
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
