from copy import deepcopy

import numpy as np

import scipy.stats

from tracklib.util import mcmc
from tracklib.models import rouse

### Utility stuff ###

def traj_likelihood(traj, looptrace, model, **kwargs):
    """
    Apply `rouse.likelihood` to a `Trajectory`

    Keyword arguments will be forwarded to `tracklib.models.rouse.likelihood`

    Parameters
    ----------
    traj : Trajectory
        the trajectory to use
    looptrace : (traj.T,) np.ndarray, dtype=bool
        the looptrace to use
    model : tracklib.models.rouse.Model
        the model to use

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
    try:
        traj = traj.relative()
    except NotImplementedError:
        pass

    return np.sum([
        rouse.likelihood(traj[:][:, i], looptrace, model, **kwargs) \
                for i in range(traj.d)])

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

### Fitting Rouse parameters to a dataset ###

# def fit_RouseParams(traces, looptraces, model_init, unknown_params):
#     """
#     Run gradient descent on trace(s) to find best fit for Rouse parameters
# 
#     Input
#     -----
#     traces : (T,) or (N, T) array

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
    def setup(self, traj, model, **kwargs):
        """
        Load the data for loop estimation

        Keyword arguments will be forwared to
        `tracklib.models.rouse.likelihood`

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
        self.logL_kwargs = kwargs

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

        return traj_likelihood(self.traj, sequence.toLooptrace(), self.model, **self.logL_kwargs) \
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
        return traj_likelihood(self.traj, looptrace, self.model, **self.logL_kwargs)
