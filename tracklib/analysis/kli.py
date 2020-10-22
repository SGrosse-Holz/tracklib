from copy import deepcopy

import numpy as np

import scipy.stats

from tracklib.util import mcmc
from tracklib.models import rouse

### Utility stuff ###

def _fill_config(config):
    """
    Fill a given config dict with default values where nothing else is provided

    Input
    -----
    config : dict
        some dict, possibly containing configuration fields

    Output
    ------
    config : dict
        a copy of the input, augmented with default values for missing fields
    """
    default_config = {
        'MCMC config' : mcmc._fill_config({}),
        'unknown params' : ['D', 'k'],
        'numIntervals' : 10,
        'pLoop_method' : 'sequence', # 'sequence' or 'trace'
        }
    return default_config.update(config) or default_config

class LoopSequence:
    """
    Represent a sequence of looping intervals
    
    Attributes:
    t : (n-1,) float-array
        the boundaries of a loop/no-loop region
        the region t[i]---t[i+1] contains all trajectory indices ind with
             t[i]-0.5 <= ind < t[i+1]-0.5
    isLoop : (n,) bool-array
        isLoop[i] tells us whether the interval ending at t[i] has a loop
    """
    def __init__(self, T, numInt):
        """ initialize equally spaced intervals, randomly """
        self.T = T
        self.t = np.linspace(0, T, numInt+1)[1:-1]
        self.isLoop = np.array([np.random.rand() > 0.5 for _ in range(numInt)])
        
    def toLooptrace(self):
        """ give a list of True/False for each time point """
        looptrace = np.array(self.T*[False])
        t1 = 0
        for t2, isLoop in zip(np.append(self.t, self.T), self.isLoop):
            looptrace[ np.ceil(t1-0.5).astype(int) : np.ceil(t2-0.5).astype(int) ] = isLoop
            t1 = t2
        
        return looptrace

    @classmethod
    def fromLooptrace(cls, looptrace):
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

    def numLoops(self):
        return np.sum(self.isLoop[:-1] * (1-self.isLoop[1:])) + self.isLoop[-1]
        
    def plottable(self):
        """ Return t, loop for plotting """
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
    Here we will use LoopSequence objects as parameters
    """
    def setup(self, traj, model, **kwargs):
        """
        Load the data for loop estimation

        Input
        -----
        traj : Trajectory
            the trajectory to run on
        model : tracklib.models.rouse.Model
            the model to use
        Remaining kwargs go to tracklib.models.rouse.likelihood()
        """
        if traj.N == 1:
            self.traj = traj # We already are given a distance trajectory
        elif traj.N == 2:
            self.traj = traj.relative()
        else:
            raise ValueError("Don't know what to do with trajectory with N = {}".format(traj.N))

        self.model = model
        self.logL_kwargs = kwargs
        
    def propose_update(self, current_sequence):
        proposed_sequence = deepcopy(current_sequence)
        p_forward = 0
        p_backward = 0
        
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
        return np.sum([
            rouse.likelihood(self.traj[:][:, i], sequence.toLooptrace(), self.model, **self.logL_kwargs) \
                    for i in range(self.traj.d)]) \
                + (sequence.numLoops() - 1)*np.log(1-0.3) # from Christoph's code
    
    def callback_logging(self, current_sequence, best_sequence):
        pass
    
class LoopTraceMCMC(LoopSequenceMCMC):
    """
    Run MCMC directly on the loop trace.
    """
    def propose_update(self, current_looptrace):
        proposed_looptrace = deepcopy(current_looptrace)
        ind_up = np.random.randint(len(proposed_looptrace))
        proposed_looptrace[ind_up] = not proposed_looptrace[ind_up]
        return proposed_looptrace, 0, 0
    
    def logL(self, looptrace):
        # Could also do this by reusing LoopSequenceMCMC's logL, but maybe
        # better to have separate control (for example over loop counting)
        return np.sum([
            rouse.likelihood(self.traj[:][:, i], looptrace, self.model, **self.logL_kwargs) \
                    for i in range(self.traj.d)])
