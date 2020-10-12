from copy import deepcopy

import numpy as np

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
        'MCMC iterations'    :   100,
        'MCMC burn-in'       :    50,
        'MCMC log every'     :    -1,
        'MCMC best only'     : False,
        'MCMC show progress' : False,
        'MCMC stepsize'      :   0.1,
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
        for i, loop in enumerate(looptrace)[1:]:
            if loop != isLoop[-1]:
                isLoop.append(loop)
                t.append(i)

        obj = cls(len(looptrace), len(isLoop))
        obj.isLoop = isLoop
        obj.t = t
        return obj

    def numLoops(self):
        return np.sum(self.isLoop[:-1] * (1-self.isLoop[1:])) + self.isLoop[-1]
        
    def plottable(self):
        """ Return t, loop for plotting """
        t = np.array([0] + [time for time in self.t for _ in range(2)] + [self.T]) - 0.5
        loop = np.array([float(isLoop) for isLoop in self.isLoop for _ in range(2)])
        return t, loop
