from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import scipy.stats

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
#         'MCMC stepsize'      :   0.1,
        }
    return default_config.update(config) or default_config

class Sampler(ABC):
    """
    Abstract base class for MCMC sampling
    
    To implement a sampling scheme, subclass this and override
     - propose_update(current_values) # may use self.stepsize
         propose a step in parameter space. Should return the proposed
         parameter values, forward, and backward probabilities.
     - logL(params)
         calculate log-likelihood for a given set of parameters
     - (optional) callback_logging(current_values, best_values)
         will be called whenever a logging line is printed. Can be
         used to print additional info, plot stuff, etc.
    """
    default_stepsize = 0.1 # Default value for __init__, can be overridden in subclass
    
    def __init__(self, stepsize=None):
        """
        stepsize : float
            a generic stepsize parameter that can subsequently
            be used in propose_update(). The main use of this is
            to be able to make it adaptive in the base class.
        """
        self.stepsize = stepsize or Sampler.default_stepsize
    
    @abstractmethod
    def propose_update(self, current_values):
        """
        Propose an update
        
        Output
        ------
        proposed_values : array of proposed values
        p_forward : probability of proposing these values from the current ones
        p_backward : probability of proposing the current values from the proposed ones
        """
        pass
        
    @abstractmethod
    def logL(self, params):
        """
        Give the log-likelihood of the given set of parameters
        """
        pass
    
    def callback_logging(self, current_values, best_values):
        """
        Callback upon logging
        """
        pass
    
    def run(self, initial_values, config={}):
        """
        Run the sampling scheme
        
        Input
        -----
        config : dict
            'MCMC iterations'
            'MCMC burn-in'
            'MCMC best only'
            'MCMC log every'
            'MCMC show progress'
            # 'MCMC stepsize'
            See README.md for more details
            
        Output
        ------
        logL : (M,) array
            list of log-likelihood at each iteration
            EDIT: right now, returning full-length logL. We usually just use
            this to check convergence, so it doesn't make sense to cut off
            exactly that burn-in period.
        params : (M, ...) array
            list of the sampled parameter sets.

        where M = 'MCMC iterations' - 'MCMC burn-in'
        """
        # Input processing
        current_values = initial_values
        try:
            Nparams = len(current_values)
        except TypeError:
            Nparams = 1
            
        config = _fill_config(config)
        config['_logging'] = config['MCMC log every'] > 0
        
        # Setup
        cur_logL = -np.inf
        max_logL = -np.inf
        cnt_accept = 0
        if not config['MCMC best only']:
            logL = -np.inf*np.ones((config['MCMC iterations'],))
            params = []
        
        Mrange = range(config['MCMC iterations'])
        if config['MCMC show progress']:
            from tqdm.notebook import tqdm
            Mrange = tqdm(Mrange)
            del tqdm

        # Run
        for i in Mrange:
            # Proposal
            proposed_values, p_forward, p_backward = self.propose_update(current_values)
            new_logL = self.logL(proposed_values)
            p_accept = np.exp(new_logL - cur_logL - p_forward + p_backward)

            # Acceptance
            if np.random.rand() < p_accept:
                cnt_accept += 1
                current_values = proposed_values
                cur_logL = new_logL
            accept_rate = cnt_accept / (i+1)
            
            # Output
            if not config['MCMC best only']:
                params.append(current_values)
                logL[i] = cur_logL
            
            # Keeping track of best parameters
            if cur_logL > max_logL:
                max_logL = deepcopy(cur_logL)
                best_values = deepcopy(current_values)
            
            # Logging
            if config['_logging'] and ( (i+1) % config['MCMC log every'] == 0 or i+1 == config['MCMC iterations'] ):
                logstring = "iteration {}: acceptance = {:.0f}%, logL = {}".format(i+1, accept_rate*100, cur_logL)
                print(logstring)
                self.callback_logging(current_values, best_values)
                
        if config['MCMC best only']:
            return max_logL, best_values
        else:
            s = slice(config['MCMC burn-in'], None)
#             return logL[s], params[s]
            return logL, params[s]
        
# An example for sampling from a standard normal
class normMCMC(Sampler):
    def propose_update(self, current_value):
        proposed_value = current_value + np.random.normal(scale=self.stepsize)
        p_forward = scipy.stats.norm.pdf(proposed_value, loc=current_value, scale=self.stepsize)
        p_backward = scipy.stats.norm.pdf(current_value, loc=proposed_value, scale=self.stepsize)
        return proposed_value, p_forward, p_backward
        
    def logL(self, value):
        return -0.5*value**2
    
    def callback_logging(self, current_values, best_values):
        print("Current value: {}".format(current_value))
