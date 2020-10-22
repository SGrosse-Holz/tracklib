# This module contains children of the mcmc.Sampler class that are responsible
# for running various sampling schemes.

from copy import deepcopy

import numpy as np
import scipy.stats
import scipy.optimize

from . import mcmc
from . import rouse
from . import util
    
class LoopSequenceMCMC(mcmc.Sampler):
    """
    Here we will use LoopSequence objects as parameters
    """
    default_stepsize = 0.1
    
    def setup(self, trace, model, config={}, **kwargs):
        """
        Load the data for loop estimation

        Input
        -----
        trace : (T,) or (T, d) array
            the trace(s) to fit to
        model : rouse.Model
            the model to use
        config : dict
            <None>
            See README.md for more details
        Remaining kwargs go to rouse.likelihood()
        """
        # Note: self.traces should be seen as list of traces.
        if len(trace.shape) == 1:
            self.traces = np.expand_dims(trace, 0)
        elif len(trace.shape) == 2:
            self.traces = trace.T
        else:
            raise ValueError("'trace' should be 1- or 2-dimensional")

        self.model = model
        self.config = util._fill_config(config)
        self.config['_logL_kwargs'] = kwargs
        
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
            t2 = self.traces.shape[1]
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
            rouse.likelihood(trace, sequence.toLooptrace(), self.model, **self.config['_logL_kwargs']) \
                    for trace in self.traces]) \
                + (sequence.numLoops() - 1)*np.log(1-0.3) # from Christoph's code
    
    def callback_logging(self, current_sequence, best_sequence):
        pass
    
class LoopTraceMCMC(mcmc.Sampler):
    """
    Run MCMC directly over the loop trace
    """
    def setup(self, trace, model, config={}, **kwargs):
        """
        Input
        -----
        trace : (T,) or (T, d) array
            the trace(s) to fit to
        model : rouse.Model
            the model to use
        config : dict
            <None>
            See README.md for more details
        Remaining kwargs go to rouse.likelihood()
        """
        # Note: self.traces should be seen as list of traces.
        if len(trace.shape) == 1:
            self.traces = np.expand_dims(trace, 0)
        elif len(trace.shape) == 2:
            self.traces = trace.T
        else:
            raise ValueError("'trace' should be 1- or 2-dimensional")

        self.model = model
        self.config = util._fill_config(config)
        self.config['_logL_kwargs'] = kwargs
        
    def propose_update(self, current_looptrace):
        proposed_looptrace = deepcopy(current_looptrace)
        ind_up = np.random.randint(len(proposed_looptrace))
        proposed_looptrace[ind_up] = not proposed_looptrace[ind_up]
        return proposed_looptrace, 0, 0
    
    def logL(self, looptrace):
        return np.sum([
            rouse.likelihood(trace, looptrace, self.model, **self.config['_logL_kwargs']) \
                    for trace in self.traces])    

    def callback_logging(self, current_looptrace, best_looptrace):
        pass

# def fit_RouseParams(traces, looptraces, model_init, unknown_params):
#     """
#     Run gradient descent on trace(s) to find best fit for Rouse parameters
# 
#     Input
#     -----
#     traces : (T,) or (N, T) array
        


### DEPRECATED ###

# def logpdf_lognormal(x, mu, sigma):
#     return -(np.log(x) - mu)**2/(2*sigma**2) - np.log(sigma*x)
# 
# class RouseParamsMCMC(mcmc.Sampler):
#     """
#     We will use rouse.Model as the parameter structure
# 
#     NOTE: using gradient descent actually works way better
#     """
#     default_stepsize = 0.1
#     
#     def setup(self, trace, looptrace=None, config={}, **kwargs):
#         """
#         Load the data we need to run the parameter estimation
#         
#         Input
#         -----
#         trace : (T,) array
#             the trace we use for estimation
#         looptrace : the corresponding ground truth sequence of loop/no loop
#             default: T*[False]
#         noise : float
#             localization error on the trace
#             default: 0
#         config : dict
#             'unknown params'
#             See README.md for more details.
#         remaining kwargs will be used for rouse.likelihood().
#         """
#         self.trace = trace
#         self.looptrace = looptrace
#         if self.looptrace is None:
#             self.looptrace = np.array(len(self.trace)*[False])
# 
#         self.config = util._fill_config(config)
#         self.config['_logL_kwargs'] = kwargs
#             
#     def propose_update(self, current_model):
#         proposed_model = deepcopy(current_model)
#         p_forward = 0
#         p_backward = 0
#         for attr in self.config['unknown params']:
#             current_value = getattr(current_model, attr)
#             proposed_value = np.random.lognormal(np.log(current_value), self.stepsize)
#             setattr(proposed_model, attr, proposed_value)
# #             p_forward += logpdf_lognormal(proposed_value, current_value, self.stepsize)
# #             p_backward += logpdf_lognormal(current_value, proposed_value, self.stepsize)
#         
#         return proposed_model, p_forward, p_backward
#     
#     def logL(self, model):
#         return rouse.likelihood(self.trace, self.looptrace, model, **self.config['_logL_kwargs'])
#     
#     def callback_logging(self, current_model, best_model):
#         pass
