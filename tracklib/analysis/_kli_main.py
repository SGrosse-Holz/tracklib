# Toplevel interface for the submodule

import numpy as np

from . import samplers
from . import util
    
def estimate_Rouse_params(init_model, trace, looptrace=None, *, noise, config={}):
    """
    Run the MCMC parameter estimation on an individual trace

    Input
    -----
    init_model : rouse.Model
        the initial guess for the parameters
    trace : (T,) np.array
        the trace to fit to
    looptrace : (T,) np.array(dtype=bool)
        the ground truth loop trace for the given trace
        default: T*[False], i.e. no loops
    noise : float
        the localization error in the trace
    config : dict
        'MCMC ...', 'unknown params'
        See README.md for more details.

    Output
    ------
    Lists of length 'MCMC iterations' - 'MCMC burn-in', in this order
     - log-likelihood
     - first parameter in 'unknown params'
     - ...
     - last parameter in 'unknown params'

    Notes
    -----
    #parameters is O(1), do we really have to run MCMC here? Grid search might
    be easier
    Similar to the other estimate_ functions, this is mainly just a wrapper
    around samplers.RouseParamsMCMC. For more flexibility, use that directly.
    """
    config = util._fill_config(config)

    mc = samplers.RouseParamsMCMC(stepsize=0.1)
    mc.setup(trace, looptrace, noise=noise, config=config)
    logL, models = mc.run(init_model, config=config)

    vals_list = [np.array([getattr(model, param) for model in models]) for param in config['unknown params']]
    return (logL, *vals_list)

def estimate_pLoop(model, trace, *, noise, config={}):
    """
    Estimate the looping probability on the given trace

    Input
    -----
    model : rouse.Model
        the model to use
    trace : (T,) or (T, d) np.array
        the trace(s) to fit to
    noise : float
        the localization error on the trace
    config : dict
        'MCMC ...'
        'numIntervals'
        'pLoop_method'
        See README.md for more details

    Output
    ------
    logL : list of the log-likelihoods at each MCMC step
    pLoop : (T,) array with values in (0, 1)

    Notes
    -----
    Similar to the other estimate_ functions, this is mainly just a wrapper
    around samplers.LoopSequenceMCMC. For more flexibility, use that directly.
    """
    config = util._fill_config(config)

    if config['pLoop_method'] == 'sequence':
        mc = samplers.LoopSequenceMCMC(stepsize=config['MCMC stepsize'])
        mc.setup(trace, model, noise=noise, config=config)
        logL, loopSequences = mc.run(util.LoopSequence(len(trace), config['numIntervals']), config=config)

        return logL, np.mean([seq.toLooptrace().astype(float) for seq in loopSequences], axis=0)
    elif config['pLoop_method'] == 'trace':
        mc = samplers.LoopTraceMCMC(stepsize=config['MCMC stepsize'])
        mc.setup(trace, model, noise=noise, config=config)
        logL, looptraces = mc.run(np.array([np.random.rand() > 0.5 for _ in range(len(trace))]), config=config)

        return logL, np.mean(looptraces, axis=0)
    else:
        raise ValueError("'{}' not a valid option for 'pLoop_method'".format(config['pLoop_method']))
