"""
This module uses the Rouse model to infer looping probabilities from two-locus
trajectories. This scheme tries to infer the cohesin loop size. 
"""

from copy import deepcopy

import numpy as np

import scipy.stats

from tracklib.util import mcmc
from tracklib.models import rouse


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

### Utility stuff ###
def traj_likelihood(traj, model_list, looptrace=None):
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
        model_likelihood_filter(traj[:][:, i], model_list, looptrace, traj.meta['localization_error'][i]) \
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
    def __init__(self, T, numInt,numChainStates=2):
        self.T = T
        self.t = np.linspace(0, T, numInt+1)[1:-1]
        self.isLoop = np.random.randint(0,numChainStates,numInt) 
        
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
        looptrace = np.array(self.T*[0])
        t1 = 0
        for t2, isLoop in zip(np.append(self.t, self.T), self.isLoop):
            looptrace[ np.ceil(t1-0.5).astype(int) : np.ceil(t2-0.5).astype(int) ] = isLoop
            t1 = t2
        
        return looptrace

        
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
    def setup(self, traj, model_list):
        """
        Load the data for loop estimation

        Parameters
        ----------
        traj : Trajectory
            The trajectory to run on
        model_list : list of tracklib.models.rouse.Model
            Each model will represent a state of the connetivity matrix. 

        See also
        --------
        traj_likelihood, tracklib.models.rouse
        """
        self.traj = traj
        self.model_list = model_list
        self.num_states = len(model_list)

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

        proposed_sequence.t[ind_up] = t1 + prop_tau * (t2 - t1)
        
        # Update a loop interval 
        ind_up = np.random.randint(len(current_sequence.isLoop))
        if np.random.rand() < 0.5:
            proposed_sequence.isLoop[ind_up] = np.random.randint(self.num_states) ## MODIFY HERE TO MAKE DISCRETE NUMBER
            
        return proposed_sequence, p_forward, p_backward

    def logL(self, sequence):
        "" # Remove the default docstring from mcmc.Sampler; this is an internal function now

        return traj_likelihood(self.traj, self.model_list, sequence.toLooptrace()) #\
                #+ (sequence.numLoops() - 1)*np.log(1-0.3) # from Christoph's code
       
    
# We define the likelihood outside the Model class mainly for a conceptual
# reason: the trace, looptrace, and model enter the likelihood on equal
# footing, i.e. it is just as fair to talk about the likelihood of the model
# given the trace as vice versa. Or likelihood of looptrace given trace and
# model, etc.
def model_likelihood_filter(trace, model_list, looptrace, noise):
    """
    Likelihood calculation using Kalman filter.
    """
    T = len(trace)
    try:
        w = model_list[looptrace[0]].measurement
    except AttributeError:
        w = np.zeros((model_list[looptrace[0]].N,))
        w[ 0] =  1
        w[-1] = -1

    for model in model_list:
        model.check_setup_called()
        
    dt = model_list[looptrace[0]]._propagation_memo['dt']

    M0, C0 = model_list[looptrace[0]].steady_state(looptrace[0])

    logL = np.empty((T,), dtype=float)
    logL[:] = np.nan
    for i, state in enumerate(looptrace):
        model = model_list[state]
        if state > 0:
            
            M1, C1 = model.propagate(M0, C0, dt, 1) # bond chooses from models, and "bond on==True"
        else:
            M1, C1 = model.propagate(M0, C0, dt, 1) # bond chooses from models, and "bond on==True"
            
        if np.isnan(trace[i]):
            M0 = M1
            C0 = C1
            continue
        
        # Update step copied from Christoph
        if noise > 0:
            InvSigmaPrior = scipy.linalg.inv(C1)
            InvSigma = np.tensordot(w, w, 0)/noise**2 + InvSigmaPrior
            SigmaPosterior = scipy.linalg.inv(InvSigma)
            MuPosterior = SigmaPosterior @ (w*trace[i] / noise**2 + InvSigmaPrior @ M1)
        else:
            SigmaPosterior = 0*C1
            MuPosterior = scipy.linalg.inv(np.tensordot(w, w, 0)) @ w*trace[i]
        
        # Same for likelihood calculation
        m = w @ M1
        s = C1[0, 0] + C1[-1, -1] - 2*C1[0, -1]
        if s < 0:
            raise RuntimeError("Prediction covariance negative: {}\nModel: {}".format(s, model))
#         logL[i] = np.log(scipy.stats.norm.pdf(trace[i], m, np.sqrt(s + noise**2)))
        logL[i] = -0.5*(trace[i] - m)**2 / (s+noise**2) - 0.5*np.log(2*np.pi*(s+noise**2))
        
        M0 = MuPosterior
        C0 = SigmaPosterior
        
    return np.nansum(logL)

