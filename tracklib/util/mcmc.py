"""
A small module for running MCMC sampling. It provides the abstract base class
`Sampler` that can be subclassed for concrete sampling schemes.
"""

from copy import deepcopy
from abc import ABC, abstractmethod

from tqdm.auto import tqdm

import numpy as np

class MCMCRun:
    """
    The results of an MCMC run

    Attributes
    ----------
    logLs : np.array(dtype=float)
        the likelihoods from the run
    samples : list of <sample data type>)
        the actual MCMC sample

    Notes
    -----
    The list of samples is not automatically copied upon initialization.
    """
    # Implementation note: `samples` should be a list, such that we can take
    # advantage of mutable objects not being copied.
    def __init__(self, logLs=None, samples=None):
        if logLs is None:
            logLs = np.array([])
        self.logLs = np.asarray(logLs)

        if samples is None:
            samples = []
        self.samples = samples

    def logLs_trunc(self):
        """
        Give only the likelihoods associated with the samples

        `Sampler.run` returns likelihoods starting at the first iteration, but
        samples only after a given burn-in period. This function cuts this
        initial overhang from the likelihood array

        Returns
        -------
        (len(samples),) np.ndarray
        """
        return self.logLs[-len(self.samples):]

    def best_sample_logL(self):
        """
        Give the best sample (and log-likelihood)

        Returns
        -------
        sample : <sample data type>
            the maximum likelihood estimate
        logL : float
            the maximum likelihood value
        """
        logLs = self.logLs_trunc()
        i_best = np.argmax(logLs)
        return self.samples[i_best], logLs[i_best]

    def acceptance_rate(self, criterion='sample_identity', comp_fun=None):
        """
        Calculate fraction of accepted moves

        We can see whether a move was accepted or rejected by checking whether
        the samples are actually different. There are three different ways to
        do so: syntactically correct would be sample comparison (using ``==``).
        However, since samples have a user-defined data type, we do not
        necessarily have the == operator defined. For mutable objects we can
        exploit that they would not be copied for an unaccepted step, i.e. we
        can use identity check (``is``). As a last resort, we can use the
        likelihood as a proxy: it is very unlikeliy that we did a move where
        the likelihood remained exactly the same.

        Alternatively, you can provide a comparison function
        ``comp_fun(sample0, sample1)->bool`` that provides the comparison. This
        overrides the `!criterion` setting.

        Parameters
        ----------
        criterion : {'sample_equality', 'sample_identity', 'likelihood_equality'}
            which method to use to determine whether a step was accepted or
            rejected.
        comp_fun : callable, optional
            should return ``True`` if the two arguments compare equal,
            ``False`` otherwise

        Returns
        -------
        float
            the calculated acceptance rate
        """
        if comp_fun is None:
            if criterion == 'sample_equality':
                def comp_fun(sample0, sample1): return sample0 == sample1
            elif criterion == 'sample_identity':
                def comp_fun(sample0, sample1): return sample0 is sample1

        if comp_fun is not None:
            n_reject = np.count_nonzero([comp_fun(sample0, sample1)
                                         for sample0, sample1 in zip(self.samples[:-1],
                                                                     self.samples[1:])])
        elif criterion == 'likelihood_equality':
            logLs = self.logLs_trunc()
            n_reject = np.sum(logLs[:-1] == logLs[1:])
        else: # pragma: no cover
            raise ValueError("Did not understand inputs")

        return 1 - ( float(n_reject) / (len(self.samples)-1) )

    def evaluate(self, fun):
        """
        Evaluate a function on all samples

        This exploits that (if the sample data type is e.g. a user-defined
        class) many samples will actually be identical and we have to evaluate
        the function significantly fewer than ``len(samples)`` times.

        Parameters
        ----------
        fun : callable of signature ``fun(sample) --> object``
            the function to evaluate. It should expect a single sample as input
            and return something.

        Returns
        -------
        list
            a list of output values, in the order of `samples`.

        Notes
        -----
        This function is supposed to decrease computational cost. It is usually
        quicker however, to use a vectorized function ``fun`` instead.
        """
        last_val = fun(self.samples[0])
        last_sample = self.samples[0]
        out = [last_val]
        for sample in self.samples[1:]:
            if sample is not last_sample:
                last_sample = sample
                last_val = fun(sample)
            out.append(last_val)
        return out

class Sampler(ABC):
    """
    Abstract base class for MCMC sampling
    
    To implement a sampling scheme, subclass this and override

    - `propose_update`
    - `logL`
    - `callback_logging` (optional)
    - `callback_stopping` (optional)
    
    See their documentations in this base class for more details.

    The above methods take an argument `!params` that is supposed to represent
    the parameters of the model. The structure of this representation (e.g. a
    list of parameters, a `!np.ndarray`, or a custom class) is completely up to
    the user. Since all explicit handling of the parameters happens in the
    `propose_update` and `logL` methods, the code in this base class is
    completely independent from the parameterization.

    Attributes
    ----------
    config : dict
        some configuration values. See `configure`.

    Example
    -------

    >>> import numpy as np
    ... import scipy.stats

    >>> class normMCMC(Sampler):
    ...     def __init__(self, stepsize=0.1):
    ...         self.stepsize = stepsize
    ...
    ...     def propose_update(self, current_value):
    ...         proposed_value = current_value + np.random.normal(scale=self.stepsize)
    ...         logp_forward = -0.5*(proposed_value - current_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
    ...         logp_backward = -0.5*(current_value - proposed_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
    ...         return proposed_value, logp_forward, logp_backward
    ...         
    ...     def logL(self, value):
    ...         return -0.5*value**2 - 0.5*np.log(2*np.pi)
    ...     
    ...     def callback_logging(self, current_value, best_value):
    ...         print("Current value: {}".format(current_value))

    >>> mc = normMCMC()
    ... mc.configure(iterations=100, burn_in=50, log_every=10, show_progress=True)
    ... logL, vals = mc.run(1)

    """

    @abstractmethod
    def propose_update(self, params):
        """
        Propose an update step.

        Parameters
        ----------
        params : <user-specified parameter structure>
            the current parameters

        Returns
        -------
        proposed_values : <user-specified parameter structure>
            the proposed new parameters
        logp_forward : float
            (log of the) probability of proposing these values from the current
            ones
        logp_backward : float
            (log of the) probability of proposing the current values from the
            proposed ones

        See also
        --------
        logL, callback_logging
        """
        pass # pragma: no cover
        
    @abstractmethod
    def logL(self, params):
        """
        Calculate log-likelihood of the given parameters.

        Parameters
        ----------
        params : <user-specified parameter structure>
            the parameters to use

        Returns
        -------
        float
            the log-likelihood for the given parameters.

        See also
        --------
        propose_update, callback_logging
        """
        pass # pragma: no cover
    
    def callback_logging(self, current_params, best_params):
        """
        Callback upon logging.

        This function will be called whenever a status line is printed. It can
        thus be used to provide additional information about where the sampler
        is, or how the current values compare to the best ones found so far,
        etc.

        Parameters
        ----------
        current_params : <user-specified parameter structure>
            the current parameter set.
        best_params : <user-specified parameter structure>
            the best (i.e. highest likelihood) parameter set the sampler has
            seen so far.

        See also
        --------
        propose_update, logL
        """
        pass # pragma: no cover

    def callback_stopping(self, myrun):
        """
        Callback to enable early stopping

        This function will be called at well-defined intervals to check whether
        sampling should abort. Judgement is to be made based on the data so
        far, which is handed over as the `!myrun` argument. This is an
        `MCMCRun` object.

        Parameters
        ----------
        myrun : MCMCRun
            the data generated so far

        Returns
        -------
        stop : bool
            should be ``True`` if the sampling is to stop, ``False`` otherwise
        """
        pass # pragma: no cover
    
    def configure(self,
            iterations=1,
            burn_in=0,
            log_every=-1,
            check_stopping_every=-1,
            show_progress=False,
            ):
        """
        Set the configuration of the sampler.

        Note that after calling this function once, you can also access single
        configuration entries via the attribute `config`.

        Parameters
        ----------
        iterations : int
            how many MCMC iterations to run total
        burn_in : int
            how many steps to discard at the beginning
        log_every : int
            print a status line every ... steps. Set to ``-1`` (default) to
            disable status lines.
        check_stopping_every : int
            how often to call the `callback_stopping`. Set to ``-1`` (default)
            to never check.
        show_progress : bool
            whether to show a progress bar using `!tqdm`

        See also
        --------
        run
        """
        self.config = {
                'iterations' : iterations,
                'burn_in' : burn_in,
                'log_every' : log_every,
                'check_stopping_every' : check_stopping_every,
                'show_progress' : show_progress,
                }

    def run(self, initial_values):
        """
        Run the sampling scheme

        Remember to run `configure` before this.
        
        Parameters
        ----------
        initial_values : <user-specified parameter structure>
            the initial values for the sampling scheme
            
        Returns
        -------
        MCMCRun
            the output data of the run. This is essentially a wrapper class for
            an array of likelihoods and the associated samples. Plus some
            convenience functions.

        See also
        --------
        configure

        Notes
        -----
        The returned `MCMCRun` contains likelihoods for all steps starting at
        initialization, while the list of sampled parameters starts only after
        the burn-in period.
        """
        # Input processing
        try:
            config = self.config
        except AttributeError: # pragma: no cover
            raise RuntimeError("Trying to run MCMC sampler before calling configure()")
#         config['_logging'] = config['log_every'] > 0
        
        current_values = initial_values
            
        # Setup
        cur_logL = self.logL(current_values)
        cnt_accept = 0
        cnt_logging = config['log_every']
        cnt_stopping = config['check_stopping_every']
        if cnt_stopping > 0:
            cnt_stopping += config['burn_in']

        max_logL = cur_logL
        best_values = deepcopy(current_values)

        # Have both of these be lists for now, because appending is easier
        # Converted to np.array when handed to MCMCRun
        logLs = []
        params = []
        
        Mrange = range(config['iterations'])
        if config['show_progress']: # pragma: no cover
            Mrange = tqdm(Mrange)

        # Run
        for i in Mrange:
            # Update counters
            cnt_logging -= 1
            cnt_stopping -= 1
                
            # Proposal
            proposed_values, logp_forward, logp_backward = self.propose_update(current_values)
            new_logL = self.logL(proposed_values)
            if np.isfinite(cur_logL):
                with np.errstate(under='ignore', over='ignore'):
                    p_accept = np.exp(new_logL - cur_logL - logp_forward + logp_backward)
            else:
                if cur_logL == -np.inf:
                    # Always move on, even if new_logL might be -inf as well
                    p_accept = 1
                elif cur_logL == np.inf:
                    print("Encountered absorbing state (L = inf) at {}".format(current_values))
                    if i > config['burn_in']: # pragma: no cover
                        print("Aborting here")
                        break
                    else:
                        raise RuntimeError("Absorbing state during burn-in")
                else: # i.e. nan
                    raise RuntimeError("Encountered invalid state (L = NaN) at {}".format(current_values))

            # Acceptance
            if np.random.rand() < p_accept:
                cnt_accept += 1
                current_values = proposed_values
                cur_logL = new_logL
            accept_rate = cnt_accept / (i+1)
            
            # Output
            params.append(current_values)
            logLs.append(cur_logL)
            
            # Keeping track of best parameters
            if cur_logL > max_logL:
                max_logL = deepcopy(cur_logL)
                best_values = deepcopy(current_values)
            
            # Logging
            if cnt_logging == 0 or (i+1 == config['iterations'] and config['log_every'] > 0):
                cnt_logging = config['log_every']
                logstring = "iteration {}: acceptance = {:.0f}%, logL = {}".format(i+1, accept_rate*100, cur_logL)
                print(logstring)
                self.callback_logging(current_values, best_values)

            # Check early stopping
            if cnt_stopping == 0:
                cnt_stopping = config['check_stopping_every']
                if self.callback_stopping(MCMCRun(logLs, params)):
                    break

        # Return
        return MCMCRun(logLs, params[config['burn_in']:])
