"""
A small module for running MCMC sampling. It provides the abstract base class
`Sampler` that can be subclassed for concrete sampling schemes.
"""

from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

class Sampler(ABC):
    """
    Abstract base class for MCMC sampling
    
    To implement a sampling scheme, subclass this and override

    - `propose_update`
    - `logL`
    - `callback_logging` (optional)
    
    See their documentations in this base class for more details.

    The above methods take an argument `!params` that is supposed to represent
    the parameters of the model. The structure of this representation (e.g. a
    list of parameters, a `!np.ndarray`, or a custom class) is completely up to
    the user. Since all explicit handling of the parameters happens in the
    `propose_update` and `logL` methods, the code in this base class is
    completely independent from the parameterization.

    Attributes
    ----------
    stepsize : float
        a generic stepsize variable that can be used in `propose_update`.
    config : dict
        some configuration values. See `configure`.

    Example
    -------

    >>> import numpy as np
    ... import scipy.stats

    >>> class normMCMC(Sampler):
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

        You can use ``self.stepsize`` for the proposal.
        
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
    
    def configure(self,
            stepsize=0.1,
            iterations=100,
            burn_in=50,
            log_every=-1,
            best_only=False,
            show_progress=False,
            assume_notebook_for_progress_display=True,
            ):
        """
        Set the configuration of the sampler.

        Note that after calling this function once, you can also access single
        configuration entries via the attribute `config`.

        Keyword arguments
        -----------------
        stepsize : float
            this will be written to `stepsize` and can then be used e.g. in
            `propose_update`.
        iterations : int
            how many MCMC iterations to run total
        burn_in : int
            how many steps to discard at the beginning
        log_every : int
            print a status line every ... steps. Set to ``-1`` (default) to
            disable status lines.
        best_only : bool
            instead of the whole sampling history, have `run` return only the
            best fit.
        show_progress : bool
            whether to show a progress bar using `!tqdm`
        assume_notebook_for_progress_display : bool
            if ``True``, use `!tqdm.notebook.tqdm` otherwise `!tqdm.tqdm`. The
            former displays the progress bar as a jupyter widget, the latter as
            ASCII.

        See also
        --------
        run
        """
        self.config = {
                'stepsize' : stepsize,
                'iterations' : iterations,
                'burn_in' : burn_in,
                'log_every' : log_every,
                'best_only' : best_only,
                'show_progress' : show_progress,
                'assume_notebook_for_progress_display' : assume_notebook_for_progress_display,
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
        logL : np.ndarray
            log-likelihood for the parameters at each step.
        params : list of <user-specified parameter structure>
            list of the sampled parameter sets, after the burn-in period

        See also
        --------
        configure

        Notes
        -----
        The returned list of log-likelihoods contains likelihoods for all steps
        starting at initialization, while the list of sampled parameters starts
        only after the burn-in period.
        """
        # Input processing
        try:
            config = self.config
        except AttributeError: # pragma: no cover
            raise RuntimeError("Trying to run MCMC sampler before calling configure()")
        config['_logging'] = config['log_every'] > 0
        self.stepsize = config['stepsize']
        
        current_values = initial_values
            
        # Setup
        cur_logL = -np.inf
        max_logL = -np.inf
        cnt_accept = 0
        if not config['best_only']:
            logL = -np.inf*np.ones((config['iterations'],))
            params = []
        
        Mrange = range(config['iterations'])
        if config['show_progress']: # pragma: no cover
            if config['assume_notebook_for_progress_display']:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            Mrange = tqdm(Mrange)
            del tqdm

        # Run
        for i in Mrange:
            # Proposal
            proposed_values, p_forward, p_backward = self.propose_update(current_values)
            new_logL = self.logL(proposed_values)
            with np.errstate(under='ignore', over='ignore'):
                p_accept = np.exp(new_logL - cur_logL - p_forward + p_backward)

            # Acceptance
            if np.random.rand() < p_accept:
                cnt_accept += 1
                current_values = proposed_values
                cur_logL = new_logL
            accept_rate = cnt_accept / (i+1)
            
            # Output
            if not config['best_only']:
                params.append(current_values)
                logL[i] = cur_logL
            
            # Keeping track of best parameters
            if cur_logL > max_logL:
                max_logL = deepcopy(cur_logL)
                best_values = deepcopy(current_values)
            
            # Logging
            if config['_logging'] and ( (i+1) % config['log_every'] == 0 or i+1 == config['iterations'] ):
                logstring = "iteration {}: acceptance = {:.0f}%, logL = {}".format(i+1, accept_rate*100, cur_logL)
                print(logstring)
                self.callback_logging(current_values, best_values)
                
        if config['best_only']:
            return max_logL, best_values
        else:
            s = slice(config['burn_in'], None)
            return logL, params[s]
