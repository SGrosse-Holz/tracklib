"""
Main module of the neda inference package

This introduces the `Environment` class, which we use to facilitate repeating
tasks like running MCMC, calculating or estimating evidence. Finally, the
`main` function runs the whole scheme. Note that both of these are imported
into the ``tracklib.analysis.neda`` namespace, i.e. can be imported from there
(instead of ``tracklib.analysis.neda.neda``).
"""

import numpy as np
import scipy.optimize

from . import mcmc

class Environment:
    """
    Environment for inference runs

    This is essentially a semi-disguised way of using global variables for the
    `Trajectory` we are looking at, the inference `models.Model` we want to use, and
    the `MCMCscheme` together with the `MCMCconfig`. We use this approach
    mainly for code readability / useability.

    Attributes
    ----------
    traj : Trajectory
    model : models.Model
    MCMCconfig : dict
        see `tracklib.util.mcmc.Sampler`
    MCMCscheme : type, optional
        this should implement the `MCMCScheme` interface (i.e. be a subclass of
        this abstract base class).
    """
    # Here we use ``env`` instead of ``self``.
    def __init__(self, traj, model, MCMCconfig, MCMCscheme=mcmc.TPWMCMC):
        self.traj = traj
        self.model = model
        self.MCMCconfig = MCMCconfig
        self.MCMCscheme = MCMCscheme

    def runMCMC(env, prior):
        """
        Run the `MCMCscheme` with a given `prior`

        Parameters
        ----------
        prior : Prior
        
        Returns
        -------
        mcmc.MCMCRun
        """
        mc = env.MCMCscheme()
        mc.setup(env.traj, env.model, prior)
        mc.configure(**env.MCMCconfig)
        return mc.run()

    def posterior_density(env, mcmcrun, prior, trace_eval,
                          nSample_proposal=float('inf')):
        r"""
        Estimate the posterior density at a specific point from an `MCMCRun`

        Parameters
        ----------
        mcmcrun : MCMCRun
        prior : Prior
        trace_eval : Loopingtrace
            the point in parameter space for which to estimate the posterior
            density
        nSample_proposal : float, optional
            how many samples from the proposal distribution to use for
            evaluation of the exit rate. Defaults to exhaustive sampling, so
            may be reduced if that seems excessive.

        Returns
        -------
        float
            the estimated posterior density for `!trace_eval`. Might be
            ``np.inf`` if the MCMC ensemble collapsed to a single sample.

        Notes
        -----
        This estimation follows eq. (9) of [1]_. Essentially, we estimate the
        density at the sample point as the number of times it is entered from
        another point in the steady state ensemble, times the average number of
        steps it takes to leave this point again, divided by the total sample
        number :math:`M`. This yields

        .. math:: \hat{p}(\theta^*) = \frac{N_\text{enter}N_\text{stay}}{M} = \frac{P_\text{enter}}{k_\text{exit}} = \frac{M^{-1}\sum_{m=1}^M\, \alpha(\theta^{(m)}, \theta^*) q(\theta^{(m)}, \theta^*)}{J^{-1}\sum_{j=1}^J\, \alpha(\theta^*, \theta^{(j)})}\,,

        where the :math:`\theta^{(m)}` in the numerator are the samples from
        the given MCMC run (i.e. are assumed to be sampled from the posterior
        distribution), while :math:`\theta^{(j)}` in the denominator are
        independent samples from the proposal distribution
        :math:`q(\theta_\text{from} = \theta^*, \theta_\text{to})`
        around the evaluation point :math:`\theta^*`. Finally,
        :math:`\alpha(\theta_\text{from}, \theta_\text{to})` is the
        acceptance probability for a given step.

        References
        ----------
        .. [1] Chib, S. & Jeliazkov, I. Marginal Likelihood From the Metropolis-Hastings Output. Journal of the American Statistical Association 96, 270-281 (2001)
        """
        # Check whether the MCMC sample collapsed, in which case the posterior
        # estimation would diverge
        logLs = mcmcrun.logLs_trunc()
        if np.all(logLs == logLs[0]):
            print('MCMC ensemble collapsed to single sample')
            return np.inf

        L_eval = env.MCMCscheme.likelihood(env.traj, trace_eval, env.model, prior)

        # Chance to enter the target state from steady state
        def get_p_step_to_eval(trace): return env.MCMCscheme.stepping_probability(trace, trace_eval)
        p_step_to_eval = np.array(mcmcrun.evaluate(get_p_step_to_eval))

        # Note: p_accept = 1 *if* we evaluate at the maximum likelihood trace
        p_accept_move = env.MCMCscheme.acceptance_probability(mcmcrun.logLs_trunc(), L_eval)
        p_enter = np.mean(p_step_to_eval * p_accept_move)

        # Survival in target state
        neighbor_logLs = [env.MCMCscheme.likelihood(env.traj, trace, env.model, prior) \
                          for trace in env.MCMCscheme.gen_proposal_sample_from(trace_eval, nSample=nSample_proposal)]
        k_leave = np.mean(env.MCMCscheme.acceptance_probability(L_eval, neighbor_logLs))

        # Occupancy of the target state = p_enter * survival
        return p_enter / k_leave

    def evidence(env, prior, mcmcrun):
        """
        Calculate the (log-)evidence from an `MCMCRun`

        Parameters
        ----------
        prior : Prior
            the prior over `Loopingtraces <Loopingtrace>` used for the MCMC run
        mcmcrun : MCMCRun
            the MCMC sample

        Returns
        -------
        float
            the estimated log-evidence

        Notes
        -----
        The log-evidence is given by ``log(likelihood) + log(prior) -
        log(posterior)``.

        See also
        --------
        posterior_density
        """
        trace_eval, L_eval = mcmcrun.best_sample_L()
        log_post = np.log(env.posterior_density(mcmcrun, prior, trace_eval))
        log_prior = prior.logpi(trace_eval)

        if np.isinf(log_post):
            return L_eval
        else:
            return L_eval + log_prior - log_post

    def evidence_differential(env, prior, ref_prior, ref_mcmcrun):
        r"""
        Estimate the relative evidence given a reference point

        Parameters
        ----------
        prior : Prior
            the prior whose evidence we want to estimate
        ref_prior : Prior
            the reference point in prior space
        ref_mcmcrun : MCMCRun
            an `MCMCRun` using the reference prior

        Returns
        -------
        float
            the estimated log-evidence, relative to the reference

        Notes
        -----
        The relative evidence is given by the expectation value of the prior
        ratio over the MCMC sample: ``E = < prior/ref_prior >`` where ``<.>``
        indicates an average over the MCMC sample.
        """
        return np.log(np.mean(np.exp(prior.logpi_vectorized(ref_mcmcrun.samples) - \
                                     ref_prior.logpi_vectorized(ref_mcmcrun.samples))))

def main(traj, model, priorfam,
         MCMCconfig, MCMCscheme=mcmc.TPWMCMC,
         max_iterations=20, min_iterations=5,
         return_ = 'nothing', # 'traj', 'dict', or anything else
         show_progress=False, assume_notebook_for_progressbar=True,
        ):
    """
    Run the neda looping inference scheme

    The output of the inference run will be assembled into a dict whose fields
    are detailed below. Where exactly this dict will end up depends on the
    setting of `!return_`.
    
    Parameters
    ----------
    traj : Trajectory
        the `Trajectory` whose looping profile to infer
    model : models.Model
        the inference model to use
    priorfam : ParametricFamily
        a family of priors
    MCMCconfig : dict
        configuration for the MCMC runs. See `tracklib.util.mcmc.Sampler`
    MCMCscheme : type, optional
        a class implementing the `MCMCScheme` interface (i.e. a subclass of
        this abstract base class). Defines the MCMC sampling scheme
    max_iterations : int, optional
        maximum number of iterations to run
    min_iterations : int, optional
        run at least this many iterations
    return_ : {'nothing', 'None', 'traj', 'dict'}
        what the return value of this function should be. Generally, the
        results of the inference run will be stored in a dict. If
        ``return_='dict'``, that dict is directly returned. Otherwise it is
        written to ``traj.meta['neda']``, and if ``return_='traj'`` the
        trajectory is returned (useful for parallelization). Otherwise this
        function returns nothing, i.e. the inference results can be accessed
        simply from ``traj.meta['neda']`` after calling this function.
    show_progress : bool, optional
        whether to show a progress bar. Note: for ``show_progress=True`` it
        might happen that the termination condition is fulfilled during the
        last "required" run, in which case the progress bar will stop before
        reaching its maximum.
    assume_notebook_for_progressbar : bool, optional
        set to ``False`` if running outside Jupyter notebook to show
        progressbar ASCII style

    Returns
    -------
    prior_prams : np.array
        the prior parameters for each iteration
    mcmcrun : list of MCMCRun
        the MCMC runs for each iteration
    evidence : np.array
        the evidence estimated from each `MCMCRun`
    evidence_diff : np.array
        the estimated evidence differential (relative evidence) at each
        iteration
    final : dict
        the corresponding values for the iteration that should be considered
        the final result. Has entries ``'prior_params'``, ``'mcmcrun'``,
        ``'evidence'``, and ``'iteration'`` where the last one is the index of
        the iteration the other three refer to.

    Example
    -------
    Assuming we have a `Trajectory` ``traj`` that we want to run the looping
    inference on:

    >>> # Set up inference scheme
    ... looppositions = [(0, 0), (0, -1)] # 2 states: unlooped (=0), fully looped (=1)
    ... model = neda.models.RouseModel(N=20, D=1, k=5, k_extra=1, looppositions=looppositions)
    ... priorfam = neda.ParametricFamily(start_params=(0), bounds=[(None, 0)])
    ... priorfam.get = lambda logq : neda.priors.GeometricPrior(logq, nStates=len(looppositions))
    ... MCMCconfig = {
    ...         'iterations' : 1000,
    ...         'burn_in'    :  100,
    ...         }
    ... 
    ... # Run the inference
    ... neda.main(traj, model, priorfam, MCMCconfig, show_progress=True)
    ... 
    ... # Visualize output
    ... from matplotlib import pyplot as plt
    ... neda.plot.butterfly(traj)
    ... plt.show()

    See also
    --------
    tracklib.analysis.kli.fit_RouseParams
    """
    assert min_iterations >= 2
    assert return_ in {'nothing', 'None', 'traj', 'dict'}

    # Set up environment
    env = Environment(traj, model, MCMCconfig, MCMCscheme)

    # Set up iterative scheme
    iterations = range(max_iterations)
    if show_progress: # pragma: no cover
        if assume_notebook_for_progressbar:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        iterations = tqdm(iterations, total=min(min_iterations, max_iterations))

    prior_params = [priorfam.start_params]
    mcmcruns = []
    evidences = []
    evidence_diffs = []

    # Run iterations
    for it in iterations:
        prior = priorfam.get(*prior_params[-1])
        mcmcruns.append(env.runMCMC(prior))
        evidences.append(env.evidence(prior, mcmcruns[-1]))

        if it+1 >= min_iterations:
            if np.argmax(evidences) < it:
                break
        
        # Maximize estimated evidence differential to find new prior parameters
        def minimization_target(*params):
            return -env.evidence_differential(priorfam.get(*params), prior, mcmcruns[-1])
        minimization_result = scipy.optimize.minimize(minimization_target,
                                                      x0=prior_params[-1],
                                                      bounds=priorfam.bounds)

        if not minimization_result.success: # pragma: no cover
            print(minimization_result)
            raise RuntimeError('Relative evidence maximization did not converge')

        evidence_diffs.append(-minimization_result.fun)
        prior_params.append(tuple(minimization_result.x))

    # Output
    best_it = np.argmax(evidences)
    output = {
        'prior_params'  : np.array(prior_params),
        'mcmcrun'       : mcmcruns,
        'evidence'      : np.array(evidences),
        'evidence_diff' : np.array(evidence_diffs),
        'final'         : {
            'prior_params' : prior_params[best_it],
            'mcmcrun'      : mcmcruns[best_it],
            'evidence'     : evidences[best_it],
            'iteration'    : best_it,
            },
        }

    if return_ == 'dict':
        return output
    else:
        traj.meta['neda'] = output
        if return_ == 'traj':
            return traj
        else:
            return None
