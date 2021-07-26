"""
Reimplement the evidence based scheme
"""

import numpy as np

from . import mcmc, priors

class Environment:
    def __init__(self, traj, model, MCMCSampler, priorfam):
        # Sampler should be configured!
        self.traj = traj
        self.model = model
        self.sampler = MCMCSampler
        self.priorfam = priorfam
    
    def runMCMC(self, *prior_params):
        self.sampler.setup(self.traj, self.model,
                           mcmc.FixedScheme(self.priorfam, *prior_params),
                           )
        return self.sampler.run()

    def posterior_density(self, mcmcrun, trace_eval):
        prior_params = mcmcrun.samples[0][1]
        prior = self.priorfam(*prior_params)
        self.sampler.setup(self.traj, self.model,
                           mcmc.FixedScheme(self.priorfam, *prior_params),
                           )

        logLs = mcmcrun.logLs_trunc()
        if np.all(logLs == logLs[0]):
            print('MCMC ensemble collapsed to a single sample')
            return np.inf

        L_eval = self.sampler.logL((trace_eval, prior_params))

        # Chance to enter the target state from steady state
        def get_p_step_to_eval(sample):
            return self.sampler.stepping_probability(sample, (trace_eval, prior_params))
        p_step_to_eval = np.array(mcmcrun.evaluate(get_p_step_to_eval))

        # Note: p_accept = 1 *if* we evaluate at the maximum likelihood trace
        with np.errstate(over='ignore', under='ignore'):
            p_accept_move = np.minimum(1, np.exp(L_eval - logLs))
        p_enter = np.mean(p_step_to_eval * p_accept_move)

        # Survival in target state
        neighbor_logLs = [self.sampler.logL(sample) \
                          for sample in self.sampler.gen_proposal_sample_from(
                                                    (trace_eval, prior_params),
                                                    nSample=len(logLs),
                                                )]
        with np.errstate(over='ignore', under='ignore'):
            k_leave = np.mean(np.minimum(1, np.exp(neighbor_logLs - L_eval)))

        # Occupancy of the target state = p_enter * survival
        return p_enter / k_leave

    def evidence(self, mcmcrun):
        (trace_eval, prior_params), L_eval = mcmcrun.best_sample_logL()

        log_post = np.log(self.posterior_density(mcmcrun, trace_eval))
        log_prior = self.priorfam(*prior_params).logpi(trace_eval)

        if np.isinf(log_post):
            return L_eval
        else:
            return L_eval + log_prior - log_post
