"""
Reimplement the evidence based scheme
"""

import numpy as np

from . import mcmc, priors
from .util import ParametricFamily

class Environment:
    def __init__(self, traj, model, MCMCSampler, priorfam):
        # Sampler should be configured!
        self.traj = traj
        self.model = model
        self.sampler = MCMCSampler
        self.priorfam = priorfam

    @classmethod
    def for_numInt_pruning(cls, traj, model, MCMCSampler):
        priorfam = ParametricFamily((10, 0.3), [(0, None), (0, 1)])
        priorfam.get = lambda K, q : priors.NumIntPrior(K+1, logq=np.log(q), nStates=model.nStates)

        return cls(traj, model, MCMCSampler, priorfam)
    
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

def numInt_pruning(env, q=0.3, max_iteration=20, update='K'):
    assert update in {'K', 'mean k'}

    geometric_prior = priors.GeometricPrior(logq=np.log(q), nStates=env.model.nStates)
    geometric_p = 1/(1 + 1/((env.model.nStates - 1)*q))

    runs = []
    cur_K = len(env.traj) - 1
    for _ in range(max_iteration):
        # Do the current run
        mcmcrun = env.runMCMC(cur_K, q)
        
        # Add geometric likelihoods
        cur_prior = env.priorfam(cur_K, q)
        mcmcrun.geometric_logLs_trunc = np.array([ \
                logL - cur_prior.logpi(profile) + geometric_prior.logpi(profile) \
                for (profile, _), logL in zip(mcmcrun.samples, mcmcrun.logLs_trunc()) \
        ])

        # Find best profile
        i = np.argmax(mcmcrun.geometric_logLs_trunc)
        profile = mcmcrun.samples[i][0]
        k = profile.count_switches()
        geom_logL = mcmcrun.geometric_logLs_trunc[i]

        # Save
        runs.append({
            'mcmcrun' : mcmcrun,
            'K' : cur_K,
            'best_profile' : profile,
            'best_k' : k,
            'best_logL' : geom_logL,
        })

        if update == 'K':
            if k == cur_K:
                break
            else:
                cur_K = k
        else:
            if k >= cur_K*geometric_p:
                break
            else:
                cur_K = int(np.floor(k / geometric_p))


    else:
        print("Run with q = {:.3g} did not converge after {} iterations\nThe K sampled so far are {}".format(q, max_iteration, str([run['K'] for run in runs])))
        raise RuntimeError

    i_best_run = np.argmax([run['best_logL'] for run in runs])
    final_profile = runs[i_best_run]['best_profile']

    return final_profile, runs
