"""
This module is used to execute the NEDA scheme to infer looping dynamics from
locus tracking trajectories.

Notes
-----
"Loopingtrace"s contain an integer for each non-nan data point in a trajectory,
indicating which out of a given number of models to use for evolution between
the last data point and this one.
"""

import abc
from copy import deepcopy

import numpy as np
import scipy.stats
import scipy.optimize

from tracklib.util import mcmc
from tracklib.models import rouse

class Loopingtrace:
    """
    Trace of looping states
    """
    def __init__(self, traj, nStates=2, thresholds=None):
        self.T = len(traj)
        self.n = nStates
        self.t = np.array([i for i in range(len(traj)) if not np.any(np.isnan(traj[i]))])
        self.state = np.zeros((len(self.t),))

        if thresholds is not None:
            self.n = len(thresholds)+1
            dist_traj = traj.abs()[:][:, 0]
            self.state = len(thresholds) - \
                         np.sum([ \
                                 (dist_traj[self.t] < thres).astype(float) \
                                 for thres in thresholds \
                                ], axis=0)

    def copy(self):
        # Faster than deepcopy()
        new = self.__new__(type(self)) # Skip init
        new.T = self.T
        new.n = self.n
        new.t = self.t.copy()
        new.state = self.state.copy()
        return new

    def __len__(self):
        return len(self.state)

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, val):
        self.state[key] = val

    def plottable(self):
        tplot = np.array([np.insert(self.t[:-1], 0, self.t[0]-1), self.t]).T.flatten()
        yplot = np.array([self.state, self.state]).T.flatten()
        return tplot, yplot

    def full_valid(self):
        """
        Return a full-length array of looping states, no nans.
        """
        full = np.zeros((self.T,))
        last_ind = 0
        for cur_ind, cur_val in zip(self.t, self.state):
            full[last_ind:(cur_ind+1)] = cur_val
            last_ind = cur_ind
        return full

class Model(metaclass=abc.ABCMeta):
    def initial_loopingtrace(self, traj):
        return Loopingtrace(traj)

    @abc.abstractmethod
    def logL(self, loopingtrace, traj):
        raise NotImplementedError

class RouseModel(Model):
    """
    Use ``n+1`` `rouse.Model` instances
    
    Remember to include the unlooped model in `looppositions`.
    """
    def __init__(self, N, D, k, looppositions=[(0, 0), (0, -1)], k_extra=None, measurement="end2end"):
        if k_extra is None:
            k_extra = k
        if measurement == "end2end":
            measurement = np.zeros((N,))
            measurement[0]  = -1
            measurement[-1] =  1

        looppositions.insert(0, (0, 0))

        for iloop in looppositions:
            mod = rouse.Model(N, D, k, k_extra, extrabond=loop)
            mod.measurement = measurement
            self.models.append(mod)

    def initial_loopingtrace(self, traj):
        # TODO: come up with a good scheme here
        return Loopingtrace(traj, len(self.models))

    def logL(self, loopingtrace, traj):
        if traj.N == 2:
            traj = traj.relative()

        looptrace = loopingtrace.full_valid()
        return np.sum([ \
                rouse.multistate_likelihood(traj[:][:, i],
                                            self.models,
                                            looptrace,
                                            traj.meta['localization_error'][i],
                                           ) \
                for i in range(traj.d)])

class FactorizedModel(Model):
    """
    distributions should be list of ``scipy.stats.rv_continuous``.
    In fact, we only need the (vectorized) function ``logpdf``.
    """
    def __init__(self, distributions):
        self.distributions = distributions
        self._known_trajs = dict()

    def _memo(self, traj):
        if not traj in self._known_trajs:
            logL_table = np.array([dist.logpdf(traj.abs()[:][:, 0]) 
                                   for dist in self.distributions
                                   ])
            self._known_trajs[traj] = {'logL_table' : logL_table}

    def initial_loopingtrace(self, traj):
        self._memo(traj)
        loopingtrace = Loopingtrace(traj, len(self.distributions))
        loopingtrace.state = np.argmax(self._known_trajs[traj]['logL_table'][:, loopingtrace.t], axis=0)
        return loopingtrace

    def logL(self, loopingtrace, traj):
        self._memo(traj)
        return np.sum(self._known_trajs[traj]['logL_table'][loopingtrace.state, loopingtrace.t])

class Prior(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def logpi(self, loopingtrace):
        raise NotImplementedError

class PriorFactory:
    """
    Example
    ------
    >>> priorfac = PriorFactory((0,), [(None, 0)])
    ... priorfac.get = lambda logq : GeometricPrior(logq)
    ... priorfac.get = GeometricPrior # alternative
    """
    def __init__(self, start_params, bounds):
        self.start_params = start_params
        self.nParams = len(start_params)
        self.bounds = bounds

    def get(self, *params):
        raise NotImplementedError

class UniformPrior(Prior):
    def __init__(self, nStates=2):
        self.n = nStates

    def logpi(self, loopingtrace):
        return -len(loopingtrace.t)*np.log(self.n)

class GeometricPrior(Prior):
    def __init__(self, logq=0, nStates=2):
        self.logq = logq
        self.n = nStates

        self._log_n = np.log(self.n)
        self._log_norm_per_dof = np.log(1+np.exp(self.logq)*(self.n-1))

    def logpi(self, loopingtrace):
        # optimized
        k = np.count_nonzero(loopingtrace.state[1:] != loopingtrace.state[:-1])
        return k*self.logq - (len(loopingtrace.t)-1)*self._log_norm_per_dof  - self._log_n

    @classmethod
    def factory(cls, nStates=2):
        fac = PriorFactory((0,), [(None, 0)])
        fac.get = lambda logq : cls(logq, nStates)
        return fac

class MCMCRun:
    def __init__(self, logLs=None, samples=None):
        self.logLs = deepcopy(logLs)
        self.samples = deepcopy(samples)

    def logLs_trunc(self):
        return self.logLs[-len(self.samples):]

    def best_sample_L(self):
        logLs = self.logLs_trunc()
        i_best = np.argmax(logLs)
        return self.samples[i_best], logLs[i_best]

    def acceptance_rate(self, criterion='sample_identity'):
        if criterion == 'parameter_identity':
            n_accept = np.sum([1 if sample0 is not sample1 else 0
                               for sample0, sample1 in zip(self.samples[:-1], self.samples[1:])
                               ])
        elif criterion == 'likelihood_equality':
            logLs = self.logLs_trunc()
            n_accept = np.sum(logLs[:-1] != logLs[1:])

        return float(n_accept) / (len(self.samples)-1)

    def evaluate(self, fun):
        # fun(sample) --> whatever
        # Exploits that many samples are going to be the same
        last_val = fun(self.samples[0])
        last_sample = self.samples[0]
        out = [last_val]
        for sample in self.samples[1:]:
            if sample is not last_sample:
                last_sample = sample
                last_val = fun(sample)
            out.append(last_val)
        return out

class MCMCScheme(mcmc.Sampler, metaclass=abc.ABCMeta):
    def setup(self, traj, model, prior):
        self.traj = traj
        self.model = model
        self.prior = prior

    def run(self, *args, **kwargs):
        res = MCMCRun()
        res.logLs, res.samples = mcmc.Sampler.run(self,
                                                 self.model.initial_loopingtrace(self.traj),
                                                 *args, **kwargs)
        return res

    @staticmethod
    def acceptance_probability(L_from, L_to):
        """ generic formula, do not change """
        with np.errstate(over='ignore', under='ignore'):
            try:
                return np.minimum(1, np.exp(L_to - L_from))
            except Exception as err:
                print(L_to, L_from)
                raise err

    @staticmethod
    def likelihood(traj, loopingtrace, model, prior):
        return model.logL(loopingtrace, traj) + prior.logpi(loopingtrace)

    def logL(self, loopingtrace):
        return self.likelihood(self.traj, loopingtrace, self.model, self.prior)

    #--------------------------------------------------------------------------
    # NOTE: the following three functions are all different representations of
    # the sampling scheme, so they *all* have to be updated when changing
    # anything about that!

    @abc.abstractmethod
    def propose_update(self, loopingtrace_cur):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def stepping_probability(loopingtrace_from, loopingtrace_to):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def gen_proposal_sample_from(loopingtrace, nSample):
        # NOTE: this has to be an actual sample from the distribution. We'll
        # use it to replace an average value with mean over this ensemble.
        raise NotImplementedError

class TPWMCMC(MCMCScheme):
    # NOTE: the following three functions are all different representations of
    # the sampling scheme, so they *all* have to be updated when changing
    # anything about that!

    def propose_update(self, loopingtrace_cur):
        nNeighbors = len(loopingtrace_cur.t)*(loopingtrace_cur.n-1)

        loopingtrace_prop = loopingtrace_cur.copy()
        ind_up = np.random.randint(len(loopingtrace_prop))
        cur_val = loopingtrace_prop[ind_up]
        loopingtrace_prop[ind_up] = np.random.choice(list(range(cur_val))+list(range(cur_val+1, loopingtrace_prop.n)))

        return loopingtrace_prop, 1./nNeighbors, 1./nNeighbors

    @staticmethod
    def stepping_probability(loopingtrace_from, loopingtrace_to):
        nNeighbors = len(loopingtrace_from.t)*(loopingtrace_from.n-1)

        if (np.all(loopingtrace_from.t == loopingtrace_to.t) and
            np.sum(loopingtrace_from[:] != loopingtrace_to[:]) == 1):
            return 1/nNeighbors
        else:
            return 0

    @staticmethod
    def gen_proposal_sample_from(loopingtrace, nSample=float('inf')):
        nNeighbors = len(loopingtrace.t)*(loopingtrace.n-1)
        nSample = min(nSample, nNeighbors)
        sample_ids = np.random.choice(nNeighbors, size=(nSample,), replace=False)

        for sid in sample_ids:
            i_update = int(sid // (loopingtrace.n-1))
            new_val = int(sid % (loopingtrace.n-1))
            if new_val >= loopingtrace.t[i_update]:
                new_val += 1

            neighbor = loopingtrace.copy()
            neighbor[i_update] = new_val
            yield neighbor

class Environment:
    """
    Here we use ``env`` instead of ``self``.
    """
    def __init__(self, traj, model, MCMCconfig, MCMCscheme=TPWMCMC):
        self.traj = traj
        self.model = model
        self.MCMCconfig = MCMCconfig
        self.MCMCscheme = MCMCscheme

        for key in list(self.traj.meta.keys()):
            if key.startswith('_neda_'):
                del self.traj.meta[key]

    def close(self):
        for key in list(self.traj.meta.keys()):
            if key.startswith('_neda_'):
                del self.traj.meta[key]


    def runMCMC(env, prior):
        mc = env.MCMCscheme()
        mc.setup(env.traj, env.model, prior)
        mc.configure(**env.MCMCconfig)
        return mc.run()

    def posterior_density(env, mcmcrun, prior, trace_eval,
                          nSample_proposal=float('inf')):
        # This estimation follows [Chib & Jeliazkov, 2001, eq. (9)].

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
        # Technically: log-evidence
        trace_eval, L_eval = mcmcrun.best_sample_L()
        log_post = np.log(env.posterior_density(mcmcrun, prior, trace_eval))
        log_prior = prior.logpi(trace_eval)
        return L_eval + log_prior - log_post

    def evidence_differential(env, prior, ref_prior, ref_mcmcrun):
        # Technically: log-evidence
        def prior_ratio(trace): return prior.logpi(trace) - ref_prior.logpi(trace)
        return np.log(np.mean(np.exp(ref_mcmcrun.evaluate(prior_ratio))))

def main(traj, model, priorfac,
         MCMCconfig, MCMCscheme=TPWMCMC,
         max_iterations=20, run_full=False,
         return_ = 'nothing', # 'traj', 'dict', or anything else
         show_progress=False, assume_notebook_for_progressbar=True,
        ):
    # Set up environment
    env = Environment(traj, model, MCMCconfig, MCMCscheme)

    # Set up iterative scheme
    iterations = range(max_iterations)
    if show_progress:
        if assume_notebook_for_progressbar:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        iterations = tqdm(iterations, total=max_iterations if run_full else 0)

    prior_params = [priorfac.start_params]
    mcmcruns = []
    evidences = []
    evidence_diffs = []

    # Run iterations
    for it in iterations:
        prior = priorfac.get(*prior_params[-1])
        mcmcruns.append(env.runMCMC(prior))
        evidences.append(env.evidence(prior, mcmcruns[-1]))

        if it > 0 and not run_full:
            if evidences[-1] < evidences[-2] + 1e-7: # is the wiggle room necessary?
                break
        
        # Maximize estimated evidence differential to find new prior parameters
        def minimization_target(*params):
            return -env.evidence_differential(priorfac.get(*params), prior, mcmcruns[-1])
        minimization_result = scipy.optimize.minimize(minimization_target,
                                                      x0=prior_params[-1],
                                                      bounds=priorfac.bounds)

        if not minimization_result.success:
            print(minimization_result)
            raise RuntimeError('Relative evidence maximization did not converge')

        evidence_diffs.append(-minimization_result.fun)
        prior_params.append(tuple(minimization_result.x))
    
    # Done
    env.close()

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

from matplotlib import pyplot as plt

def butterfly_plot(traj, fig=None, title='Example trajectory',
                   ylim=[0, None], ylabel='distance',
                  ):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if fig is None:
        fig = plt.figure(figsize=[12, 7])
    axs = fig.subplots(3, 4,
                       gridspec_kw={'height_ratios' : [0.7, 0.3, 1],
                                    'hspace' : 0,
                                    'width_ratios' : [1, 0.3, 0.3, 0.3],
                                    'wspace' : 0,
                                   },
                       sharex='col',
                       sharey='row',
                      )
    
    ref_lt = traj.meta['neda']['final']['mcmcrun'].samples[0]

    # Trajectory
    ax = axs[0, 0]
    
    ax.set_title(title)
    ax.plot(ref_lt.t, traj.abs()[ref_lt.t][:, 0], label='distance', color=colors[0])
#     ax.legend()
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    
    # Loop states
    ax = axs[1, 0]
    
    statedists = np.empty((ref_lt.n, len(traj)))
    statedists[:] = np.nan
    for state in range(statedists.shape[0]):
        statedists[state, ref_lt.t] = np.sum([trace[:] == state
                                              for trace in traj.meta['neda']['final']['mcmcrun'].samples
                                             ], axis=0)
    statedists /= np.sum(statedists, axis=0)
    
    pcm = ax.pcolormesh(np.arange(statedists.shape[1]+1)-0.5,
                        np.arange(statedists.shape[0]+1)-0.5,
                        statedists,
                        cmap='Greens',
                       )
    ax.plot(ref_lt.t, np.argmax(statedists[:, ref_lt.t], axis=0), label='MmAP', color=colors[1])
#     ax.legend()
    ax.set_ylim([-0.5, statedists.shape[0]-0.5])
    ax.set_ylabel('loop state')
    ### interesting bug: ax.get_yticks() gives values outside of ylim here...
#     ticks = ax.get_yticks()
#     ylim = ax.get_ylim()
#     ax.set_yticks([tick for tick in ticks if tick % 1 == 0 and tick >= ylim[0] and tick <= ylim[1]])
    ax.set_yticks(np.arange(ref_lt.n))
    
    cax = fig.add_axes([0.05, 0.51, 0.015, 0.1])
    fig.colorbar(pcm, cax=cax, label='p(state)', orientation='vertical', ticks=[0, 1])
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    # Waterfall
    ax = axs[2, 0]
    
    ensembles = np.empty((len(traj.meta['neda']['mcmcrun']), len(traj)))
    ensembles[:] = np.nan
    for i, mcmc in enumerate(traj.meta['neda']['mcmcrun']):
        ensembles[i, ref_lt.t] = np.mean([trace[:] for trace in mcmc.samples], axis=0)
    
    pcm = ax.pcolormesh(np.arange(ensembles.shape[1]+1)-0.5,
                        np.arange(ensembles.shape[0]+1)-0.5,
                        ensembles,
                        cmap='viridis',
                       )
    ax.set_ylim([-0.5, ensembles.shape[0]-0.5])
    ax.set_ylabel('iteration')
    ticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ax.set_yticks([tick for tick in ticks if tick % 1 == 0 and tick >= ylim[0] and tick <= ylim[1]])
    
    cax = fig.add_axes([0.05, 0.18, 0.015, 0.25])
    fig.colorbar(pcm, cax=cax, label='mean state', orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')
    
    # X-axis for this column
    ax = axs[2, 0]
    ax.set_xlabel('time [frames]')
    ax.set_xlim([-0.5, len(traj)-0.5])

    # log(q)
    ax = axs[2, 1]
    
    ax.plot(np.array(traj.meta['neda']['prior_params']), np.arange(len(traj.meta['neda']['prior_params'])), marker='o')
    ax.set_title('prior parameter')
    ax.set_xlabel('log(q)')
    ax.invert_xaxis()
    
    # Δevidence
    ax = axs[2, 2]
    
    ax.plot(traj.meta['neda']['evidence_diff'], np.arange(len(traj.meta['neda']['evidence_diff']))+0.5, marker='v', color='g')
    ax.set_title('evidence gain')
    ax.set_xlabel('Δlog P(y)')
    
    # real evidence
    ax = axs[2, 3]
    
    ax.plot(traj.meta['neda']['evidence'], np.arange(len(traj.meta['neda']['evidence'])), marker='^', color='purple')
    ax.set_title('real evidence')
    ax.set_xlabel('log P(y)')
    
    # best iteration
    it_final = traj.meta['neda']['final']['iteration']
    for ax in axs[2, :]:
        ax.axhline(it_final, linestyle='--', color='k')

    # Housekeeping
    axs[2, 0].invert_yaxis()
    axs[2, 1].invert_yaxis() # Bug?
    axs[2, 2].invert_yaxis() # Bug?
    axs[2, 3].invert_yaxis() # Bug?
    
    for ax in list(axs[0, 1:]) + list(axs[1, 1:]):
        ax.axis('off')

    return fig, axs


























































