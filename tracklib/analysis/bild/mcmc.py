"""
The MCMC sampler used for inference
"""

import abc
from copy import deepcopy

import numpy as np
import scipy.stats

from tracklib.util import mcmc
from .util import Loopingtrace
from .priors import GeometricPrior

class PriorScheme(metaclass=abc.ABCMeta):
    """
    "Beefy ParametricFamily" for MCMC sampling
    Similar to the old MCMCScheme
    """
    @abc.abstractmethod
    def get(self, params):
        raise NotImplementedError # pragma: no cover

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    @property
    def nStates(self):
        raise NotImplementedError # pragma: no cover

    @abc.abstractmethod
    def stepping_probability(self, params_from, params_to):
        raise NotImplementedError # pragma: no cover

    @abc.abstractmethod
    def gen_proposal_sample_from(self, params=None, nSample=1):
        # without params give initial values
        raise NotImplementedError # pragma: no cover

class GeometricPriorScheme(PriorScheme):
    def __init__(self, nStates=2, stepsize=0.1):
        self._nStates = nStates
        self.stepsize = stepsize

    @property
    def nStates(self):
        return self._nStates

    def get(self, logq):
        return GeometricPrior(logq, self.nStates)

    def stepping_probability(self, logq_from, logq_to):
        return scipy.stats.gamma(a = -logq_from/self.stepsize + 1,
                                 scale = self.stepsize).pdf(-logq_to)

    def gen_proposal_sample_from(self, logq=None, nSample=1):
        if logq is None:
            return iter(nSample * [-1e5])
        else:
            return iter(-scipy.stats.gamma(a = -logq/self.stepsize + 1,
                                           scale = self.stepsize).rvs(nSample))

class FullMCMC(mcmc.Sampler):
    def __init__(self, move_weights=(0.1, 1, 1), alpha=2, initialize_from='model'):
        self.cum_moves = np.cumsum(move_weights).astype(float)
        self.cum_moves /= self.cum_moves[-1]
        self.alpha = float(alpha)

        assert initialize_from in {'model', 'random'}
        self.initialize_from = initialize_from

    def setup(self, traj, model, priorscheme):
        assert traj.d == model.d
        assert model.nStates == priorscheme.nStates

        self.traj = traj
        self.model = model
        self.priorscheme = priorscheme

####### Overwrite the necessary mcmc.Sampler functions

    def configure(self,
            check_stopping_every=1000,
            min_approaches_to_best_sample=10,
            **kwargs,
            ):
        mcmc.Sampler.configure(self,
                               check_stopping_every=check_stopping_every,
                               **kwargs)
        self.config['min_approaches_to_best_sample'] = min_approaches_to_best_sample

    def callback_stopping(self, myrun):
        (best_trace, best_pp), max_logL = myrun.best_sample_logL()
        best_trace_set = {trace for trace, _ in myrun.samples if trace == best_trace}
        n_indep_approaches = len(best_trace_set)
        return n_indep_approaches >= self.config['min_approaches_to_best_sample']

    def run(self, *args, **kwargs):
        if self.initialize_from == 'model':
            init_trace = self.model.initial_loopingtrace(self.traj)
        else:
            init_trace = Loopingtrace.forTrajectory(self.traj, nStates=self.model.nStates)

        return mcmc.Sampler.run(self,
                                (init_trace, next(self.priorscheme.gen_proposal_sample_from())),
                                *args, **kwargs)

    def logL(self, params):
        loopingtrace, priorparams = params
        prior = self.priorscheme(priorparams)
        return self.model.logL(loopingtrace, self.traj) + prior.logpi(loopingtrace)

    def propose_update(self, params):
        proposed = next(self.gen_proposal_sample_from(params))
        p_fwd = self.stepping_probability(params, proposed)
        p_bwd = self.stepping_probability(proposed, params)
        if p_bwd == 0: # pragma: no cover
            raise RuntimeError("Found irreversible move")
        return proposed, np.log(p_fwd), np.log(p_bwd)

####### Parts for propose_update

    def stepping_probability(self, params_from, params_to):
        lt_from, pp_from = params_from
        lt_to,   pp_to   = params_to
        return self.lt_stepping_probability(lt_from, lt_to) \
                * self.priorscheme.stepping_probability(pp_from, pp_to)

    def gen_proposal_sample_from(self, params, nSample=1):
        lt, pp = params
        return zip(self.lt_gen_proposal_sample_from(lt, nSample),
                   self.priorscheme.gen_proposal_sample_from(pp, nSample))

####### The actual proposal scheme for Loopingtraces

    def lt_stepping_probability(self, lt_from, lt_to):
        assert len(lt_from) == len(lt_to)

        pad_from = np.pad(lt_from.state, (1, 1), constant_values=-1)
        pad_to   = np.pad(  lt_to.state, (1, 1), constant_values=-1)

        # Number of intervals in the original trace
        N_interval = np.count_nonzero(np.diff(pad_from)) - 1

        # Find the end points of the interval where the traces differ and check
        # that it is a single interval indeed
        ind = np.nonzero(np.diff(pad_to != pad_from))[0]
        if len(ind) != 2:
            return 0.
        a, b = ind[:2]
        if np.any(np.diff(lt_from[a:b])): # don't have to check lt_to, that's already guaranteed
            return 0.

        L = b - a

        # Find embedding interval in lt_from
        old_state = lt_from[a]
        ind = np.arange(len(lt_from)+1)
        a_embed = np.nonzero((ind <= a) * (pad_from[:-1] != old_state))[0][-1]
        b_embed = np.nonzero((ind >= b) * (pad_from[1:]  != old_state))[0][0]

        N = b_embed - a_embed

        # Add up different pathways
        pwl_norm = np.sum(np.arange(1, N+1)**(-self.alpha))
        p_powerlaw = L**(-self.alpha)/pwl_norm 

        n_start = N+1-L
        p_L = 2*(N+1-L)/(N*(N+1))
        p_uniform = p_L / n_start / (lt_from.n - 1)

        p_boundary = 0

        if a == 0:
            p_boundary += p_powerlaw / (lt_from.n - 1)
        elif (a == a_embed and lt_to[a] == lt_from[a-1]):
            p_boundary += p_powerlaw

        if b == len(lt_from):
            p_boundary += p_powerlaw / (lt_from.n - 1)
        elif (b == b_embed and lt_to[b-1] == lt_from[b]):
            p_boundary += p_powerlaw

        p_cf = float(a == a_embed and b == b_embed) / (lt_from.n - 1)

        # Divide p_boundary by 2, since on top of choosing the boundary move,
        # we also have to choose the correct boundary
        p_interval_change = np.sum(np.diff(self.cum_moves, prepend=0)*np.array([p_cf, p_boundary/2, p_uniform]))

        return p_interval_change / N_interval

    def lt_gen_proposal_sample_from(self, lt, nSample=1):
        if np.isinf(nSample):
            nSample = len(lt)**2*(lt.n-1)

        def new_state(n, old):
            s = np.random.choice(n-1)
            if s >= old:
                s += 1
            return s

        states_padded = np.pad(lt.state, (1, 1), constant_values=-1)
        boundaries = np.nonzero(np.diff(states_padded))[0]

        for _ in range(nSample):
            i_interval = np.random.choice(len(boundaries)-1)
            a_embed, b_embed = boundaries[i_interval:(i_interval+2)]
            N = b_embed - a_embed

            r_move = np.random.rand()
            if r_move < self.cum_moves[0]: # cluster flip
                new_lt = deepcopy(lt)
                new_lt.state[a_embed:b_embed] = new_state(lt.n, lt[a_embed])
            elif r_move < self.cum_moves[1]: # boundary move
                p_L = np.cumsum(np.arange(1, N+1)**(-self.alpha))
                p_L /= p_L[-1]
                L = np.nonzero(np.random.rand() < p_L)[0][0] + 1
                
                a, b = a_embed, b_embed
                state = new_state(lt.n, lt[a])
                if np.random.rand() < 0.5: # move left boundary
                    b = a + L
                    if a > 0:
                        state = lt[a-1]
                else: # move right boundary
                    a = b - L
                    if b < len(lt):
                        state = lt[b]

                new_lt = deepcopy(lt)
                new_lt.state[a:b] = state
            else : # uniform triangular
                p_L = np.cumsum(np.flip(np.arange(1, N+1))).astype(float)
                p_L /= p_L[-1]
                L = np.nonzero(np.random.rand() < p_L)[0][0] + 1

                start = a_embed + np.random.choice(N - L + 1)

                new_lt = deepcopy(lt)
                new_lt.state[start:(start+L)] = new_state(lt.n, lt[start])

            yield new_lt
