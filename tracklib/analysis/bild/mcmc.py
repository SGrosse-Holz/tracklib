"""
The MCMC sampler used for inference
"""

import abc

import numpy as np
import scipy.stats

from tracklib.util import mcmc
from .util import Loopingtrace, ParametricFamily
from . import priors

class PartScheme(ParametricFamily, metaclass=abc.ABCMeta):
    """
    "Beefy ParametricFamily" for MCMC sampling
    Similar to the old MCMCScheme
    Note that here we assume that ``get()`` takes a single tuple as argument,
    instead of individual parameters. This makes handling in the MCMC easier.
    Also here the idea is to subclass and implement members, whereas for
    ParametricFamily we would instantiate and simply overwrite the `!get`
    attribute.
    """
    def get(self, params): # pragma: no cover
        return params

    @abc.abstractmethod
    def stepping_probability(self, params_from, params_to):
        raise NotImplementedError # pragma: no cover

    @abc.abstractmethod
    def gen_proposal_sample_from(self, params=None, nSample=1):
        # without params give initial values
        raise NotImplementedError # pragma: no cover

class FixedScheme(PartScheme):
    def __init__(self, family, *fixed_args):
        self.obj = family(*fixed_args)
        self.args = fixed_args

    def get(self, *args):
        return self.obj
    def stepping_probability(self, params_from, params_to):
        return float(all([pf == a and pt == a for pf, pt, a in zip(params_from,
                                                                   params_to,
                                                                   self.args)]))
    def gen_proposal_sample_from(self, params=None, nSample=1):
        return iter(nSample*[self.args])

class GeometricPriorScheme(PartScheme):
    def __init__(self, nStates=2, stepsize=0.1, initial_logq=-1e-5):
        self.nStates = nStates
        self.stepsize = stepsize
        self.initial_logq = initial_logq

    def get(self, logq):
        return priors.GeometricPrior(logq, self.nStates)

    def stepping_probability(self, logq_from, logq_to):
        return scipy.stats.gamma(a = -logq_from/self.stepsize + 1,
                                 scale = self.stepsize).pdf(-logq_to)

    def gen_proposal_sample_from(self, params=None, nSample=1):
        if params is None:
            return iter(nSample * [self.initial_logq])
        else:
            return iter(-scipy.stats.gamma(a = -params/self.stepsize + 1,
                                           scale = self.stepsize).rvs(nSample))

class FullMCMC(mcmc.Sampler):
    def __init__(self, move_weights=(0.1, 1, 1), alpha=2, initialize_from='model'):
        self.cum_moves = np.cumsum(move_weights).astype(float)
        self.cum_moves /= self.cum_moves[-1]
        self.alpha = float(alpha)

        assert initialize_from in {'model', 'random', 'flat'} or isinstance(initialize_from, Loopingtrace)
        self.initialize_from = initialize_from

    def setup(self, traj, model, priorscheme):
        assert traj.d == model.d

        self.traj = traj
        self.model = model
        self.priorscheme = priorscheme

####### Overwrite the necessary mcmc.Sampler functions

    def configure(self,
            min_approaches_to_best_sample=None,
            **kwargs,
            ):
        if 'check_stopping_every' not in kwargs:
            try:
                _ = self.config['check_stopping_every']
            except:
                kwargs['check_stopping_every'] = 1000

        mcmc.Sampler.configure(self, **kwargs)
        if min_approaches_to_best_sample is None:
            if 'min_approaches_to_best_sample' not in self.config:
                self.config['min_approaches_to_best_sample'] = 10
        else:
            self.config['min_approaches_to_best_sample'] = min_approaches_to_best_sample

    def callback_stopping(self, myrun):
        (best_trace, _), max_logL = myrun.best_sample_logL()
        traces = [trace for trace, _ in myrun.samples if trace == best_trace]

        n = 1
        curtrace = traces[0]
        for trace in traces:
            if trace is not curtrace:
                n += 1
                curtrace = trace
        return n >= self.config['min_approaches_to_best_sample']

    def run(self, *args, **kwargs):
        if self.initialize_from == 'model':
            init_trace = self.model.initial_loopingtrace(self.traj)
        elif self.initialize_from == 'random':
            init_trace = Loopingtrace.forTrajectory(self.traj, nStates=self.model.nStates)
            init_trace.state = np.random.choice(self.model.nStates, size=init_trace.state.shape)
        elif self.initialize_from == 'flat':
            init_trace = Loopingtrace.forTrajectory(self.traj, nStates=self.model.nStates)
            init_trace.state[:] = 0
        else:
            init_trace = self.initialize_from.copy()

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
            print()
            print(params[0].state)
            print(proposed[0].state)
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
                new_lt = lt.copy()
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

                new_lt = lt.copy()
                new_lt.state[a:b] = state
            else : # uniform triangular
                p_L = np.cumsum(np.flip(np.arange(1, N+1))).astype(float)
                p_L /= p_L[-1]
                L = np.nonzero(np.random.rand() < p_L)[0][0] + 1

                start = a_embed + np.random.choice(N - L + 1)

                new_lt = lt.copy()
                new_lt.state[start:(start+L)] = new_state(lt.n, lt[start])

            yield new_lt

class FixedSwitchMCMC(FullMCMC):
    def __init__(self, k=10, move_weights=(1, 1), alpha=2):
        super().__init__(move_weights=move_weights, alpha=alpha)

        self.k = k
        self.priorscheme = FixedScheme(lambda k : priors.FixedSwitchPrior(k), k)

    def setup(self, traj, model, priorscheme=None):
        if model.nStates == 3:
            raise ValueError("A three state model with fixed number of switches gives weird topological artifacts, don't do this")

        if priorscheme is not None:
            try:
                self.k = priorscheme.obj.k
            except:
                raise ValueError("expected priorscheme to be FixedScheme of FixedSwitchPrior")
            self.priorscheme = priorscheme

        super().setup(traj, model, self.priorscheme)


        # Put together an initial loopingtrace
        lt = Loopingtrace.forTrajectory(self.traj, nStates=self.model.nStates)
        switches = np.round(np.arange(1, self.k+1)/(self.k+1)*len(lt)).astype(int)
        switches = np.insert(np.append(switches, len(lt)), 0, 0)
        if np.any(switches[1:] == switches[:-1]):
            raise ValueError(f"Cannot impose {self.k} switches on trajectory with {self.traj.valid_frames()} valid frames")

        if self.model.nStates == 2: # parameter space is disconnected, choose 0101... component
            curstate = 0
        else:
            curstate = np.random.choice(self.model.nStates)
        for i in range(1, len(switches)):
            lt[switches[i-1]:switches[i]] = curstate

            oldstate = curstate
            curstate = np.random.choice(self.model.nStates-1)
            if curstate >= oldstate:
                curstate += 1

        self.initialize_from = lt

    def lt_gen_proposal_sample_from(self, lt, nSample=1):
        def new_state(n, old_states):
            old_states = np.unique(old_states) # unique also sorts
            s = np.random.choice(n-len(old_states))
            for old in old_states:
                if s >= old:
                    s += 1
            return s

        boundaries = np.nonzero(np.diff(lt.state))[0] # boundary between i, i+1

        for _ in range(nSample):
            r_move = np.random.rand()
            if lt.n == 2: # Clusterflips make no sense here
                r_move = 0

            if r_move < self.cum_moves[0]: # boundary move

                # find boundary to move
                pick = None
                while pick is None:
                    iboundary = np.random.choice(len(boundaries))
                    boundary = boundaries[iboundary]
                    move_left = np.random.rand() < 0.5 # move left if True, right if False

                    if iboundary == 0 and move_left:
                        N = boundary + 1
                    elif iboundary == len(boundaries) - 1 and not move_left:
                        N = len(lt) - 1 - boundary
                    elif move_left:
                        N = boundary - boundaries[iboundary-1]
                    else:
                        N = boundaries[iboundary+1] - boundary

                    N -= 1 # Stop moving before we abolish the switch

                    if N > 0: # there is space for movement
                        pick = (boundary, move_left, N)

                boundary, move_left, N = pick

                # pick move distance
                p_L = np.cumsum(np.arange(1, N+1)**(-self.alpha))
                p_L /= p_L[-1]
                L = np.nonzero(np.random.rand() < p_L)[0][0] + 1

                # move
                new_lt = lt.copy()
                if move_left:
                    new_lt[(boundary-L+1):(boundary+1)] = new_lt[boundary+1]
                else:
                    new_lt[(boundary+1):(boundary+L+1)] = new_lt[boundary]
                
            else: # cluster flip

                # pick cluster to flip
                cid = np.random.choice(len(boundaries)+1)
                if len(boundaries) == 0:
                    a = 0
                    b = len(lt)
                    old_states = [lt[a]]
                elif cid == 0:
                    a = 0
                    b = boundaries[cid]+1
                    old_states = [lt[a], lt[b]]
                elif cid == len(boundaries):
                    a = boundaries[cid-1]+1
                    b = len(lt)
                    old_states = [lt[a-1], lt[a]]
                else:
                    a = boundaries[cid-1]+1
                    b = boundaries[cid]+1
                    old_states = [lt[a-1], lt[a], lt[b]]

                # flip
                new_lt = lt.copy()
                new_lt[a:b] = new_state(lt.n, old_states)

            yield new_lt

    def lt_stepping_probability(self, lt_from, lt_to):
        assert len(lt_from) == len(lt_to)

        k    = np.count_nonzero(np.diff(lt_from.state))
        k_to = np.count_nonzero(np.diff(  lt_to.state))
        if k != k_to:
            return 0.

        pad_from = np.pad(lt_from.state, (1, 1), constant_values=-1)
        pad_to   = np.pad(  lt_to.state, (1, 1), constant_values=-1)

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
        
        move_probs = np.diff(np.insert(self.cum_moves, 0, 0))
        if lt_from.n == 2:
            move_probs = np.array([1, 0])

        if N == L:
            if a == 0 and b == len(lt_from):
                old_states = [lt_from[a]]
            elif a == 0:
                old_states = [lt_from[a], lt_from[b]]
            elif b == len(lt_from):
                old_states = [lt_from[a-1], lt_from[a]]
            else:
                old_states = [lt_from[a-1], lt_from[a], lt_from[b]]

            n_clusters = k + 1
            return move_probs[1]/n_clusters / (lt_from.n - len(np.unique(old_states)))
        else:
            boundaries = np.nonzero(np.diff(lt_from.state))[0]
            move_count = 2*len(boundaries)
            if boundaries[0] == 0:
                move_count -= 1
            if boundaries[-1] == len(lt_from) - 2:
                move_count -= 1
            move_count -= 2*np.count_nonzero(np.diff(boundaries) == 1)

            p_L = np.arange(1, N)**(-self.alpha) # Note the N -= 1 in gen_proposal
            p_L /= np.sum(p_L)
            return move_probs[0]/move_count * p_L[L-1]
