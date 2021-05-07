"""
Formalizing the process of running `util.mcmc`
"""

import abc
from copy import deepcopy

import numpy as np

from tracklib.util import mcmc
from .util import Loopingtrace

class MCMCRun:
    """
    The results of an MCMC run

    Attributes
    ----------
    logLs : np.ndarray
        the likelihoods from the run
    samples : list of <sample data type>
        the actual MCMC sample
    """
    def __init__(self, logLs=None, samples=None):
        self.logLs = deepcopy(logLs)
        self.samples = deepcopy(samples)

    def logLs_trunc(self):
        """
        Give only the likelihoods associated with the samples

        `tracklib.util.mcmc.Sampler.run` returns likelihoods starting at the
        first iteration, but samples only after a given burn-in period. This
        function cuts this initial overhang from the likelihood array

        Returns
        -------
        (len(samples),) np.ndarray
        """
        return self.logLs[-len(self.samples):]

    def best_sample_L(self):
        """
        Give the best sample (and likelihood)

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

    def acceptance_rate(self, criterion='sample_identity'):
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

        Parameters
        ----------
        criterion : {'sample_equality', 'sample_identity', 'likelihood_equality'}
            which method to use to determine whether a step was accepted or
            rejected.

        Returns
        -------
        float
            the acceptance rate
        """
        if criterion == 'sample_equality':
            n_reject = np.sum([1 if sample0 == sample1 else 0
                               for sample0, sample1 in zip(self.samples[:-1], self.samples[1:])
                               ])
        elif criterion == 'sample_identity':
            n_reject = np.sum([1 if sample0 is sample1 else 0
                               for sample0, sample1 in zip(self.samples[:-1], self.samples[1:])
                               ])
        elif criterion == 'likelihood_equality':
            logLs = self.logLs_trunc()
            n_reject = np.sum(logLs[:-1] == logLs[1:])

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

class MCMCScheme(mcmc.Sampler, metaclass=abc.ABCMeta):
    """
    Abstract base class for `Loopingtrace` MCMC schemes.

    We might want to test different MCMC schemes, so this class intends to
    reduce the overhead of introducing a new sampling scheme to a minimum. A
    sampling scheme is given by its proposal distribution, so this is the
    minimum information needed. This should be specified in two forms:
    `stepping_probability` evaluates the actual proposal distribution, while
    `gen_proposal_sample_from` should yield samples from the proposal
    distribution. While `propose_update` in principle can be assembled from
    these (and is by default), it usually makes sense to implement this special
    case separately, since it might save computational time (significantly).

    Attributes
    ----------
    traj : Trajectory
        the trajectory we want to find `Loopingtraces <Loopingtrace>` for
    model : models.Model
        the inference model to use
    prior : priors.Prior
        prior over `Looingtraces <Loopingtrace>`

    Example
    -------
    Assume we implemented some sampling scheme ``MyScheme`` and have ``traj``,
    ``model``, and ``prior`` defined

    >>> MCMCconfig = {
    ...     'iterations' : 1000,
    ...     'burn_in'    :  100,
    ...     }
    ... mcmc = MyScheme()
    ... mcmc.setup(traj, model, prior)
    ... mcmc.configure(**MCMCconfig)
    ... res = mcmc.run()
    """
    def setup(self, traj, model, prior):
        """
        Set up everything

        Parameters
        ----------
        traj : Trajectory
            the trajectory we want to find `Loopingtraces <Loopingtrace>` for
        model : models.Model
            the inference model to use
        prior : priors.Prior
            prior over `Looingtraces <Loopingtrace>`
        """
        self.traj = traj
        self.model = model
        self.prior = prior

    def run(self, *args, **kwargs):
        """
        Run the MCMC sampling and store results in an `MCMCRun`

        The `model` will be queried for an initial `Loopingtrace`.

        All arguments are forwarded to `tracklib.util.mcmc.Sampler.run`, which
        at the time of writing does not take any more arguments than the
        initial profile.

        Returns
        -------
        MCMCRun
        """
        res = MCMCRun()
        res.logLs, res.samples = mcmc.Sampler.run(self,
                                                 self.model.initial_loopingtrace(self.traj),
                                                 *args, **kwargs)
        return res

    @staticmethod
    def acceptance_probability(L_from, L_to):
        """
        Calculate acceptance probability from two likelihoods

        This is given by ``min(1, exp(L_to - L_from))``. This function exists
        mostly for stylistic reasons.

        Parameters
        ----------
        L_from, L_to : float
            log-likelihoods of the states we are moving from and to

        Returns
        -------
        float
        """
        # generic formula, do not change
        with np.errstate(over='ignore', under='ignore'):
            return np.minimum(1, np.exp(L_to - L_from))

    @staticmethod
    def likelihood(traj, loopingtrace, model, prior):
        """
        Evaluate the likelihood of the (loopingtrace, model, prior) combination

        Parameters
        ----------
        loopingtrace : Loopingtrace
        model : models.Model
        priors : priors.Prior

        Returns
        -------
        float
        """
        return model.logL(loopingtrace, traj) + prior.logpi(loopingtrace)

    def logL(self, loopingtrace):
        return self.likelihood(self.traj, loopingtrace, self.model, self.prior)

    @abc.abstractmethod
    def stepping_probability(self, loopingtrace_from, loopingtrace_to):
        """
        Evaluate the proposal distribution

        Parameters
        ----------
        loopingtrace_from : Loopingtrace
            the "current" `Loopingtrace`
        loopingtrace_to : Loopingtrace
            another `Loopingtrace`

        Returns
        -------
        float
            the probability that `!loopingtrace_to` was proposed, given that we
            are currently at `!loopingtrace_from`.

        See also
        --------
        gen_proposal_sample_from
        """
        raise NotImplementedError # pragma: no cover

    @abc.abstractmethod
    def gen_proposal_sample_from(self, loopingtrace, nSample=float('inf')):
        """
        Sample from the proposal distribution

        This function returns a generator yielding `!nSample` samples from the
        proposal distribution around `!loopingtrace`.

        Parameters
        ----------
        loopingtrace : Loopingtrace
        nSample : float, optional
            how many samples to draw. The default value ``float('inf')``
            indicates exhaustive sampling of the distribution

        Yields
        ------
        Loopingtrace
            samples from the proposal distribution

        See also
        --------
        stepping_probability
        """
        # NOTE: this has to be an actual sample from the distribution. We'll
        # use it to replace an average value with mean over this ensemble.
        raise NotImplementedError # pragma: no cover

    def propose_update(self, loopingtrace_cur):
        # Should be overridden, but it is possible to leave it like this
        proposed = next(self.gen_proposal_sample_from(loopingtrace_cur, nSample=1))
        return (proposed,
                self.stepping_probability(loopingtrace_cur, proposed),
                self.stepping_probability(proposed, loopingtrace_cur),
                )

class TPWMCMC(MCMCScheme):
    """
    A time-point wise sampling scheme

    Probably the most straight-forward sampling scheme. At each step, randomly
    assign a new state for a single random frame.
    """
    # NOTE: the following three functions are all different representations of
    # the sampling scheme, so they *all* have to be updated when changing
    # anything about that!

    def stepping_probability(self, loopingtrace_from, loopingtrace_to):
        nNeighbors = len(loopingtrace_from.t)*(loopingtrace_from.n-1)

        if (np.all(loopingtrace_from.t == loopingtrace_to.t) and
            np.sum(loopingtrace_from[:] != loopingtrace_to[:]) == 1):
            return 1/nNeighbors
        else:
            return 0

    def gen_proposal_sample_from(self, loopingtrace, nSample=float('inf')):
        nNeighbors = len(loopingtrace.t)*(loopingtrace.n-1)
        if nSample == float('inf'):
            nSample = nNeighbors
        sample_ids = np.random.choice(nNeighbors, size=(nSample,), replace=(nSample>nNeighbors))

        for sid in sample_ids:
            i_update = int(sid // (loopingtrace.n-1))
            new_val = int(sid % (loopingtrace.n-1))
            if new_val >= loopingtrace[i_update]:
                new_val += 1

            neighbor = loopingtrace.copy()
            neighbor[i_update] = new_val
            yield neighbor

    def propose_update(self, loopingtrace_cur):
        # This should be faster than MCMCScheme.propose_update
        nNeighbors = len(loopingtrace_cur.t)*(loopingtrace_cur.n-1)

        loopingtrace_prop = loopingtrace_cur.copy()
        ind_up = np.random.randint(len(loopingtrace_prop))
        cur_val = loopingtrace_prop[ind_up]
        loopingtrace_prop[ind_up] = np.random.choice(list(range(cur_val))+list(range(cur_val+1, loopingtrace_prop.n)))

        return loopingtrace_prop, 1./nNeighbors, 1./nNeighbors

class TriMCMC(MCMCScheme):
    """
    An interval oriented sampling scheme

    For each proposal select one interval in the current trace and change some
    part of it. We use three different moves:
     + cluster flip : choose a new value for the whole interval
     + boundary move : move one of the interval boundaries. How far the move
        goes is governed by a power law
     + uniform triangular : select a random sub-interval ``[a, b)`` to change.
        The tuple ``(a, b)`` comes from a uniform distribution over the
        triangle ``0 <= a < b <= N``, where ``N`` is the length of the current
        interval.

    The probabilities for the different moves are governed by the weights
    handed to the initializer.

    Parameters
    ----------
    weights : array-like, optional
        the weights with which to choose the different moves. Order is
        ``(cluster flip, boundary move, uniform triangular)``.  Note that the
        triangular sampling in principle also includes the other two moves.
    boundary_alpha : float, optional
        the exponent in the power law for boundary moves. We recommend ``alpha
        > 1`` such that most moves are short.
    """
    def __init__(self, weights=(1, 1, 1), boundary_alpha=1):
        self.cum_moves = np.cumsum(weights).astype(float)
        self.cum_moves /= self.cum_moves[-1]
        self.boundary_alpha = float(boundary_alpha)

    def stepping_probability(self, lt_from, lt_to):
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
        p_L = 2*(N+1-L)/(N*(N+1))
        n_start = N+1-L
        p_uniform = p_L / n_start / (lt_from.n - 1)

        boundary_norm = np.sum(np.arange(1, N+1)**(-self.boundary_alpha))
        p_powerlaw = L**(-self.boundary_alpha)/boundary_norm 
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
#         print(f"({a}, {b}) c ({a_embed}, {b_embed})")
#         print(p_cf, p_boundary, p_uniform)
        p_interval_change = np.sum(np.diff(self.cum_moves, prepend=0)*np.array([p_cf, p_boundary/2, p_uniform]))

        return p_interval_change / N_interval

    def gen_proposal_sample_from(self, lt, nSample=float('inf')):
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
                p_L = np.cumsum(np.arange(1, N+1)**(-self.boundary_alpha))
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
