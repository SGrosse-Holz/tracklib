"""
Fixed-k Evidence by Sampling

Parametrize binary profiles as (s, theta) where
 + Σs = 1, s_i > 0
 + theta ∈ {0, 1}
 + s are interval lengths, as fraction of the whole trajectory. Thus it has k+1 values.
"""

from copy import deepcopy

import numpy as np
from scipy import stats

import tracklib as tl
from tracklib.analysis import bild

### Parametrization ###

def st2profile(s, theta, traj):
    """ s are interval lengths (as fraction of total trajectory length) """
    states = theta*np.ones(len(traj))
    if len(s) > 1:
        switchpos = np.cumsum(s)[:-1]
        
        switches = np.floor(switchpos*(len(traj)-1)).astype(int) + 1 # floor(0.0) + 1 = 1 != ceil(0.0)
        for i in range(1, len(switches)):
            states[switches[i-1]:switches[i]] = theta if i % 2 == 0 else 1-theta
            
        states[switches[-1]:] = theta if len(switches) % 2 == 0 else 1-theta
    
    return bild.Loopingprofile(states)

def dirichlet_methodofmoments(ss, normalized_weights):
    m = normalized_weights @ ss
    v = normalized_weights @ (ss - m[np.newaxis, :])**2
    s = np.mean(m*(1-m)/v) - 1
    return s*m

### Likelihood, i.e. targed distribution ###

def logL(params):
    ((s, theta), (traj, model)) = params
    profile = st2profile(s, theta, traj)
    return model.logL(profile, traj)

def calculate_logLs(ss, thetas, traj, model):
    """ parallel-aware (ordered) """
    todo = zip(ss, thetas)
    todo = zip(todo, len(ss)*[(traj, model)])
    imap = tl.util.parallel._map(logL, todo)
    return np.array(list(imap))

### Proposal distribution ###

def proposal(a, m, s, theta0):
    try:
        return (
            stats.dirichlet(a).pdf(s.T)
            * ( m*theta0 + (1-m)*(1-theta0) )
        )
    except ValueError:
        # dirichlet.pdf got an argument that has a zero somewhere, but its alpha is < 1.
        if len(s.shape) == 1:
            s = s[None, :]
            theta0 = np.asarray(theta0)

        ind = np.any(s[:, a < 1] == 0, axis=1)
        if np.sum(ind) == 0:
            raise RuntimeError("Could not identify 0s in sample")

        out = np.empty(len(s), dtype=float)
        if np.sum(~ind) > 0:
            out[~ind] = proposal(a, m, s[~ind], theta0[~ind])
        out[ind] = np.inf

        return out

# def log_proposal(a, m, s, theta0):
#     return (
#         stats.dirichlet(a).logpdf(s.T)
#         + np.log( m*theta0 + (1-m)*(1-theta0) )
#     )

def sample_proposal(a, m, N):
    ss = stats.dirichlet(a).rvs(N)
    thetas = (np.random.rand(N) < m).astype(int)
    return ss, thetas

def fit_proposal(ss, thetas, weights):
    weights = weights / np.sum(weights)
    a = dirichlet_methodofmoments(ss, weights)
    m = thetas @ weights
    return a, m

### Sampling ###

class FixedkSampler:
    # Potential further improvements:
    #  + make each proposal a mixture of Dirichlet's to catch multimodal behavior
    #  + improve proposal fitting / braking (better than MOM, maybe more something gradient like?)
    def __init__(self, traj, model,
                 k, N=100,
                 concentration_brake=1e-2,
                 polarization_brake=1e-3,
                 max_fev = 20000,
                ):
        self.k = k
        self.N = N
        self.brakes = (concentration_brake, polarization_brake)
        
        self.max_fev = max_fev
        self.exhausted = False
        
        self.traj = traj
        self.model = model
        
        self.parameters = [(np.ones(k+1), 0.5)]
        self.samples = [] # each sample is a list: [s, theta, logL, δ, w], where each entry is an (N, ...) array
        self.evidences = [] # each entry: (logev, dlogev, KL)
        self.max_logL = -np.inf
        
        # Just sample exhaustively for k = 0, 1
        # Note that we assume the parameter space volume n*(n-1)^k * 1 = 2 for n = 2.
        # The Dirichlet distribution, being defined on the simplex, assumes a parameter space
        # volume of 1/k!. This gives a correction term on the evidence!
        # The deltas below should be 1/(Dirichlet parameter space volume) = k!/2.
        if self.k == 0:
            self._fix_exhaustive([np.ones((2, 1)), np.array([0, 1]),
                                  None, 1/2*np.ones(2), None,
                                 ])
        elif self.k == 1:
            switches = np.arange(len(self.traj)-1)/(len(self.traj)-1)
            ss = np.array([switches, 1-switches]).T
            self._fix_exhaustive([np.concatenate(2*[ss], axis=0),
                                  np.concatenate([np.zeros(len(ss)), np.ones(len(ss))], axis=0),
                                  None, 1/2*np.ones(2*len(ss)), None,
                                 ])
            
    def _fix_exhaustive(self, sample):
        sample[2] = calculate_logLs(sample[0], sample[1], self.traj, self.model)
        self.max_logL = np.max(sample[2])
        with np.errstate(under='ignore'):
            Ls = np.exp(sample[2] - self.max_logL)
        sample[4] = Ls / sample[3]
        
        self.samples.append(sample)
        
        ev_offac = np.mean(sample[4]) # offset by a factor exp(max_logL)
        dlogev = 1e-10
        logev = np.log(ev_offac) + self.max_logL + np.sum(np.log(np.arange(self.k)+1))
        with np.errstate(divide='ignore'): # we might get log(0), but in that case KL is just +inf, that's fine
            KL = logev - np.mean(np.log(sample[4]) + self.max_logL)
        self.evidences.append((logev, dlogev, KL))
        
        # Prevent the actual sampling from running
        self.exhausted = True

    def step(self):
        """ returns False if Sampler is exhausted, True otherwise """
        if self.exhausted:
            return False
        
        # Update δ's on old samples
        for sample in self.samples:
            sample[3] += proposal(*self.parameters[-1], sample[0], sample[1])

        # Put together new sample
        sample = 5*[None]
        sample[0], sample[1] = sample_proposal(*self.parameters[-1], self.N)
        sample[2] = calculate_logLs(sample[0], sample[1], self.traj, self.model)

        self.max_logL = max(self.max_logL, np.max(sample[2])) # Keep track of max(logL) such that we can properly convert to weights later

        sample[3] = np.zeros(self.N)
        for a, m in self.parameters:
            sample[3] += proposal(a, m, sample[0], sample[1])

        self.samples.append(sample)

        # Calculate weights for all samples
        for sample in self.samples:
            with np.errstate(under='ignore'):
                Ls = np.exp(sample[2] - self.max_logL)
            sample[4] = Ls / (sample[3] / len(self.parameters))

        # Update proposal
        # full_ensemble is [s, theta, w_offac]
        full_ensemble = [np.concatenate([sample[i] for sample in self.samples], axis=0) for i in [0, 1, 4]]

        old_a, old_m = self.parameters[-1]
        new_a, new_m = fit_proposal(*full_ensemble)

        # Keep concentration from exploding
        concentration_ratio = np.sum(new_a) / np.sum(old_a)
        if np.abs(np.log(concentration_ratio))/self.N > self.brakes[0]:
            logfac = self.N*self.brakes[0]
            if concentration_ratio < 1:
                logfac *= -1

            new_a = old_a * np.exp(logfac)

        # Keep polarization from exploding
        if np.abs(new_m - old_m)/self.N > self.brakes[1]:
            if new_m > old_m:
                new_m = old_m + self.N*self.brakes[1]
            else:
                new_m = old_m - self.N*self.brakes[1]

        self.parameters.append((new_a, new_m))

        # Evidence & KL
        ev_offac = np.mean(full_ensemble[2]) # offset by a factor exp(max_logL)
        dlogev = stats.sem(full_ensemble[2]) / ev_offac
        logev = np.log(ev_offac) + self.max_logL + np.sum(np.log(np.arange(self.k)+1)) # Note k! correction
        
        with np.errstate(divide='ignore'): # we might get log(0), but in that case KL is just +inf, that's fine
            KL = np.log(np.mean(full_ensemble[2])) - np.mean(np.log(full_ensemble[2]))
        
        self.evidences.append((logev, dlogev, KL))
        
        # Check whether we can still sample more in the future
        if len(self.samples)*self.N > self.max_fev:
            self.exhausted = True
        return True
        
    def t_stat(self, other):
        logev0, dlogev0 = self.evidences[-1][:2]
        logev1, dlogev1 = other.evidences[-1][:2]

        effect = logev0 - logev1
        return effect / np.sqrt( dlogev0**2 + dlogev1**2 )
    
    def MLE_profile(self):
        best_logL = -np.inf
        for sample in self.samples:
            i = np.argmax(sample[2])
            if sample[2][i] > best_logL:
                best_logL = sample[2][i]
                s = sample[0][i]
                t = sample[1][i]
                
        return st2profile(s, t, self.traj)
            
### Full iterative scheme ###

def sample(traj, model,
           samples_per_step = 100,
           init_runs = 20,
           significant_separation_sem_fold = 5,
           k_lookahead = 2,
           k_max = 20,
           sampler_kw = {},
           show_progress=False,
           assume_notebook_for_progress_bar=True,
          ):
    """ uses calculate_logLs, which is parallel-aware (ordered) """
    if show_progress:
        if assume_notebook_for_progress_bar:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        bar = tqdm()
        del tqdm
    else:
        bar = lambda : None
        bar.update = lambda *_ : None
        bar.close = lambda *_ : None
    
    # Some conditions that come in handy
    def all_exhausted(xs):
        return all(x.exhausted for x in xs)
    def within_lookahead(k, candidates):
        return k - np.max([sampler.k for sampler in candidates]) <= k_lookahead
    def is_significant(sampler, against, t_thres=significant_separation_sem_fold):
        return np.abs(sampler.t_stat(against)) > t_thres
    
    # Initialize
    samplers = []
    insignificants = []
    k_updates = []
    
    # Body of the insignificance resolution
    def step_highest_sem(samplers, k_updates=k_updates):
        candidates = [sampler for sampler in samplers if not sampler.exhausted]
        i_worst = np.argmax([candidate.evidences[-1][1] for candidate in candidates])
        k_updates.append(candidates[i_worst].k)
        candidates[i_worst].step() # no need to pay attention to return values, this is done by the loop condition
    def get_insignificants(samplers):
        # Note on the selection of insignificants:
        # we pick them such that the whole set is mutually insignificant, as opposed to say picking everything with
        # an insignificant comparison against the best sample. The reason is that if for any k we can find a
        # significantly better one, we should use that and forget about the worse one. Also, comparing only to the
        # best sample is insane when that happens to have large sem and is exhausted already (which is the
        # practical situation that alerted me to this issue).
        evidences = np.array([sampler.evidences[-1][0] for sampler in samplers])
        ks = np.argsort(evidences)[::-1]
        insignificants = []
        for k in ks: # Note that since any([]) == False the first iteration works fine
            if not any(is_significant(samplers[k], other) for other in insignificants):
                insignificants.append(samplers[k])
                
        return insignificants
    
    # Run
    for k in range(k_max+1):
        samplers.append(FixedkSampler(traj, model,
                                      k=k, N=samples_per_step,
                                      **sampler_kw,
                                     ))
        assert len(samplers) == k+1 # Paranoia
        
        # Initial sampling
        for _ in range(init_runs):
            if not samplers[k].step():
                break
            k_updates.append(k)
            bar.update()
            
        # Update significances
        insignificants = get_insignificants(samplers)
        
        # Insignificance resolution
        # Note that we only resolve insignificances if it is "important", i.e. if we have
        # exhausted the lookahead already. This ensures that we don't get trapped resolving
        # irrelevant insignificances, i.e. those away from the optimum
        while (not within_lookahead(k+1, insignificants)
               and not all_exhausted(insignificants)
               and len(insignificants) > 1
              ):
            step_highest_sem(insignificants)
            bar.update()
            insignificants = get_insignificants(samplers)
            
        # If the next iteration would exceed the lookahead, stop
        if not within_lookahead(k+1, insignificants):
            break

    # There might be insignificancies left, in which case we will continue sampling until
    #  a) they are resolved, or
    #  b) we exhaust all the samplers
    while (len(insignificants) > 1
           and not all_exhausted(insignificants)
          ):
        step_highest_sem(insignificants)
        bar.update()
        insignificants = get_insignificants(samplers)
        
    bar.close()
    
    # Assemble output
    out = {}
    out['samplers'] = samplers
    out['k_updates'] = k_updates
    out['ks'] = np.array([sampler.k for sampler in samplers])
    out['evidence'] = np.array([sampler.evidences[-1][0] for sampler in samplers])
    out['evidence_se'] = np.array([sampler.evidences[-1][1] for sampler in samplers])
    out['best_k'] = np.argmax(out['evidence'])
    out['profile'] = samplers[out['best_k']].MLE_profile()
    return out

### Post-processing: gradient descent of switch positions ###

def logLR_boundaries(profile, traj, model):
    def likelihood(profile, traj=traj, model=model):
        return model.logL(profile, traj)
    
    boundaries = np.nonzero(np.diff(profile.state))[0] # boundary between b and b+1
    if len(boundaries) == 0:
        return np.array([])
    
    profile_new = profile.copy()
    Ls = np.empty((len(boundaries), 2))
    Ls[:] = np.nan
    for i, b in enumerate(boundaries):
        old_states = deepcopy(profile_new[b:(b+2)])
        
        # move left
        profile_new[b] = profile_new[b+1]
        Ls[i, 0] = likelihood(profile_new)
        profile_new[b] = old_states[0]
        
        # move right
        profile_new[b+1] = profile_new[b]
        Ls[i, 1] = likelihood(profile_new)
        profile_new[b+1] = old_states[1]
        
    return Ls - likelihood(profile)

def optimize_boundary(profile, traj, model,
                      max_iteration = 10000,
                     ):
    profile_new = profile.copy()
    for _ in range(max_iteration):
        logLR = logLR_boundaries(profile_new, traj, model)
        if len(logLR) == 0:
            break
            
        i, j = np.unravel_index(np.argmax(logLR), logLR.shape)
        
        if logLR[i, j] > 0:
            boundaries = np.nonzero(np.diff(profile_new.state))[0]
            if profile_new[boundaries[i]+j] == profile_new[boundaries[i]+(1-j)]:
#                 raise RuntimeError(f"Trying to abolish boundary at {boundaries[i]}")
                return profile_new

            profile_new[boundaries[i]+j] = profile_new[boundaries[i]+(1-j)]
        else:
            break
    else:
        raise RuntimeError(f"Exceeded max_iteration = {max_iteration}")
        
    return profile_new
