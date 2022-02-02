"""
A module collecting some useful post-processing functions.
"""

import itertools
from copy import deepcopy

import numpy as np

from .util import Loopingtrace

MAX_ITERATION = 1000

def logLR_clusterflip(lt, traj, model, prior):
    """
    Likelihood ratio for clusterflips

    Calculate the log likelihood ratio for clusterflips, i.e. replacing whole
    intervals with a different state

    Parameters
    ----------
    lt : Loopingtrace
        the profile to start from
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation

    Returns
    -------
    (N, n) np.ndarray
        the likelihood ratios. ``N`` is the number of contiguous intervals in
        the `Loopingtrace`, ``n = lt.n - 1`` is the number of alternative
        states each interval could have.
    """
    def likelihood(lt):
        return model.logL(lt, traj) + prior.logpi(lt)

    loops = lt.loops(return_='index')

    lt_new = lt.copy()
    Ls = np.zeros((len(loops), lt.n-1))
    for i, loop in enumerate(loops):
        for j, new_state in enumerate(list(range(loop[2])) + list(range(loop[2]+1, lt.n))):
            lt_new.state[loop[0]:loop[1]] = new_state
            Ls[i, j] = likelihood(lt_new)

        lt_new.state[loop[0]:loop[1]] = loop[2]

    return Ls - likelihood(lt)

def optimize_clusterflip(lt, traj, model, prior):
    """
    Deterministic optimization using (only) clusterflips

    We evaluate `logLR_clusterflip` and perform the (one) best flip it found
    (if ``log(LR) > 0``). Then repeat until no improvement possible.

    Parameters
    ----------
    lt : Loopingtrace
        the profile to be optimized
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation

    Returns
    -------
    Loopingtrace
        a new, optimized `Loopingtrace`
    """
    lt_new = lt.copy()
    for _ in range(MAX_ITERATION):
        logLR = logLR_clusterflip(lt_new, traj, model, prior)
        i, j = np.unravel_index(np.argmax(logLR), logLR.shape)
        if logLR[i, j] > 0:
            loop = lt_new.loops(return_='index')[i]
            lt_new.state[loop[0]:loop[1]] = j if j < loop[2] else j+1
        else:
            break
    else: # pragma: no cover
        raise RuntimeError("Exceeded MAX_ITERATION ({})".format(MAX_ITERATION))

    return lt_new

def logLR_nbit(lt, traj, model, prior, n):
    """
    Give likelihood ratios for exhaustive resampling of n-bit moving window

    Parameters
    ----------
    lt : Loopingtrace
        the profile to start from
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation
    n : int
        the window size, i.e. number of contiguous bits to sample exhaustively

    Returns
    -------
    (N, n) np.ndarray
        the likelihood ratios, where ``N = len(lt)-n+1`` and ``n = (lt.n)^n`` (we
        insert the value of the unchanged window for completeness).

    Note
    ----
    This scales as ``O(len(lt) * lt.n ^ n)``, i.e. becomes very expensive very
    fast.
    """
    def likelihood(lt):
        return model.logL(lt, traj) + prior.logpi(lt)

    L_ref = likelihood(lt)
    
    lt_new = lt.copy()
    Ls = np.zeros((len(lt)-n+1, lt.n**n))
    window_vals = list(itertools.product(range(lt.n), repeat=n))

    for i in range(len(lt)-n+1):
        cur_val = deepcopy(lt_new[i:(i+n)])
        cur_j = np.sum(cur_val * lt.n**(np.flip(np.arange(n))))

        for j, window_val in enumerate(window_vals):
            if j == cur_j:
                Ls[i, j] = L_ref
            else:
                lt_new.state[i:(i+n)] = window_val
                Ls[i, j] = likelihood(lt_new)

        lt_new.state[i:(i+n)] = cur_val

    return Ls - L_ref

def optimize_nbit(lt, traj, model, prior, n):
    """
    Deterministic optimization using n-bit flips

    We evaluate `logLR_nbit` and perform the (one) best flip it found (if
    ``log(LR) > 0``). Then repeat until no improvement possible.

    Parameters
    ----------
    lt : Loopingtrace
        the profile to be optimized
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation
    n : int
        the window size, i.e. number of contiguous bits to sample exhaustively

    Returns
    -------
    Loopingtrace
        a new, optimized `Loopingtrace`
    """
    lt_new = lt.copy()
    for _ in range(MAX_ITERATION):
        logLR = logLR_nbit(lt_new, traj, model, prior, n)
        i, j = np.unravel_index(np.argmax(logLR), logLR.shape)

        if logLR[i, j] > 0:
            window_val = np.zeros(n)
            for k in range(n):
                j, it = divmod(j, lt_new.n)
                window_val[n-k-1] = it

            lt_new.state[i:(i+n)] = window_val
        else:
            break
    else: # pragma: no cover
        raise RuntimeError("Exceeded MAX_ITERATION ({})".format(MAX_ITERATION))

    return lt_new

def logLR_boundary(lt, traj, model, prior):
    """
    Give likelihood ratios for boundary moves

    Parameters
    ----------
    lt : Loopingtrace
        the profile to start from
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation

    Returns
    -------
    (N, 2) np.ndarray
        the likelihood ratios, where ``N`` is the number of boundaries within
        the profile and the two entries are the likelihood ratios for moving
        left and right respectively.
    """
    def likelihood(lt):
        return model.logL(lt, traj) + prior.logpi(lt)

    boundaries = np.nonzero(np.diff(lt.state))[0] # boundary is between b and b+1
    if len(boundaries) == 0:
        return np.array([])

    lt_new = lt.copy()
    Ls = np.zeros((len(boundaries), 2))
    for i, b in enumerate(boundaries):
        old_states = deepcopy(lt_new[b:(b+2)])

        # try to move left
        lt_new[b] = lt_new[b+1]
        Ls[i, 0] = likelihood(lt_new)
        lt_new[b] = old_states[0]

        # try to move right
        lt_new[b+1] = lt_new[b]
        Ls[i, 1] = likelihood(lt_new)
        lt_new[b+1] = old_states[1]

    return Ls - likelihood(lt)

def optimize_boundary(lt, traj, model, prior):
    """
    Deterministic optimization using boundary moves

    We evaluate `logLR_boundary` and perform the (one) best boundary move it
    found (if ``log(LR) > 0``). Then repeat until no improvement possible.

    Parameters
    ----------
    lt : Loopingtrace
        the profile to be optimized
    traj, model, prior: Trajectory, Model, Prior
        the context for likelihood calculation

    Returns
    -------
    Loopingtrace
        a new, optimized `Loopingtrace`
    """
    lt_new = lt.copy()
    for _ in range(MAX_ITERATION):
        logLR = logLR_boundary(lt_new, traj, model, prior)
        if len(logLR) == 0:
            break

        i, j = np.unravel_index(np.argmax(logLR), logLR.shape)

        if logLR[i, j] > 0:
            boundaries = np.nonzero(np.diff(lt_new.state))[0] # boundary is between b and b+1
            lt_new[boundaries[i]+j] = lt_new[boundaries[i]+(1-j)]
        else:
            break
    else: # pragma: no cover
        raise RuntimeError("Exceeded MAX_ITERATION ({})".format(MAX_ITERATION))

    return lt_new
