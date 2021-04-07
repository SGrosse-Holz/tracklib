from copy import deepcopy

import random
import numpy as np
from sklearn.neighbors import KDTree
import scipy.fftpack

from tracklib import Trajectory, TaggedSet

def perezcruz(dataset, n=10, k=20, dt=1):
    """
    Apply the KLD estimator presented by (Perez-Cruz, 2008).
    
    We reduce the bias of the estimator by randomly choosing half the snippets
    for estimation of the densities and then sample at the other half.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the data to run the KLD estimation on. Note that the trajectories
        should have the ``meta['parity']`` field set (to either ``'even'`` or
        ``'odd'``)
    n : int, optional
        snippet length ( = window size)
    k : int, optional
        order of nearest neighbor for the estimator
    dt : int, optional
        number of frames between two data points in a snippet.
    parity : {'even', 'odd'}, optional
        the parity of the trajectories in `!dataset` under time reversal.

    Returns
    -------
    Dest : float
        estimated KLD in nats

    See also
    --------
    tracklib.util.sweep.Sweeper

    Notes
    -----
    This function flattens snippets, i.e. if the trajectory has 2 loci and 3
    dimensions, the KLD estimation will be run in 6`!n`-dimensional space.
    Since this might not be the desired behavior, the user might have to do
    some pre-processing.
    """
    parity = dataset.map_unique(lambda traj : traj.meta['parity'])
    assert parity in {'even', 'odd'}

    # Generate snippets
    snips = []
    for traj in dataset:
        newsnips = [traj[start:(start+(n*dt)):dt] for start in range(len(traj)-(n*dt)+1)]
        snips += [snip for snip in newsnips if not np.any(np.isnan(snip))]
    snips = np.array(snips)

    if dataset.map_unique(lambda traj : traj.N) > 1: # pragma: no cover
        snips = snips.swapaxes(2, 1)
    snips = snips.reshape((snips.shape[0], snips.shape[1], -1))

    # DCT seems to speed up neighbor search. Analytically it is irrelevant, as
    # long as normalized and we account for the switching parity of the
    # components (see below)
    snips = scipy.fftpack.dct(snips, axis=1, norm='ortho')

    # Split in two halves for estimation/sampling
    ind = random.sample(range(len(snips)), len(snips))
    halfN = np.ceil(len(snips)/2).astype(int)

    estimation_snips = snips[ind[:halfN]]
    sample_snips = snips[ind[halfN:]]

    # Note that time reversal in DCT space means multiplying all odd modes by -1
    rev_estimation_snips = estimation_snips * ((-1)**np.arange(estimation_snips.shape[1])).reshape(1, -1, 1)
    if parity == 'odd':
        rev_estimation_snips *= -1

    # Build neighbor trees and run estimation
    N_snips = estimation_snips.shape[0]
    tree_fw = KDTree(estimation_snips.reshape(N_snips, -1))
    tree_bw = KDTree(rev_estimation_snips.reshape(N_snips, -1))

    N_snips = sample_snips.shape[0]
    rk = tree_fw.query(sample_snips.reshape(N_snips, -1), k)[0][:, -1]
    sk = tree_bw.query(sample_snips.reshape(N_snips, -1), k)[0][:, -1]
    return np.prod(snips.shape[1:]) * np.mean(np.log(sk/rk))
