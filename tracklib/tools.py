import os,sys
from copy import deepcopy

import numpy as np

from .trajectory import Trajectory
from .taggedlist import TaggedList
from . import analysis
from . import util

def MSDdataset(msd, N=2, Ts=100*[None], d=3, **kwargs):
    """
    A utility function for generating a dataset of MSD sampled trajectories.

    Input
    -----
    msd : 1d numpy.ndarray
        the MSD to sample from.
    n : integer
        number of trajectories to generate
        default: 100
    N : integer
        number of particles per trajectory
        default: 2
    Ts : list of int
        list of trajectory lengths, i.e. this determines number and length of
        trajectories. Any None entry will be replaced by the maximum possible
        value, len(msd). If there are values bigger than that, raises a
        ValueError.
        default: 100*[None]
    d : integer
        spatial dimension of the trajectories to sample
        default: 3
    other kwargs are forwarded to util.sampleMSD()

    Output
    ------
    A TaggedList of trajectories (with only the trivial tag '_all').
    """
    Tmax = len(msd)
    for i, T in enumerate(Ts):
        if T is None:
            Ts[i] = Tmax
        elif T > Tmax:
            raise ValueError("Cannot sample trajectory of length {} from MSD of length {}".format(T, Tmax))

    # Timing execution of sampleMSD indicates that as long as we have a
    # reasonable distribution of lengths (e.g. exponential with mean 10% of
    # len(msd)), sampling all traces at the same time is faster, even though we
    # might generate a bunch of unused data.
    kwargs['n'] = len(Ts)*N*d
    traces = util.sampleMSD(msd, **kwargs)

    def gen():
        for iT, T in enumerate(Ts):
            mytraces = [traces[:T, ((iT*N + n)*d):((iT*N + n+1)*d)] for n in range(N)]
            yield (Trajectory.fromArray(mytraces), [])

    return TaggedList.generate(gen())

def MSDcontrol(dataset, msd=None):
    """
    Generate a sister data set where each trajectory is sampled from a
    stationary Gaussian process with MSD equal to the ensemble mean of the
    given data set or the explicitly given MSD. Note generation from
    experimental data (i.e. the ensemble mean) does not always work, because
    that is noisy. Thus the option to provide a cleaned version.
    """
    if msd is None:
        msd = MSD(dataset)

    def gen():
        for (traj, mytags) in dataset(giveTags=True):
            newtraj = deepcopy(traj)
            try:
                traces = util.sampleMSD(msd, n=newtraj.N*newtraj.d)
            except np.linalg.LinAlgError:
                raise RuntimeError("Could not generate trajectories from provided (or ensemble) MSD. Try to use something cleaner.")
            newtraj._data = [traces[:, (i*newtraj.d):((i+1)*newtraj.d)] for i in range(newtraj.N)]
            yield (newtraj, deepcopy(mytags))

    return TaggedList.generate(gen())
