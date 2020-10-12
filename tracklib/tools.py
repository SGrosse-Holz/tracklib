# Tools that make use of the library, such as a function to generate an
# MSDsampled dataset

import os,sys
from copy import deepcopy

import numpy as np

from .trajectory import Trajectory
from .taggedlist import TaggedList
from . import analysis
from . import util

def MSDdataset(msd, N=2, Ts=None, d=3, **kwargs):
    """
    Generate a dataset of MSD sampled trajectories.

    Input
    -----
    msd : 1d numpy.ndarray
        the MSD to sample from.
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
    if Ts is None:
        Ts = 100*[None]

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

def MSDcontrol(dataset, msd=None, setMean='copy'):
    """
    Generate a sister data set where each trajectory is sampled from a
    stationary Gaussian process with MSD equal to the ensemble mean of the
    given data set or the explicitly given MSD. Note generation from
    experimental data (i.e. the ensemble mean) does not always work, because
    that is noisy. Thus the option to provide a cleaned version.

    The mean of each trajectory will be set to coincide with the mean of the
    sister trajectory it is generated from.

    Input
    -----
    dataset : TaggedList of Trajectory
        the dataset to generate a control for
    msd : (T,) np.ndarray
        the MSD to use for sampling. Note that this will be divided by
        (#loci)x(#dimensions) before sampling scalar traces, matching the usual
        notion of MSD of (e.g.) multidimensional trajectories.

    Output
    ------
    A TaggedList that's an MSD generated sister data set to the input

    Implementation Note
    -------------------
    Should this be merged/unified with MSDdataset?
    """
    if msd is None:
        msd = analysis.MSD(dataset)

    msd /= dataset.getHom('N')*dataset.getHom('d')

    def gen():
        for (traj, mytags) in dataset(giveTags=True):
            try:
                traces = util.sampleMSD(msd[:len(traj)], n=traj.N*traj.d)
            except np.linalg.LinAlgError:
                raise RuntimeError("Could not generate trajectories from provided (or ensemble) MSD. Try to use something cleaner.")
            newdata = np.array([traces[:, (i*traj.d):((i+1)*traj.d)] for i in range(traj.N)])
            newdata += np.mean(traj[:], axis=1, keepdims=True)

            newtraj = Trajectory.fromArray(newdata, **deepcopy(traj.meta))

            if len(traj) != len(newtraj):
                print("traj : {}, newtraj : {} entries".format(len(traj), len(newtraj)))

            yield (newtraj, deepcopy(mytags))

    return TaggedList.generate(gen())
