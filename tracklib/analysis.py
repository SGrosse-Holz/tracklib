import os, sys
from copy import deepcopy

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import scipy.fftpack

from . import util
from .trajectory import Trajectory
from .taggedlist import TaggedList

import mkl
mkl.numthreads = 1
import multiprocessing

def MSD(dataset, giveN=False, memo=True):
    """
    Similar to Trajectory.msd: a memoized method to calculate the ensemble
    MSD. Use memo=False to circumvent memoization.

    Respects selection (see TaggedList)
    """
    msdKnown = hasattr(MSD, 'msdN')
    try:
        tagsHaveChanged = dataset._selection_tags != MSD.lasttagset
    except AttributeError:
        tagsHaveChanged = True

    if not msdKnown or tagsHaveChanged or not memo:
        msdNs = [traj.msd(giveN=True, memo=memo) for traj in dataset]

        maxlen = max(len(msdN[0]) for msdN in msdNs)
        emsd = msdNs[0][0]
        npad = [(0, maxlen-len(emsd))] + [(0, 0) for _ in emsd.shape[2:]]
        emsd = np.pad(emsd, npad, constant_values=0)
        eN = np.pad(msdNs[0][1], npad, constant_values=0)
        emsd *= eN

        for msd, N in msdNs[1:]:
            emsd[:len(msd)] += msd*N
            eN[:len(N)] += N
        emsd /= eN
        MSD.msdN = (emsd, eN)

    MSD.lasttagset = dataset._selection_tags
    if giveN:
        return MSD.msdN
    else:
        return MSD.msdN[0]

def hist_lengths(dataset, **kwargs):
    lengths = [len(traj) for traj in dataset]
    
    if 'bins' not in kwargs.keys():
        kwargs['bins'] = 'auto'

    plt.figure()
    h = plt.hist(lengths, **kwargs)
    plt.title("Histogram of trajectory lengths")
    plt.xlabel("Length in frames")
    plt.ylabel("Count")
    return h

def plot_msds(dataset, **kwargs):
    ensembleLabel = 'ensemble mean'
    if 'label' in kwargs.keys():
        ensembleLabel = kwargs['label']
    kwargs['label'] = None

    plt.figure()
    lines = []
    for traj in dataset:
        msd = traj.msd()
        tmsd = np.arange(len(msd))*traj.dt
        lines.append(plt.loglog(tmsd, msd, **kwargs))
    msd = MSD(dataset)
    tmsd = np.arange(len(msd))*dataset._data[0].dt # NOTE: we're assuming that all trajectories have the same dt!
    lines.append(plt.loglog(tmsd, msd, color='k', linewidth=2, label='ensemble mean'))
    plt.legend()

    if dataset._selection_tags == {'_all'}:
        plt.title('MSDs')
    else:
        plt.title('MSDs for tags {}'.format(str(dataset._selection_tags)))
    plt.xlabel("time") # TODO: How do we get a unit description here?
    plt.ylabel("MSD")
    
    return lines

def plot_trajectories(dataset, **kwargs):
    """
    Notes:
     - trajectories will be colored by one of the tags they're associated
       with.
     - here we have to handle selection manually, because the order might
       be relevant. Consequently, this function listens to the kwargs tags
       and logic. If neither are provided, the selection will be used. If
       only tags is provided, logic=any will be used.

    """
    # Input processing 1 : augment the selection process
    (tags, logic) = dataset.getTagsAndLogicFromKwargs(kwargs)
    tags = dataset.makeTagsList(tags)

    # Input processing 2 : actually react to inputs
    flags = {'_all in tags' : False, 'single tag' : False}
    if '_all' in tags:
        tags = list(dataset.tagset() - {'_all'})
        flags['_all in tags'] = True
    if len(tags) == 1:
        colordict = {tags[0] : None}
        flags['single tag'] = True
    else:
        try:
            if isinstance(kwargs['color'], list):
                colors = kwargs['color']
            else:
                colors = [kwargs['color']]
        except KeyError:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        colordict = {tag : colors[i%len(colors)] for i, tag in enumerate(tags)}

    # Plotting
    plt.figure()
    lines = []
    for traj, trajtags in dataset.byTag(tags, logic=logic, giveTags=True):
        for mytag in trajtags & set(tags):
            break # Cheat to get some tag for the trajectory (out of the given ones)
        kwargs['color'] = colordict[mytag]
        lines.append(traj.plot_spatial(**kwargs))

    # Need some faking for the legend
    x0 = sum(plt.xlim())/2
    y0 = sum(plt.ylim())/2
    for tag in tags:
        kwargs['color'] = colordict[tag]
        kwargs['label'] = tag
        if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                kwargs['linestyle'] = kwargs['linestyle'][0]
        plt.plot(x0, y0, **kwargs)

    # Control appearance
    if not flags['single tag']:
        plt.legend()

    if flags['single tag']:
        plt.title('Trajectories for tag "{}"'.format(tags[0]))
    elif flags['_all in tags']:
        plt.title("Trajectories by tag")
    else:
        plt.title("Trajectories for tags {}".format(str(tags)))

    # Done
    return lines

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
        for (traj, mytags) in dataset.byTag(tags=dataset._selection_tags, \
                                            logic=dataset._selection_logic, \
                                            giveTags=True):
            newtraj = deepcopy(traj)
            try:
                traces = util.sampleMSD(msd, n=newtraj.N*newtraj.d)
            except np.linalg.LinAlgError:
                raise RuntimeError("Could not generate trajectories from provided (or ensemble) MSD. Try to use something cleaner.")
            newtraj._data = [traces[:, (i*newtraj.d):((i+1)*newtraj.d)] for i in range(newtraj.N)]
            yield (newtraj, deepcopy(mytags))

    return type(dataset).generate(gen())

def KLD_PC(dataset, n=10, k=20, dt=1):
    """
    Apply the KLD estimator presented by (Perez-Cruz, 2008). We reduce the
    bias of the estimator by randomly choosing half the snippets for
    estimation of the densities and then sample at the other half.

    Input
    -----
    n : integer
        snippet length ( = window size)
        default: 10
    k : integer
        order of nearest neighbor for the estimator
        default: 20
    dt : integer
        number of frames between two data points in a snippet.
        default: 1

    Output
    ------
    Dest : estimated KLD

    Notes
    -----
    This function flattens snippets, i.e. if the trajectory has 2 loci and 3
    dimensions, the KLD estimation will be run in 6n-dimensional space. Since
    this might not be the desired behavior, the user might have to do some
    pre-processing.
    For more advanced use, refer to class KLDestimator.
    """
    # Check that the trajectory format is homogeneous
    if not dataset.isHomogeneous():
        raise ValueError("Cannot calculate KLD on an inhomogenous dataset")

    # Generate snippets
    snips = []
    for traj in dataset:
        newsnips = [traj[start:(start+(n*dt)):dt].flatten() for start in range(len(traj)-(n*dt)+1)]
        snips += [snip for snip in newsnips if not np.any(np.isnan(snip))]
    snips = np.array(snips)

    # DCT seems to speed up neighbor search. Analytically it is irrelevant, as
    # long as normalized.
    snips = scipy.fftpack.dct(snips, axis=1, norm='ortho')

    # Split in two halves for estimation/sampling
    ind = random.sample(range(len(snips)), len(snips))
    halfN = np.ceil(len(snips)/2).astype(int)

    estimation_snips = snips[ind[:halfN]]
    sample_snips = snips[ind[halfN:]]

    # Build neighbor trees and run estimation
    # Note that time reversal in DCT space means multiplying all odd modes by -1
    tree_fw = KDTree(estimation_snips)
    tree_bw = KDTree(estimation_snips*[(-1)**i for i in range(estimation_snips.shape[1])])

    rk = tree_fw.query(sample_snips, k)[0][:, -1]
    sk = tree_bw.query(sample_snips, k)[0][:, -1]
    return n * np.mean(np.log(sk/rk))

class KLDestimator:
    """
    A wrapper class for KLD estimation. Facilitates pre-processing,
    bootstrapping and plotting.
    """
    def __init__(self, dataset, copy=True):
        if copy:
            self.ds = deepcopy(dataset)
        else:
            self.ds = dataset

        self.bootstraprepeats = 20
        self.processes = 16
        self.KLDkwargs = {'n' : 10, 'k' : 20, 'dt' : 1}

    def setup(self, **kwargs):
        """
        Set up the environment/parameters for running the estimation.

        Input
        -----
        bootstraprepeats : integer
            how often to repeat each run with a different partition of the data
            set.
            default: 20
        processes : integer
            how many processes to use.
            default: 16
        other keyword arguments :
            the parameters for KLD_PC. Anything given as a list will be
            sweeped.

        Notes
        -----
        The default values for everything are set in __init__(), so you can
        call this method also to change specific values while keeping
        everything else the same.
        """
        for key in ['bootstraprepeats', 'processes']:
            try:
                setattr(self, key, kwargs[key])
                del kwargs[key]
            except KeyError:
                pass

        for key in kwargs.keys():
            if isinstance(kwargs[key], list):
                self.KLDkwargs[key] = kwargs[key]
            else:
                self.KLDkwargs[key] = [kwargs[key]]

    def useRelative(self):
        """
        Make the estimator run on the relative displacement (d-dim) of two loci.
        """
        self.ds.apply(lambda traj : traj.relativeTrajectory())

    def useDistance(self):
        """
        Make the estimator run on the (scalar) relative distance between two loci.
        """
        self.ds.apply(lambda traj : traj.relativeTrajectory().absTrajectory())

    @staticmethod
    def _parfun(args):
        """
        args should be a composite dict:
         - it should have an entry 'randomseed'
         - the rest will be passed to KLD_PC as keyword arguments
        """
        random.seed(args['randomseed'])
        del args['randomseed']
        self = args['self']
        del args['self']
        return KLD_PC(self.ds, **args)

    def run(self):
        """
        Run the estimation. Remember to setup()

        Output
        ------
        A list of tuples (n, k, dt, KLD) for each of the runs.

        Notes
        -----
        As of now, the data is copied to every child process. Maybe this could
        be improved
        """
        # Assemble args
        argslist = [{'self' : self, 'n' : n, 'k' : k, 'dt' : dt, 'randomseed' : random.getrandbits(64)} \
                    for n in self.KLDkwargs['n'] \
                    for k in self.KLDkwargs['k'] \
                    for dt in self.KLDkwargs['dt'] \
                    for _ in range(self.bootstraprepeats)]

        if self.processes == 1:
            Draw = map(KLDestimator._parfun, argslist)
        else:
            with multiprocessing.Pool(self.processes) as mypool:
                Draw = mypool.map(KLDestimator._parfun, argslist)

        return [(args['n'], args['k'], args['dt'], D) for args, D in zip(argslist, Draw)]
