import os, sys
from copy import deepcopy
import itertools

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

    Implementation note: is the memoization here really necessary? The costly
    part is calculating MSDs from trajectories and that is memoized in the
    Trajectory class. Memoization here is a bit weird, because it ties the
    method to a specific (last) dataset.
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
        tmsd = np.arange(len(msd))
        lines.append(plt.loglog(tmsd, msd, **kwargs))
    msd = MSD(dataset)
    tmsd = np.arange(len(msd))
    lines.append(plt.loglog(tmsd, msd, color='k', linewidth=2, label='ensemble mean'))
    plt.legend()

    if dataset._selection_tags == {'_all'}:
        plt.title('MSDs')
    else:
        plt.title('MSDs for tags {}'.format(str(dataset._selection_tags)))
    plt.xlabel("time in frames")
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
    flags = {'_all in tags' : False, 'single tag' : False, 'single tag is _all' : False}
    if '_all' in tags:
        flags['_all in tags'] = True
        tags = list(dataset.tagset() - {'_all'})
        if len(tags) == 0:
            tags = ['_all']
    if len(tags) == 1:
        flags['single tag'] = True
        flags['single tag is _all'] = tags[0] == '_all'
        colordict = {tags[0] : None}
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

    # Delete all non-plotting kwargs
    for key in {'dims'}:
        try:
            del kwargs[key]
        except KeyError:
            pass

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
        if flags['single tag is _all']:
            plt.title('Trajectories')
        else:
            plt.title('Trajectories for tag "{}"'.format(tags[0]))
    elif flags['_all in tags']:
        plt.title("Trajectories by tag")
    else:
        plt.title("Trajectories for tags {}".format(str(tags)))

    # Done
    return lines

def hist_distances(dataset, **kwargs):
    """
    Draw a histogram of distances. For two-locus trajectories, this is the
    absolute distance between the loci, for single locus trajectories it is
    simply the absolute value of the trajectory.

    Note:
    If you need the array of distances, use
        dists = np.concatenate([traj._data[0, :, 0] for traj in dataset.process(<preproc>)])
    where <preproc> is the preprocessing function appropriate for your dataset.
    """
    if dataset.getHom('N') == 2:
        dsprocessed = dataset.process(lambda traj : traj.relativeTrajectory().absTrajectory())
    elif dataset.getHom('N') == 1:
        dsprocessed = dataset.process(lambda traj : traj.absTrajectory())
    else:
        raise RuntimeError("Dataset has neither homogeneously N = 1 nor N = 2")

    data = np.concatenate([traj._data[0, :, 0] for traj in dsprocessed])

    if 'bins' not in kwargs.keys():
        kwargs['bins'] = 'auto'

    plt.figure()
    h = plt.hist(data, **kwargs)
    plt.title('Distance histogram')
    return h

def KLD_PC(dataset, n=10, k=20, dt=1):
    """
    Apply the KLD estimator presented by (Perez-Cruz, 2008). We reduce the
    bias of the estimator by randomly choosing half the snippets for
    estimation of the densities and then sample at the other half.

    Input
    -----
    dataset : TaggedList of Trajectory
        the data to run the KLD estimation on
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
    Dest : estimated KLD in nats

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
    parity = dataset.getHom('parity')
    assert parity in {'even', 'odd'}

    # Generate snippets
    snips = []
    for traj in dataset:
        newsnips = [traj[start:(start+(n*dt)):dt].flatten() for start in range(len(traj)-(n*dt)+1)]
        snips += [snip for snip in newsnips if not np.any(np.isnan(snip))]
    snips = np.array(snips)

    # DCT seems to speed up neighbor search. Analytically it is irrelevant, as
    # long as normalized and we account for the switching parity of the
    # components (see below)
    snips = scipy.fftpack.dct(snips, axis=1, norm='ortho')

    # Split in two halves for estimation/sampling
    ind = random.sample(range(len(snips)), len(snips))
    halfN = np.ceil(len(snips)/2).astype(int)

    estimation_snips = snips[ind[:halfN]]
    sample_snips = snips[ind[halfN:]]

    # Build neighbor trees and run estimation
    # Note that time reversal in DCT space means multiplying all odd modes by -1
    tree_fw = KDTree(estimation_snips)
    if parity == 'even':
        tree_bw = KDTree(estimation_snips*[(-1)**i for i in range(estimation_snips.shape[1])])
    else:
        tree_bw = KDTree(estimation_snips*[(-1)**(i+1) for i in range(estimation_snips.shape[1])])

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
        self.KLDmethod = KLD_PC

        self.KLDkwargs = {'n' : 10, 'k' : 20, 'dt' : 1}

    def setup(self, **kwargs):
        """
        Set up the environment/parameters for running the estimation.

        Input
        -----
        KLDmethod : callable
            the method to use for KLD estimation.
            default: KLD_PC
        bootstraprepeats : integer
            how often to repeat each run with a different partition of the data
            set.
            default: 20
        processes : integer
            how many processes to use.
            default: 16
        parity : 'even' or 'odd'
            whether the trajectories in the dataset are even or odd under time
            reversal
            default: 'even'
        other keyword arguments :
            the parameters for KLD_PC. Anything given as a list will be
            sweeped.

        Notes
        -----
        The default values for everything are set in __init__(), so you can
        call this method also to change specific values while keeping
        everything else the same.
        """
        for key in ['bootstraprepeats', 'processes', 'KLDmethod', 'parity']:
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

    def preprocess(self, preproc):
        """
        Run some preprocessing on the data set. 

        Input
        -----
        preproc : callable, taking a trajectory and returning a trajectory
            the function to use for preprocessing. Will be applied to every
            trajectory individually via TaggedList.apply().
            Examples:
                lambda traj : traj.relativeTrajectory().absTrajectory() # would give absolute distance for two-locus trajectory
                lambda traj : traj.relativeTrajectory().diffTrajectory().absTrajectory() # would give absolute displacements
            default: identity (i.e. lambda traj : traj)

        Notes
        -----
        If writing your own preproc function (i.e. not using the ones from
        Trajectory) remember to update the parity property of all trajectories.

        As of now, this function literally only calls self.ds.apply(preproc).
        It serves more as a reminder that preprocessing might be necessary.
        """
        self.ds.apply(preproc)

    @staticmethod
    def _parfun(args):
        """
        args should be a composite dict:
         - an entry 'randomseed'
         - an entry 'self' containing a reference to the caller
         - finally, 'kwargs' will be passed to the KLD calculation

        Note: the reference to the caller is necessary, because it will be
        copied to each worker. This is not optimal and might require some
        refinement
        """
        random.seed(args['randomseed'])
        self = args['self']
        return self.KLDmethod(self.ds, **(args['kwargs']))

    def run(self):
        """
        Run the estimation. Remember to setup()

        Output
        ------
        A dict of argument lists and a corresponding np.ndarray for the
        computed KLDs.

        Notes
        -----
        As of now, the data is copied to every child process. Maybe this could
        be improved
        For reproducible results, set random.seed() before calling this function
        """
        # Assemble argument list
        kwkeys = self.KLDkwargs.keys() # Fixed key sequence for looping
        argslist = [{'self' : self, 'randomseed' : random.getrandbits(64), \
                     'kwargs' : {key : mykwvals[i] for i, key in enumerate(kwkeys)}} \
                    for mykwvals in itertools.product(*[self.KLDkwargs[key] for key in kwkeys]) \
                    for _ in range(self.bootstraprepeats)]

        # Run
        if self.processes == 1:
            Draw = map(KLDestimator._parfun, argslist)
        else:
            with multiprocessing.Pool(self.processes) as mypool:
                Draw = mypool.map(KLDestimator._parfun, argslist)

        # Assemble return dict
        ret = {key : [args['kwargs'][key] for args in argslist] for key in self.KLDkwargs.keys()}
        ret['KLD'] = np.array(Draw)
        return ret
