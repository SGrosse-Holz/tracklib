import os, sys
from copy import deepcopy
import itertools

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
import scipy.fftpack
import scipy.stats
import scipy.linalg
import scipy.optimize

from . import util
from .trajectory import Trajectory
from .taggedlist import TaggedList

import mkl
mkl.numthreads = 1
import multiprocessing

def MSD(dataset, giveN=False, memo=True):
    """
    Calculate ensemble MSD for the given dataset

    Input
    -----
    dataset : TaggedList (possibly with some selection set)
        a list of Trajectory for which to calculate an ensemble MSD
    giveN : bool
        whether to return the sample size for each MSD data point
    memo : bool
        whether to use the memoization of Trajectory.msd()
        default: True

    Output
    ------
    if giveN:
        a tuple (msd, N) of (T,) arrays containing MSD and sample size
        respectively
    if not giveN:
        only msd, i.e. a (T,) array.

    Notes
    -----
    Corresponding to python's 0-based indexing, msd[0] = 0, such that
    msd[dt] is the MSD at a time lag of dt frames.
    """
    msdNs = [traj.msd(giveN=True, memo=memo) for traj in dataset]

    maxlen = max(len(msdN[0]) for msdN in msdNs)
    emsd = msdNs[0][0]
    npad = [(0, maxlen-len(emsd))] + [(0, 0) for _ in emsd.shape[2:]]
    emsd = np.pad(emsd, npad, constant_values=0)
    eN = np.pad(msdNs[0][1], npad, constant_values=0)
    emsd *= eN

    for msd, N in msdNs[1:]:
        ind = N > 0
        emsd[:len(msd)][ind] += (msd*N)[ind]
        eN[:len(N)][ind] += N[ind]
    emsd /= eN

    if giveN:
        return (emsd, eN)
    else:
        return emsd

def hist_lengths(dataset, **kwargs):
    """
    Plot a histogram of trajectory lengths for the given dataset

    Input
    -----
    dataset : TaggedList (possibly with some selection set)
        the dataset to use
    All other keyword arguments will be forwarded to plt.hist().

    Output
    ------
    The return value of plt.hist(), i.e. a tuple (n, bins, patches)

    Notes
    -----
    This should be thought of only as a quickshot way to take a look at the
    data. For more elaborate plotting, obtain the trajectory lengths as
        lengths = [len(traj) for traj in dataset]
    and produce a plot to your liking.
    """
    lengths = [len(traj) for traj in dataset]
    
    if 'bins' not in kwargs.keys():
        kwargs['bins'] = 'auto'

    plt.figure()
    h = plt.hist(lengths, **kwargs)
    plt.title("Histogram of trajectory lengths")
    plt.xlabel("Length in frames")
    return h

def plot_msds(dataset, **kwargs):
    """
    Plot individual and ensemble MSDs of the given dataset

    Input
    -----
    dataset : TaggedList (possibly with some selection set)
        the dataset to use
    All further keyword arguments are forwarded to plt.loglog for plotting of
    the individual trajectory MSDs

    Output
    ------
    Aggregate of the plt.plot outputs, i.e. a list of lines

    Notes
    -----
    Only intended as a quick overview plot, for more customization write your
    own plotting routine using analysis.MSD(dataset) and Trajectory.msd().
    """
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

    plt.title('MSDs')
    plt.xlabel("time in frames")
    plt.ylabel("MSD")
    
    return lines

def plot_trajectories(dataset, **kwargs):
    """
    Plot the trajectories in the given dataset. The preset selection can be
    overridden manually to ensure proper coloring (see Notes).

    Input
    -----
    dataset : TaggedList
        the set of trajectories to plot
    colordict : dict
        determines which tag is colored in which color. This should be a dict
        whose keys are the tags in the dataset, while the entries are anything
        recognized by the 'color' kwarg of plt.plot. 
        default: cycle through the default color cycle
    All further keyword arguments will be forwarded to
    Trajectory.plot_spatial()

    Output
    ------
    A list of lines, as returned by plt.plot()

    Notes
    -----
    Each tag will be associated with one color, and trajectories will be
    colored by one of the tags they're associated with.  There is no way to
    determine which one.

    Similarly to the other analysis.plot_* and analysis.hist_* functions, this
    is mostly intended for use in a quick overview. It does provide some more
    functionality though, in the hope that the user will not see a necessity to
    start plotting trajectories themselves.
    """
    flags = {'fallback_used' : False}

    try:
        fallback_color = kwargs['fallback_color']
        del kwargs['fallback_color']
    except KeyError:
        fallback_color = '#aaaaaa'

    # Get coloring
    try:
        colordict = kwargs['colordict']
        del kwargs['colordict']
    except KeyError:
        try:
            if isinstance(kwargs['color'], list):
                colors = kwargs['color']
            else:
                colors = [kwargs['color']]
        except KeyError:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        colordict = {tag : colors[i%len(colors)] for i, tag in enumerate(dataset.tagset())}
    
    # Plotting
    plt.figure()
    lines = []
    for traj, trajtags in dataset(giveTags=True):
        try:
            kwargs['color'] = colordict[(trajtags & colordict.keys()).pop()]
        except KeyError: # set().pop() raises a KeyError
            kwargs['color'] = fallback_color
            flags['fallback_used'] = True
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
    for tag in colordict.keys():
        kwargs['color'] = colordict[tag]
        kwargs['label'] = tag
        if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                kwargs['linestyle'] = kwargs['linestyle'][0]
        plt.plot(x0, y0, **kwargs)

    if flags['fallback_used']:
        kwargs['color'] = fallback_color
        kwargs['label'] = '<other tags>'
        if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                kwargs['linestyle'] = kwargs['linestyle'][0]
        plt.plot(x0, y0, **kwargs)
        

    # Control appearance
    if len(colordict) > 1 or flags['fallback_used']:
        plt.legend()
        plt.title("Trajectories in real space")
    else:
        plt.title("Trajectories for tag {}".format(list(colordict.keys())[0]))

    # Done
    return lines

def hist_distances(dataset, **kwargs):
    """
    Draw a histogram of distances. For two-locus trajectories, this is the
    absolute distance between the loci, for single locus trajectories it is
    simply the absolute value of the trajectory.

    Input
    -----
    dataset : TaggedList (possibly with some selection set)
        the trajectories to use
    All keyword arguments will be forwarded to plt.hist()

    Output
    ------
    The output of plt.hist(), i.e. a tuple (n, bins, patches).

    Notes
    -----
    This is intended for gaining a quick overview. For more elaborate tasks,
    obtain the distances as
        dists = np.concatenate([traj[:].flatten() for traj in dataset.process(<preproc>)])
    where <preproc> is the preprocessing function appropriate for your dataset.
    For a two-locus trajectory, this would presumably be
        preproc = lambda traj : traj.relative().abs()
    """
    if dataset.getHom('N') == 2:
        preproc = lambda traj : traj.relative().abs()
    elif dataset.getHom('N') == 1:
        preproc = lambda traj : traj.abs()
    else:
        raise RuntimeError("Dataset has neither homogeneously N = 1 nor N = 2")

    data = np.concatenate([traj[:].flatten() for traj in dataset.process(preproc)])

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
    parityset = {traj.meta['parity'] for traj in dataset}
    if len(parityset) > 1:
        raise ValueError("Trajectories have differing parity")
    parity = parityset.pop()
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
    bootstrapping and (eventually) use of different estimators.

    Notes
    -----
    There are two ways of applying pre-processing to the data:
     - either do the preprocessing upon initialization:
        est = KLDestimator(dataset.process(<preproc>), copy=False)
     - or preprocess in an individual step, using the preprocess function.
    As shown in the example, if preprocessing upon initialization, one can set
    the 'copy' argument to False to avoid unnecessary copying.
    
    For repeated evaluation on different parts of the same dataset: make sure
    to reset the selection in the dataset with a call to
    dataset.makeSelection() before initialization/preprocessing, such that all
    data will undergo preprocessing. Then you can subsequently call
    KLDestimator.dataset.makeSelection() to select parts of your dataset for
    estimation.
    """
    def __init__(self, dataset, copy=True):
        """
        Set up a new KLD estimation

        Input
        -----
        dataset : TaggedList
            the dataset to run the estimation on
        copy : bool
            whether to copy the dataset upon initialization. This might not be
            necessary if handing over a freshly preprocessed dataset (see
            Notes on the class level).
            default: True

        Notes
        -----
        This also sets default values for the estimation parameters. The user
        should adapt these by using the setup() function. For reference, the
        defaults are:
            bootstraprepeats = 20
            processes = 16
            KLDmethod = KLD_PC
            KLDkwargs = {'n' : 10, 'k' : 20, 'dt' : 1}
        """
        if copy:
            self.dataset = deepcopy(dataset)
        else:
            self.dataset = dataset

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
        other keyword arguments :
            the parameters for the estimation method. Anything given as a list
            will be sweeped.

        Notes
        -----
        The default values for everything are set in __init__(), so you can
        call this method also to change specific values while keeping
        everything else the same.
        """
        for key in ['bootstraprepeats', 'processes', 'KLDmethod']:
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
                lambda traj : traj.relative().abs() # would give absolute distance for two-locus trajectory
                lambda traj : traj.relative().diff().abs() # would give absolute displacements

        Notes
        -----
        If writing your own preproc function (i.e. not using the ones from
        Trajectory) remember to update the parity property of all trajectories.

        As of now, this function literally only calls self.dataset.apply(preproc).
        It serves more as a reminder that preprocessing might be necessary.
        """
        self.dataset.apply(preproc)

    @staticmethod
    def _parfun(args):
        """
        For internal use in parallelization

        args should be a dict with the following entries:
         - 'randomseed' : seed for random number generation
         - 'self' : a reference to the caller
         - 'kwargs' : the arguments for the KLD calculation

        Note: the reference to the caller is necessary, because it will be
        copied to each worker. This is not optimal and might require some
        refinement
        """
        random.seed(args['randomseed'])
        self = args['self']
        return self.KLDmethod(self.dataset, **(args['kwargs']))

    def run(self):
        """
        Run the estimation. Remember to setup() and possibly preprocess().

        Output
        ------
        A dict of argument lists and a corresponding np.ndarray for the
        computed KLDs.

        Notes
        -----
        For reproducible results, set random.seed() before calling this function

        Implementation Notes
        --------------------
        As of now, the data is copied to every child process. Maybe this could
        be improved
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

def chi2vsMSD(dataset, n=10, msd=None):
    """
    Calculate the snippet-wise chi-square score for trajectories vs. a given
    MSD. This can be used to check for statistical homogeneity, within single
    trajectories and across the dataset.
    
    Input
    -----
    dataset : TaggedList of Trajectory
        the trajectories to use
    n : integer
        the window size / length of the snippets to look at
    msd : array-like, longer than n
        the MSD to use for defining the reference Gaussian process. If omitted,
        this will be the empirical MSD of the data, as given by MSD(dataset).
        Note: we assume that this MSD is for trajectories like the one in the
        dataset, i.e. if you want to specify diffusive motion in d-dimensions,
        use 2*d*D*t, not 2*D*t.

    Output
    ------
    The calculated chi-square scores will be written to the trajectory metadata
    as 'chi2scores'.
    This function returns the #dof for the reference chi2 distribution (for use
    with scipy.stats.chi2)
    """
    d = dataset.getHom('d')

    if msd is None:
        msd = MSD(dataset)

    msd = np.insert(msd, 0, msd[1])
    corr = 0.5*(msd[2:] + msd[:-2] - 2*msd[1:-1])
    corr = corr / d # Correct for the fact that MSD is summed over dimensions
    maG = scipy.linalg.inv(scipy.linalg.toeplitz(corr[:n]))

    for traj in dataset:
        if traj.N == 1:
            displacements = traj.diff()
        elif traj.N == 2:
            displacements = traj.relative().diff()
        else:
            raise ValueError("Don't know what to do with trajectories with N = {}", traj.N)

        def chi2score(i):
            snip = displacements[i:(i+n)][0]
            return np.sum(np.diag( snip.T @ maG @ snip ))

        traj.meta['chi2scores'] = np.array([chi2score(i) for i in range(len(traj)-n)])

    return n*d

def plot_chi2(dataset, dof=None, p=0.05, ax=None, **kwargs):
    """
    Histogram the chi-square scores calculated with chi2vsMSD(). This assumes
    that each trajectory has a metadata field 'chi2scores'. Optionally also
    plots significance thresholds and/or the expected chi2 distribution.

    Input
    -----
    dataset : TaggedList of Trajectory
        the data to use
    dof : integer
        degrees of freedom for the reference chi2 distribution. If omitted, no
        reference will be shown.
    p : float in (0, 1)
        significance level at which to draw cutoffs. Can be set to None to
        prevent the plotting of these lines.
        default: 0.05
    ax : the axes handle into which to plot.
        default: plt.gca()
    Additional kwargs will be forwarded to plt.hist() for plotting
    """
    if ax is None:
        ax = plt.gca()

    scores = []
    for traj in dataset:
        scores += traj.meta['chi2scores'].tolist()
    scores = np.array(scores)

    preferences = {
            'bins' : 'auto',
            'density' : True,
            'histtype' : 'step'
        }
    for key in preferences.keys():
        if not key in kwargs.keys():
            kwargs[key] = preferences[key]

    ax.hist(scores[~np.isnan(scores)], **kwargs)

    if dof is not None:
        xplot = np.linspace(0, np.nanmax(scores), 1000)
        ax.plot(xplot, scipy.stats.chi2.pdf(xplot, dof), color='red', label='expected chi2')

        if p is not None:
            thres = scipy.stats.chi2.ppf(p, dof)
            ax.axvline(x=thres, color='magenta')
            thres = scipy.stats.chi2.isf(p, dof)
            ax.axvline(x=thres, color='magenta')

def fit_MSDscaling(traj, n=5):
    """
    Fit a powerlaw scaling to the first n data points of the MSD.

    Input
    -----
    traj : Trajectory
        the trajectory whose MSD we are interested in
    n : integer
        how many data points of the MSD to use for fitting

    Output
    ------
    None. The results of the fit are written to traj.meta['MSDscaling']. This
    will be a dict with keys 'alpha', 'logG', 'cov' for respectively the
    exponent, log of the prefactor, covariance matrix from the fit (cov[0, 0]
    is the variance of alpha, cov[1, 1] that for logG).

    Notes
    -----
     - logG is known to be a bad estimator for diffusivities.
     - "first n points of the MSD" means time lags 1 through n.
    """
    nmsd = traj.msd()[1:(n+1)]
    t = np.arange(len(nmsd))+1
    ind = ~np.isnan(nmsd)
    if np.sum(ind) < 2:
        traj.meta['MSDscaling'] = {
                'alpha' : np.nan,
                'logG' : np.nan,
                'cov' : np.nan*np.ones((1, 1)),
                }
    else:
        popt, pcov = scipy.optimize.curve_fit(lambda x, a, b : x*a + b, np.log(t[ind]), np.log(nmsd[ind]))
        traj.meta['MSDscaling'] = {
                'alpha' : popt[0],
                'logG' : popt[1],
                'cov' : pcov,
                }
