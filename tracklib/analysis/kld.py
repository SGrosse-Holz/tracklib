from copy import deepcopy
import itertools

import random
import numpy as np
from sklearn.neighbors import KDTree
import scipy.fftpack

from tracklib import Trajectory, TaggedSet

import mkl
mkl.numthreads = 1
import multiprocessing

def KLD_PC(dataset, n=10, k=20, dt=1):
    """
    Apply the KLD estimator presented by (Perez-Cruz, 2008). We reduce the
    bias of the estimator by randomly choosing half the snippets for
    estimation of the densities and then sample at the other half.

    Input
    -----
    dataset : TaggedSet of Trajectory
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

class Estimator:
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
        dataset : TaggedSet
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
            trajectory individually via TaggedSet.apply().
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
