from copy import deepcopy
import itertools

import random
import numpy as np

from tracklib import Trajectory, TaggedSet

import mkl
mkl.numthreads = 1
from multiprocessing import Pool

class Sweeper:
    """
    Facilitates sweeping parameters for dataset-level analyses.

    This class mostly just captures a few bits of reusable code around running
    `!multiprocessing.Pool.map`.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the data to operate on
    fun : callable, of signature ``result = fun(dataset, *, ...)``
        the analysis function to apply to the dataset. Should take the dataset
        as first argument, and then a number of keyword arguments.
    copy : bool, optional
        whether to copy the whole dataset upon initialization. Setting to
        ``False`` means that any processing done on ``self.dataset`` will
        modify the original.
    repeats : int, optional
        how many repeats of the whole argument list to run.
    processes : int, optional
        how many workers to use.

    Notes
    -----
    There are two ways of applying pre-processing to the data:

     - either do the preprocessing upon initialization:

       >>>  sweep = Sweeper(dataset.process(<preproc>), copy=False)

       Note how in this case copying is unnecessary, since `TaggedSet.process
       <tracklib.taggedset.TaggedSet.process>` already returns a new dataset.

     - or `preprocess` in an individual step.
    """
    def __init__(self, dataset, fun, copy=True, repeats=1, processes=1):
        if copy:
            self.dataset = deepcopy(dataset)
        else:
            self.dataset = dataset

        self.fun = fun

        self.repeats = repeats
        self.processes = processes

    def preprocess(self, preproc):
        """
        Run some preprocessing on the data set. 

        Parameters
        ----------
        preproc : callable, with signature ``traj = preproc(traj)``
            the function to use for preprocessing. Will be applied to every
            trajectory individually via `TaggedSet.apply()
            <tracklib.taggedset.TaggedSet.apply>`

            Examples:

            >>> lambda traj : traj.relative().abs() # would give absolute distance for two-locus trajectory
            ... lambda traj : traj.relative().diff().abs() # would give absolute displacements

        Notes
        -----
        As of now, this function literally only calls
        ``self.dataset.apply(preproc)``.  It serves more as a reminder that
        preprocessing might be necessary.
        """
        self.dataset.apply(preproc)

    @staticmethod
    def _parfun(args):
        """
        For internal use in parallelization

        args should be a dict with the following entries:

         - `!'randomseed'` : seed for random number generation with `!random` (64 bits)
         - `!'nprandomseed'` : seed for `!np.random` (32 bits)
         - `!'self'` : a reference to the caller
         - `!'kwargs'` : the keyword arguments to `!fun`.

        Notes
        -----
        The reference to the caller is necessary, because it will be copied to
        each worker. This is not optimal and might require some refinement
        """
        random.seed(args['randomseed'])
        np.random.seed(args['nprandomseed'])
        self = args['self']
        return self.fun(self.dataset, **(args['kwargs']))

    def run(self, kwargdict):
        """
        Run the sweep.

        Parameters
        ----------
        kwargdict : dict or list of dict
            a dict of keyword arguments to use in the calls to `!fun`. There are
            two modes:

             - give a dict with entries for each keyword. In this case, if any
               entry is a list, it will be sweeped through, using the outer
               product of multiple arguments (i.e. if the dict reads ``{'a' :
               [1, 2], 'b' : [3, 4]}``, the sweep will run through ``(1, 3),
               (1, 4), (2, 3), (2, 4)``). Note that if any argument to `!fun`
               should actually be a list, you have to wrap in into another list
               to avoid sweeping: ``{'list_kwarg' : [[1, 2, 3]]}``.
             - give a list of dicts. This can be used for sweeps that are not
               the outer product of argument lists (e.g., contrasted with the
               example above, just running over ``(1, 3), (1, 4), (2, 3)``). In
               this case, the individual dicts will be passed to `!fun` as
               arguments with no modification.

        Returns
        -------
        dict
            a dict with the same keys as the input `!kwargdict`, plus an
            additional one, `!'result'`. The entry for each key is a list with
            values for each run.

        Notes
        -----
        For reproducible results, set `!random.seed()` before calling this
        function (not `!np.random.seed()`).

        If handing this function a list of `!dict` with differing key sets (not
        recommended!), then the returned `!dict` will have the union off all
        those as its key set. Entries that do not occur for a specific run are
        marked as ``None``.
        """
        kwargdict_is_list = isinstance(kwargdict, list)

        # Assemble argument list
        if kwargdict_is_list:
            argslist = [{
                    'self' : self,
                    'randomseed' : random.getrandbits(64),
                    'nprandomseed' : random.getrandbits(32),
                    'kwargs' : kwargs,
                }
                for kwargs in kwargdict
                for _ in range(self.repeats)
            ]
        else:
            # wrap single arguments as list
            for key in kwargdict:
                if not isinstance(kwargdict[key], list):
                    kwargdict[key] = [kwargdict[key]]

            argslist = [{
                    'self' : self,
                    'randomseed' : random.getrandbits(64),
                    'nprandomseed' : random.getrandbits(32),
                    'kwargs' : {key : mykwvals[i] for i, key in enumerate(kwargdict.keys())}
                }
                for mykwvals in itertools.product(*kwargdict.values())
                for _ in range(self.repeats)
            ]

        # Run
        if self.processes == 1:
            reslist = list(map(Sweeper._parfun, argslist))
        else:
            with Pool(self.processes) as mypool:
                reslist = mypool.map(Sweeper._parfun, argslist)

        # Get full list of keys
        if kwargdict_is_list:
            keys = set()
            for cur in kwargdict:
                keys |= cur.keys()

            # Fill missing values with None
            for args in argslist:
                nones = {key : None for key in keys}
                args['kwargs'] = nones.update(args['kwargs']) or nones
        else:
            keys = kwargdict.keys()

        # Assemble return dict
        ret = {key : [args['kwargs'][key] for args in argslist] for key in keys}
        ret['result'] = reslist
        return ret
