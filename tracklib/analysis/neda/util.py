"""
Useful constructs used throughout the `neda` module
"""

import numpy as np

class Loopingtrace:
    """
    Trace of looping states

    `Loopingtrace` objects associate an integer state with each valid (i.e.
    non-``nan``) time point in a given trajectory. To this end, it keeps a list
    of those valid time points alongside the list of states.

    In the context of looping inference, the states indicate which out of a
    list of models to use for evolution up to the labelled point. For the first
    data point this means that the associated model specifies the equilibrium
    state we start from.

    Attributes
    ----------
    T : int
        total length of the associated `Trajectory`
    n : int
        total number of states (relevant e.g. for acceptance probabilities)
    t : (N,) np.ndarray, dtype=int
        valid frames of the associated `Trajectory`
    state : (N,) np.ndarray, dtype=int
        state associated with each (valid) frame of the trajectory. By default
        initialized with zeroes, unless `!thresholds` specified

    Notes
    -----
    This class forwards the ``Sequence`` interface of its `state` attribute,
    i.e. for a `Loopingtrace` ``lt`` ``len(lt)`` gives the number of valid
    frames, ``lt[i]`` gives the state associated with the ``i``th valid frame
    and ``lt[i] = 2`` can be used to assign that state. Note that this usage is
    preferred over accessing ``lt.state``, since it allows for more internal
    control, such as type checks upon assignment.
    """
    def __init__(self):
        self.T = -1
        self.n = -1
        self.t = np.array([], dtype=int)
        self.state = np.array([], dtype=int)

    @classmethod
    def forTrajectory(cls, traj, nStates=2, thresholds=None):
        """
        Create a new `Loopingtrace` for a given `Trajectory`

        Parameters
        ----------
        traj : tracklib.Trajectory
            the `Trajectory` to which the `Loopingtrace` is associated. This
            information is needed to assemble the list of valid time points.
        nStates : int, optional
            the number of possible states
        thresholds : sized iterable (e.g. list) of float, optional
            can be used to initialize the trajectory with states corresponding to a
            set of absolute distance thresholds. The state of each frame will be
            the number of given thresholds that are smaller than the absolute
            distance, i.e. ``thresholds=[5, 10]`` means ``dist <= 5`` is state 0,
            ``5 < dist <= 10`` state 1, ``10 < dist`` state 2. Will overwrite
            `!nStates` with ``len(thresholds)+1``.
        """
        new = cls()
        new.T = len(traj)
        new.n = nStates
        new.t = np.array([i for i in range(len(traj)) if not np.any(np.isnan(traj[i]))])
        new.state = np.zeros((len(new.t),), dtype=int)

        if thresholds is not None:
            new.n = len(thresholds)+1
            dist_traj = traj.abs()[:][new.t, 0]
            new.state = np.sum([ \
                                 (dist_traj > thres).astype(int) \
                                 for thres in thresholds \
                                ], axis=0)

        return new
    
    @classmethod
    def fromStates(cls, states, nStates=None):
        """
        Create a new `Loopingtrace` just from a list of states

        Parameters
        ----------
        states : array-like
            does not have to be of integer type. This can be used to specify
            missing frames as ``np.nan``.
        nStates : int, optional
            how many states there could be in the trajectory in principle. If
            unspecified, defaults to ``max(states)``.
        
        Returns
        -------
        Loopingtrace
        """
        states = np.asarray(states) # no type casting yet, int cannot deal with nan

        new = cls()
        new.T = len(states)
        new.t = np.where(~np.isnan(states))[0].astype(int)
        new.state = states[new.t].astype(int)
        if nStates is None:
            new.n = np.max(states)
        else:
            new.n = nStates

        return new

    def copy(self):
        """
        Copy myself

        Creates a copy of the calling object. Faster than
        ``deepcopy(loopingtrace)``, i.e. used for performance.
        """
        new = self.__new__(type(self)) # Skip init
        new.T = self.T
        new.n = self.n
        new.t = self.t.copy()
        new.state = self.state.copy()
        return new

    def __len__(self):
        return len(self.state)

    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, val):
        assert val < self.n
        assert isinstance(val, (int, np.integer))
        self.state[key] = val

    def plottable(self):
        """
        Return t, y for plotting as steps

        Returns
        -------
        t : np.ndarray
        y : np.ndarray

        Example
        -------
        For a `Loopingtrace` ``lt``:

        >>> from matplotlib import pyplot as plt
        ... plt.plot(*lt.plottable(), color='r')
        ... plt.show()

        """
        tplot = np.array([np.insert(self.t[:-1], 0, self.t[0]-1), self.t]).T.flatten()
        yplot = np.array([self.state, self.state]).T.flatten()
        return tplot, yplot

    def full_valid(self):
        """
        Give a full-length array of states

        This returns an array of looping states of length equal to the
        associated trajectory. Invalid frames in the trajectory (i.e. those
        containing ``nan``) will be labelled with the same state as the next
        valid one. This implements the interpretation that the state associated
        with a frame specifies the model governing the evolution up to that
        frame.

        Returns
        -------
        states : (self.T,) np.ndarray
            a valid states associated with each frame in the trajectory, even
            the invalid ones.
        """
        full = np.zeros((self.T,), dtype=int)
        last_ind = -1
        for cur_ind, cur_val in zip(self.t, self.state):
            full[(last_ind+1):(cur_ind+1)] = cur_val
            last_ind = cur_ind
        return full

class ParametricFamily:
    """
    A base class for parametric families

    This class is a template for parametric families of any type, i.e. its main
    capability (`get`) converts a set of parameters into an object, e.g. a
    `Prior` or `Model <models.Model>`. On top of that, meta information like
    bounds for parameters is stored here. In summary, this class provides the
    interface we need generically for fitting.

    Parameters
    ----------
    start_params : tuple of float
        the initial parameters
    bounds : list of (lower, higher) tuples
        domain for each parameter

    Attributes
    ----------
    start_params : tuple of float
    nParams : int
    bounds : list of (lower, higher) tuples

    See also
    --------
    fit_model

    Example
    -------
    A family of `priors.GeometricPrior`:

    >>> fam = ParametricFamily((0,), [(None, 0)])
    ... fam.get = lambda logq : priors.GeometricPrior(logq, nStates=2)
    
    Note that `GeometricPrior` also has a `family` method providing essentially
    this construction.

    A family of three state `RouseModels <models.RouseModel>`, which is then
    fitted to a calibration dataset `!data`:

    >>> fam = ParametricFamily((1, 5), [(1e-10, None), (1e-10, None)])
    ... fam.get = lambda D, k : models.RouseModel(20, D, k, looppositions=[(0, 0), (5, 15), (0, -1)])
    ... fitresult = fit_model(data, fam)

    Note that you cannot use the framework provided here to optimize e.g. the
    number of monomers `!N`, since that is an integer programming problem. One
    could imagine comparing runs with different (fixed) `!N` though.
    """
    def __init__(self, start_params, bounds):
        self.start_params = start_params
        self.nParams = len(start_params)
        self.bounds = bounds

    def get(self, *params):
        """
        Generate an object from the given parameters

        This function should be overwritten upon instantiation
        """
        raise NotImplementedError # pragma: no cover
