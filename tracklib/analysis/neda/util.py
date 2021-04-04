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
    def __init__(self, traj, nStates=2, thresholds=None):
        self.T = len(traj)
        self.n = nStates
        self.t = np.array([i for i in range(len(traj)) if not np.any(np.isnan(traj[i]))])
        self.state = np.zeros((len(self.t),), dtype=int)

        if thresholds is not None:
            self.n = len(thresholds)+1
            dist_traj = traj.abs()[:][self.t, 0]
            self.state = np.sum([ \
                                 (dist_traj > thres).astype(int) \
                                 for thres in thresholds \
                                ], axis=0)

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
