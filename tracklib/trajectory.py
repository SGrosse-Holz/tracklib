import importlib
from copy import deepcopy

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class Trajectory(ABC)
    """
    Represents all kinds of trajectories.

    This class represents trajectories with 1 or 2 loci in 1, 2, or 3 spatial
    dimensions. Consequently, the internal np.ndarray has shape (N, T, d).
    Besides the actual trajectory data, this class also contains a dict for
    meta data. This can be used by the end-user, is also intended for use
    within the library though. tracklib.analysis.MSD for example writes the
    'MSD' and 'MSDmeta' entries of this dict.
    
    TODO: write a list of all keys used by the library somewhere

    For creation of actual Trajectory objects, use Trajectory.fromArray(). This
    will select and instantiate the appropriate subclass based on the shape of
    the input array.

    Parameters
    ----------
    Any keyword arguments given to the constructor will be written to self.meta

    Operators
    ---------
    For a Trajectory traj, the following operations are defined:
    len(traj)
        equivalent to traj.T
    traj[ind]
        accesss the time (stretch) specified by ind. The N and T dimensions
        might be squeezed (removed if they have length 1), while the d
        dimension is guaranteed to be present.

    Notes
    -----
    This class implements the Sequence interface, i.e. it can be used
    much like a list.
    """

    ### Set up ###

    def __init__(self, **kwargs):
        self._data = None
        self.meta = kwargs

    @classmethod
    def fromArray(cls, array, **kwargs):
        """
        Create a new Trajectory from an array.

        Parameters
        ----------
        array : (N, T, d) array-like
            the data for the new trajectory. We expect N in {1, 2}, d in {1, 2,
            3}. Arrays with less than three dimensions will be interpreted as
            (T, d) or (T,), respectively.
        Additional keyword arguments are saved in the dict self.meta

        Returns
        -------
        A new Trajectory object with the specified data

        Notes
        -----
        The input data is copied.
        """
        array = np.array(array) # Note that this also copies the array
        if len(array.shape) > 3:
            raise ValueError("Array of shape {} cannot be interpreted as trajectory".format(str(array.shape)))
        elif len(array.shape) == 2:
            array = np.expand_dims(array, 0)
        elif len(array.shape) == 1:
            array = np.expand_dims(array, (0, 2))

        try:
            # TODO: is this importlib-stuff necessary, or is there a better way?
            obj = getattr(importlib.import_module(cls.__module__), "Trajectory_{:d}N{:d}d".format(array.shape[0], array.shape[2]))(**kwargs)
        except AttributeError:
            raise ValueError("Could not instantiate trajectory with (N, T, d) = {}".format(str(array.shape)))

        obj._data = array
        return obj

    ### Basic properties ###

    @property
    def N(self):
        """ Number of loci """
        return self._data.shape[0]
    @property
    def T(self):
        """ Length in frames """
        return self._data.shape[1]
    @property
    def d(self):
        """ Number of dimensions """
        return self._data.shape[2]

    def __len__(self):
        return self.T

    def __getitem__(self, key):
        """
        Element-access

        The output will be squeezed along the N and T dimensions (i.e. they
        will be removed if there is only a single entry). The d dimension is
        guaranteed to be present.

        Parameters
        ----------
        key : index or slice
        
        Returns
        -------
        The corresponding part of the trajectory.
        """
        ret = self._data[:, key, :]
        for ax in [1, 0]: # Go in reverse order
            try:
                np.squeeze(ret, axis=ax)
            except ValueError:
                pass
        return ret

#     def get(self, key, tosqueeze='N'):
#         """
#         Element-access with controlled squeezing
# 
#         This is an augmentation of the []-operator, handing control over
#         squeezing (removing of single-entry dimensions) to the user.
# 
#         Parameters
#         ----------
#         key : slice
#             indices into the time dimension of the trajectory
#         tosqueeze : str, optional
#             which dimensions to remove (if singular). Give this as a string
#             containing 'N', 'T', 'd' or combinations thereof.
# 
#         Returns
#         -------
#         np.ndarray
#             a view into the trajectory
#         """
#         ret = self._data[:, key, :]
#         # If we remove dimensions from the back, then indices in front will
#         # still be correct, so we can work iteratively
#         if 'd' in tosqueeze and ret.shape[2] == 1:
#             ret = np.squeeze(ret, 2)
#         if 'T' in tosqueeze and ret.shape[1] == 1:
#             ret = np.squeeze(ret, 1)
#         if 'N' in tosqueeze and ret.shape[0] == 1:
#             ret = np.squeeze(ret, 0)
#         return ret

    ### Modifiers ###

    def abs(self):
        """
        Give a new trajectory holding the magnitude (i.e. 2-norm) of the
        current one.

        Notes
        -----
        For multi-locus trajectories, this will take the norm of each locus
        individually. To get a Trajectory of relative distance, use
        Trajectory.relative().abs() .
        """
        return Trajectory.fromArray(np.sqrt(np.sum(self._data**2, axis=2, keepdims=True)), **deepcopy(self.meta))

    def relative(self):
        """
        Give a new trajectory holding the distance vector(s) between multiple
        loci.

        Notes
        -----
        Applies only to multi-locus trajectories and should thus be implemented
        in subclasses.
        """
        raise NotImplementedError("relative() does not apply to {}".format(type(self).__name__))

    def diff(self, dt=1):
        """
        Give a new trajectory holding the steps/displacements/derivative of the
        current one.

        Parameters
        ----------
        dt : integer
            the time lag to use for displacement calculation
            default: 1, i.e. frame to frame displacements
        """
        return Trajectory.fromArray(self._data[:, dt:, :] - self._data[:, :-dt, :], **deepcopy(self.meta))

    def dims(self, key):
        """
        Give a new trajectory with only a subset of the spatial components

        Parameters
        ----------
        key : list of int, or slice
            which dimensions to use
        """
        return Trajectory.fromArray(self._data[:, :, key], **deepcopy(self.meta))

#     def yield_dims(self):
#         """
#         A generator yielding the spatial components as individual traces
#         """
#         for i in range(self.d):
#             yield np.squeeze(self._data[:, :, i])

    ### Plotting ###

    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot the trajectory / spatial components versus time.

        See the implementations in the Trajectory\_?N subclasses for more
        detail.

        Parameters
        ----------
        ax : axes
            the axes to plot in. Can be None, in which case we plot to
            plt.gca()
        All further keyword arguments will be forwarded to ax.plot()

        Returns
        -------
        list of lines
            the output of ax.plot().
        """
        raise NotImplementedError()

    @abstractmethod
    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        """
        Plot the trajectory in space.

        For more detail see the implementation in the subclasses.

        Parameters
        ----------
        ax : axes
            the axes in which to plot. Can be None, in which case we will plot
            to plt.gca()
        dims : 2-tuple of int
            the dimensions to plot. Only relevant for d >= 3.
            default: (0, 1)
        All other keyword arguments will be forwarded to ax.plot()

        Returns
        -------
        list of lines
            the output of ax.plot().
        """
        raise NotImplementedError()

# Specialize depending on particle number or dimension, which changes behavior
# of some functions that can be overridden here
class N12Error(ValueError):
    """ For indicating that you confused N=1 and N=2 trajectories """
    pass

# Particle number specializations
class Trajectory_1N(Trajectory):
    """
    Single-locus trajectory
    """
    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot spatial components vs. time
        """
        if ax is None:
            ax = plt.gca()

        tplot = np.arange(self.T)
        return ax.plot(tplot, self._data[0], **kwargs)

    def _raw_plot_spatial(self, ax, dims, **kwargs):
        """ internal method for spatial plotting """
        if max(dims) >= self.d:
            raise ValueError("Invalid plotting dimensions")

        if 'linestyle' in kwargs.keys():
            if isinstance(kwargs['linestyle'], list) and len(kwargs['linestyle']) == 2:
                raise N12Error("Cannot apply two line styles to one-particle trajectory")
        if 'connect' in kwargs.keys():
            raise N12Error("Cannot connect one-particle trajectory")

        return ax.plot(self._data[0, :, dims[0]], \
                       self._data[0, :, dims[1]], \
                       **kwargs)

class Trajectory_2N(Trajectory):
    """
    Two-locus trajectory
    """
    def relative(self):
        return Trajectory.fromArray(self._data[0] - self._data[1], **deepcopy(self.meta))

    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot spatial components of connection vector vs. time
        """
        if ax is None:
            ax = plt.gca()

        tplot = np.arange(self.T)
        return ax.plot(tplot, self._data[1] - self._data[0], **kwargs)

    def _raw_plot_spatial(self, ax, dims, connect=True, **kwargs):
        """ internal method for spatial plotting """
        if max(dims) >= self.d:
            raise ValueError("Invalid plotting dimensions")

        linestyles = ['-', (0, (1, 1))]
        if 'linestyle' in kwargs.keys():
            if isinstance(kwargs['linestyle'], list) and len(kwargs['linestyle']) == 2:
                linestyles = kwargs['linestyle']
            else:
                linestyles = [kwargs['linestyle'], kwargs['linestyle']]

        makeLegend = 'label' in kwargs.keys()
        if makeLegend:
            label = kwargs['label']
            del kwargs['label']

        # First plot the connection, such that it is underneath the
        # trajectories
        if connect:
            m = np.mean(self._data, axis=1)
            ax.plot(m[:, dims[0]], m[:, dims[1]], color='k')

        # Plot first particle
        # This is the one setting the tone, so also create the legend entry
        kwargs['linestyle'] = linestyles[0]
        lines = ax.plot(self._data[0, :, dims[0]], \
                        self._data[0, :, dims[1]], \
                        **kwargs)
        kwargs['color'] = lines[0].get_color()

        if makeLegend:
            ax.plot(0, 0, label=label, **kwargs)

        # Plot second particle
        kwargs['linestyle'] = linestyles[1]
        lines.append(ax.plot(self._data[1, :, dims[0]], \
                             self._data[1, :, dims[1]], \
                             **kwargs))

        return lines

# Dimensionality specializations
class Trajectory_1d(Trajectory):
    """
    1d trajectory
    """
    def plot_spatial(self, *args, **kwargs):
        raise NotImplementedError("Cannot plot spatial trajectory for 1d trajectory. Use plot_vstime()")

class Trajectory_2d(Trajectory):
    """
    2d trajectory
    """
    def plot_spatial(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        return self._raw_plot_spatial(ax, (0, 1), **kwargs)

class Trajectory_3d(Trajectory):
    """
    3d trajectory
    """
    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        if ax is None:
            ax = plt.gca()

        return self._raw_plot_spatial(ax, dims, **kwargs)

# Now we're getting to the fully concrete level. It's unlikely that there is
# any further specialization here
class Trajectory_1N1d(Trajectory_1N, Trajectory_1d):
    pass

class Trajectory_1N2d(Trajectory_1N, Trajectory_2d):
    pass

class Trajectory_1N3d(Trajectory_1N, Trajectory_3d):
    pass

class Trajectory_2N1d(Trajectory_2N, Trajectory_1d):
    pass

class Trajectory_2N2d(Trajectory_2N, Trajectory_2d):
    pass

class Trajectory_2N3d(Trajectory_2N, Trajectory_3d):
    pass
