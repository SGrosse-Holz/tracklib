import importlib
from copy import deepcopy

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class Trajectory(ABC):
    """
    Represents all kinds of trajectories.

    This class represents trajectories with 1 or 2 loci in 1, 2, or 3 spatial
    dimensions. Consequently, the internal `!np.ndarray` has shape (`N`, `T`,
    `d`).  Besides the actual trajectory data, this class also contains a dict
    for meta data. This can be used by the end-user, is also intended for use
    within the library though. For reference, see :ref:`traj_meta_fields`

    For creation of actual `Trajectory` objects, use `fromArray`. This will
    select and instantiate the appropriate subclass based on the shape of the
    input array. Any keyword arguments given to that function (or the
    constructor) will simply be written to `meta`

    Attributes
    ----------
    data : (N, T, d) np.ndarray
        the actual data in the trajectory. Try to avoid direct access.
    meta : dict
        a dict for (mostly user-specified) meta data. Also occasionally used by
        the library

    Notes
    -----
    For a `Trajectory` ``traj``, the following operations are defined:

    ``len(traj)``
        equivalent to ``traj.T``
    ``traj[ind]``
        accesss the time (stretch) specified by ind. The `N` and `T` dimensions
        might be squeezed (removed if they have length 1), while the `d`
        dimension is guaranteed to be present.

    This class implements the Sequence interface, i.e. it can be used
    much like a list.
    """

    ### Set up ###

    def __init__(self, **kwargs):
        self.data = None
        self.meta = kwargs

    @classmethod
    def fromArray(cls, array, t=None, **kwargs):
        """
        Create a new `Trajectory` from an array.

        Any keyword arguments are simply written into ``self.meta``.

        Parameters
        ----------
        array : (N, T, d) array-like
            the data for the new trajectory. We expect ``N in {1, 2}``, ``d in
            {1, 2, 3}``. Arrays with less than three dimensions will be
            interpreted as ``(T, d)`` or ``(T,)``, respectively.
        t : (T,) array-like, optional
            the frame number for each data point in the array. Use this if your
            array has missing data points; they will be patched with `!np.nan`.

        Returns
        -------
        Trajectory\_?N\_?d
            a new `Trajectory` object with the specified data

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

        if t is None:
            obj.data = array
        else:
            t = t - np.min(t)
            obj.data = np.empty((array.shape[0], np.max(t)+1, array.shape[2]), dtype=float)
            obj.data[:] = np.nan
            obj.data[:, t, :] = array

        return obj

    ### Basic properties ###

    @property
    def N(self):
        """ Number of loci """
        return self.data.shape[0]
    @property
    def T(self):
        """ Length in frames """
        return self.data.shape[1]
    @property
    def d(self):
        """ Number of dimensions """
        return self.data.shape[2]

    def __len__(self):
        return self.T

    def valid_frames(self):
        """
        Return the number of frames that have data.

        We regard a frame as unusable as soon as any data is missing. So this
        function counts the number of frames in the trajectory where none of
        the data is `!np.nan`.

        Returns
        -------
        int
        """
        return np.min(np.sum(~np.isnan(self.data), axis=1))

    def __getitem__(self, key):
        """
        Element-access

        Parameters
        ----------
        key : index or slice into the `T` dimension
        
        Returns
        -------
        np.ndarray
            the corresponding part of the trajectory.

        Notes
        -----
        The `N` dimension will be squeezed, i.e. for single locus trajectories
        that first dimension will be removed. The `T` dimension of the output
        follows the numpy conventions, while `d` is guaranteed to be present.
        """
        ret = self.data[:, key, :]
        # T squeezing is already done by np element-access
        try:
            ret = np.squeeze(ret, axis=0)
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
#         ret = self.data[:, key, :]
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
        Modifier: 2-norm

        Returns
        -------
        Trajectory

        Notes
        -----
        For multi-locus trajectories, this will take the norm of each locus
        individually. To get a `Trajectory` of relative distance, use
        ``Trajectory.relative().abs()``.

        See also
        --------
        diff, dims, relative
        """
        traj = Trajectory.fromArray(np.sqrt(np.sum(self.data**2, axis=2, keepdims=True)), **deepcopy(self.meta))

        if 'localization_error' in traj.meta.keys():
            traj.meta['localization_error'] = np.sqrt(np.mean(traj.meta['localization_error'], axis=-1, keepdims=True))
        if 'parity' in traj.meta.keys():
            traj.meta['parity'] == 'even'

        return traj

    def diff(self, dt=1):
        """
        Modifier: displacements

        Calculate the displacements over `!dt` frames.

        Parameters
        ----------
        dt : integer
            the time lag to use for displacement calculation
            default: 1, i.e. frame to frame displacements

        Returns
        -------
        Trajectory

        See also
        --------
        abs, dims, relative
        """
        traj =  Trajectory.fromArray(self.data[:, dt:, :] - self.data[:, :-dt, :], **deepcopy(self.meta))

        if 'localization_error' in traj.meta.keys():
            traj.meta['localization_error'] *= np.sqrt(2)
        if 'parity' in traj.meta.keys():
            traj.meta['parity'] == 'even' if self.meta['parity'] == 'odd' else 'odd'
        
        return traj

    def dims(self, key):
        """
        Modifier: select dimensions

        Parameters
        ----------
        key : list of int, or slice
            which dimensions to use. Attention: this cannot be a single `!int`. To
            get the ``i``-th spatial component, use ``traj.dims([i])``.

        Returns
        -------
        Trajectory

        See also
        --------
        abs, diff, relative
        """
        traj = Trajectory.fromArray(self.data[:, :, key], **deepcopy(self.meta))

        if 'localization_error' in traj.meta.keys():
            if len(traj.meta['localization_error'].shape) == 2:
                traj.meta['localization_error'] = traj.meta['localization_error'][:, key]
            else:
                traj.meta['localization_error'] = traj.meta['localization_error'][key]
            # Note: if key is a single int, we already get an error above, so
            # here we do not have to worry about vanishing dimensions.
        if 'parity' in traj.meta.keys():
            pass # Parity doesn't change

        return traj

    def relative(self):
        """
        Modifier: distance vector between two loci

        Returns
        -------
        Trajectory

        See also
        --------
        abs, diff, dims

        Notes
        -----
        Applies only to multi-locus trajectories and should thus be implemented
        in subclasses.
        """
        raise NotImplementedError("relative() does not apply to {}".format(type(self).__name__))

#     def yield_dims(self):
#         """
#         A generator yielding the spatial components as individual traces
#         """
#         for i in range(self.d):
#             yield np.squeeze(self.data[:, :, i])

    ### Plotting ###

    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot the trajectory / spatial components versus time.

        See the implementations in the `!Trajectory\_?N` subclasses for more
        detail.

        Keyword arguments are forwarded to ``ax.plot()``

        Parameters
        ----------
        ax : axes, optional
            the axes in which to plot. Can be ``None``, in which case we plot
            to ``plt.gca()``

        Returns
        -------
        list of matplotlib.lines.Line2D
            the output of ``ax.plot()``.

        See also
        --------
        plot_spatial
        """
        raise NotImplementedError()

    @abstractmethod
    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        """
        Plot the trajectory in space.

        For more detail see the implementation in the subclasses.

        Keyword arguments are forwarded to ``ax.plot()``

        Parameters
        ----------
        ax : axes, optional
            the axes in which to plot. Can be ``None``, in which case we plot
            to ``plt.gca()``
        dims : 2-tuple of int, optional
            the dimensions to plot. Only relevant for ``d >= 3``.

        Returns
        -------
        list of matplotlib.lines.Line2D
            the output of ``ax.plot()``.

        See also
        --------
        plot_vstime
        """
        raise NotImplementedError()

# Specialize depending on particle number or dimension, which changes behavior
# of some functions that can be overridden here
class N12Error(ValueError):
    """ For indicating that you confused ``N=1`` and ``N=2`` trajectories """
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
        return ax.plot(tplot, self.data[0], **kwargs)

    def _raw_plot_spatial(self, ax, dims, **kwargs):
        """ internal method for spatial plotting """
        if max(dims) >= self.d:
            raise ValueError("Invalid plotting dimensions")

        if 'linestyle' in kwargs.keys():
            if isinstance(kwargs['linestyle'], list) and len(kwargs['linestyle']) == 2:
                raise N12Error("Cannot apply two line styles to one-particle trajectory")
        if 'connect' in kwargs.keys():
            raise N12Error("Cannot connect one-particle trajectory")

        return ax.plot(self.data[0, :, dims[0]], \
                       self.data[0, :, dims[1]], \
                       **kwargs)

class Trajectory_2N(Trajectory):
    """
    Two-locus trajectory
    """
    def relative(self):
        traj = Trajectory.fromArray(self.data[0] - self.data[1], **deepcopy(self.meta))

        if 'localization_error' in traj.meta.keys():
            if len(traj.meta['localization_error'].shape) == 2:
                traj.meta['localization_error'] = np.sqrt(np.sum(traj.meta['localization_error']**2, axis=0))
            else:
                traj.meta['localization_error'] *= np.sqrt(2)
        if 'parity' in traj.meta.keys():
            pass # parity remains unchanged

        return traj

    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot spatial components of connection vector vs. time
        """
        if ax is None:
            ax = plt.gca()

        tplot = np.arange(self.T)
        return ax.plot(tplot, self.data[1] - self.data[0], **kwargs)

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
            m = np.mean(self.data, axis=1)
            ax.plot(m[:, dims[0]], m[:, dims[1]], color='k')

        # Plot first particle
        # This is the one setting the tone, so also create the legend entry
        kwargs['linestyle'] = linestyles[0]
        lines = ax.plot(self.data[0, :, dims[0]], \
                        self.data[0, :, dims[1]], \
                        **kwargs)
        kwargs['color'] = lines[0].get_color()

        if makeLegend:
            ax.plot(0, 0, label=label, **kwargs)

        # Plot second particle
        kwargs['linestyle'] = linestyles[1]
        lines += ax.plot(self.data[1, :, dims[0]], \
                         self.data[1, :, dims[1]], \
                         **kwargs)

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
