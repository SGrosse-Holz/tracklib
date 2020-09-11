import os,sys
import importlib
from copy import deepcopy

from abc import ABC, abstractmethod
import collections.abc

import numpy as np
import matplotlib.pyplot as plt
import joblib

from . import util

class Trajectory(ABC, collections.abc.Sequence):
    """
    A class to represent multi-particle trajectories. The key attribute is
    Trajectory._data, which is a (N, T, d) array, where N is the number of
    particles, T is the duration in frames, and d the number of dimensions.
    Note that most functions for now might assume N = 1 or N = 2. The latter is
    the development focus of this project.

    Besides the actual data, this class also stores a dictionary containing
    user-specified meta-data. Some entries in this dict (like parity and label)
    are used within the class. IMPLEMENTATION: Is this a good idea?

    Note: this implementation should be able to deal with missing data, i.e.
    np.nan's

    Note: the different possible types of trajectories (1 or 2 particles, 1, 2
    or 3 dimensions) are implemented as subclasses, thus facilitating
    customization

    Note: this class implements the Sequence interface, i.e. it can be used
    much like a list. Element access returns (N, d) arrays as data points.
    """
    def __init__(self, **kwargs):
        """
        Set up an empty trajectory

        Any keyword arguments are saved in the dict self.meta
        """
        self._data = None

        if not 'parity' in kwargs.keys():
            kwargs['parity'] = 'even'
        else:
            assert kwargs['parity'] in {'even', 'odd'}
        self.meta = kwargs

    def __len__(self):
        """
        Return duration of trajectory, in frames.

        Notes
        -----
        Identical to Trajectory.T
        """
        return self._data.shape[1]

    def __getitem__(self, key):
        """
        Element-access

        Input
        -----
        key : index or slice
        
        Output
        ------
        The corresponding part of the trajectory, as array of shape (N, t, d),
        where t is the length of the selection.

        Note
        ----
        For a Trajectory object traj, traj[:] is equivalent to traj._data. The
        former is preferred.
        """
        return self._data[:, key, :]

    @property
    def N(self):
        """
        Get the number of loci
        """
        return self._data.shape[0]
    @property
    def T(self):
        """
        Get the length in frames
        """
        return self._data.shape[1]
    @property
    def d(self):
        """
        Get the number of dimensions
        """
        return self._data.shape[2]

    @classmethod
    def fromArray(cls, array, **kwargs):
        """
        Create a new Trajectory from an array.

        Input
        -----
        array : (N, T, d) array-like
            the data for the new trajectory. Note that for now, we expect N in
            {1, 2} and d in {1, 2, 3}.
        Additional keyword arguments are saved in the dict self.meta

        Output
        ------
        A new Trajectory object with the specified data

        Notes
        -----
        The input data is copied.
        This function returns objects of the appropriate subclass Trajectory_xNxd.
        """
        array = np.array(array) # Note that this also copies the array
        if len(array.shape) > 3:
            raise ValueError("Array of shape {} cannot be interpreted as trajectory".format(str(array.shape)))
        elif len(array.shape) == 2:
            array = np.expand_dims(array, 0)
        elif len(array.shape) == 1:
            array = np.expand_dims(array, (0, 2))

        try:
            obj = getattr(importlib.import_module(cls.__module__), "Trajectory_{:d}N{:d}d".format(array.shape[0], array.shape[2]))(**kwargs)
        except AttributeError:
            raise ValueError("Could not instantiate trajectory with (N, T, d) = {}".format(str(array.shape)))

        obj._data = array
        return obj

    def copyMeta(self, target):
        """
        Legacy function; use deepcopy(self.meta)

        Copy all the meta information (like label, dx, dt) to another object.
        This will omit attributes prefixed with an underscore. These include
        _data, which is not considered meta data, and _msdN, which is used for
        msd memoization.

        Input
        -----
        target : Trajectory
            the Trajectory object to copy the meta data to
        """
        raise NotImplementedError("This will be removed from implementation")
        for key in self.__dict__.keys():
            if not key.startswith('_'):
                setattr(target, key, self.__dict__[key])

    @abstractmethod
    def _eff1Ndata(self):
        """
        Internal use only

        Dummy function for aggregation of data over the loci. Should be identity for N=1, relative data for N=2.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_dimension_labels(self):
        """
        Internal use only

        Assemble labels for the individual spatial dimensions from the trajectory labels
        """
        raise NotImplementedError()

    def plot_vstime(self, ax=None, **kwargs):
        """
        Plot the trajectory data versus time. Different dimensions will be
        plotted individually, while different loci will be aggregated according
        to the internal function _eff1Ndata().

        Input
        -----
        ax : axes
            the axes to plot in. Can be None, in which case we plot to
            plt.gca()
        All further keyword arguments will be forwarded to ax.plot()

        Output
        ------
        The output of ax.plot(), i.e. a collection of lines.
        """
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys():
            kwargs['label'] = self._get_dimension_labels()

        tplot = np.arange(self.T)
        return ax.plot(tplot, self._eff1Ndata(), **kwargs)

    @abstractmethod
    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        """
        Plot the trajectory in a spatial coordinate system. This function
        should be overridden by the implementations of subclasses with specific
        dimension, since its behavior depends crucially on the dimension.

        Input
        -----
        ax : axes
            the axes in which to plot. Can be None, in which case we will plot
            to plt.gca()
        dims : 2-tuple of int
            the dimensions to plot. Only relevant for d >= 3.
            default: (0, 1)
        All other keyword arguments will be forwarded to ax.plot()

        Output
        ------
        The output of ax.plot(), i.e. a collection of lines.
        """
        raise NotImplementedError()

    def msd(self, memo=True, giveN=False):
        """
        Calculate mean square displacement (MSD) of the Trajectory. This
        function is memoized, i.e. will perform the actual calculation only
        upon the first call, while subsequent calls will simply return the
        previously computed MSD.

        Input
        -----
        memo : bool
            whether to use memoization. Set to False for explicit
            recalculation.
            default: True
        giveN : bool
            whether to return the sample size for each point of the MSD. This
            is important for example when calculating ensemble averages.
            default: False

        Output
        ------
        if giveN:
            a tuple (msd, N) where both are (T,) arrays containing the
            respective values
        if not giveN:
            only msd, i.e. a (T,) array containing the MSD values.

        Notes
        -----
        Corresponding to python's 0-based indexing, msd[0] = 0, such that
        msd[dt] is the MSD at a time lag of dt frames.
        """
        if not hasattr(self, '_msdN') or not memo:
            self._msdN = util.msd(self._eff1Ndata(), giveN=True)

        if giveN:
            return self._msdN
        else:
            return self._msdN[0]

    def abs(self):
        """
        Give a new trajectory holding the magnitude (i.e. 2-norm) of the
        current one.

        Notes
        -----
        For multi-locus trajectories, this will take the norm of each locus
        individually. To get a Trajectory of relative distance, use
        Trajectory.relative().abs() .

        Absolute value trajectories always have even parity, because they
        cannot have negative values.
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

        Input
        -----
        dt : integer
            the time lag to use for displacement calculation
            default: 1, i.e. frame to frame displacements

        Notes
        -----
        Parity of the displacement trajectory is always opposite to the
        original.
        """
        obj = Trajectory.fromArray(self._data[:, dt:, :] - self._data[:, :-dt, :], **deepcopy(self.meta))

        if obj.meta['parity'] == 'even':
            obj.meta['parity'] = 'odd'
        else:
            obj.meta['parity'] = 'even'

        return obj

# Specialize depending on particle number or dimension, which changes behavior
# of some functions that can be overridden here
class N12Error(ValueError):
    pass

# Particle number specializations
class Trajectory_1N(Trajectory):
    """
    Behavior specific to trajectories of a single locus
    """
    def _eff1Ndata(self):
        return self._data[0]

    def _raw_plot_spatial(self, ax, dims, **kwargs):
        """ internal method for spatial plotting """
        if max(dims) >= self.d:
            raise ValueError("Invalid plotting dimensions")

        if 'linestyle' in kwargs.keys():
            if isinstance(kwargs['linestyle'], list) and len(kwargs['linestyle']) == 2:
                # Give a reasonable error message if someone messes this up
                raise N12Error("Cannot apply two line styles to one-particle trajectory")
        if 'connect' in kwargs.keys():
            raise N12Error("Cannot connect one-particle trajectory")

        return ax.plot(self._data[0, :, dims[0]], \
                       self._data[0, :, dims[1]], \
                       **kwargs)

class Trajectory_2N(Trajectory):
    """
    Behavior specific to two-locus trajectories
    """
    def _eff1Ndata(self):
        return self._data[0] - self._data[1]

    def relative(self):
        return Trajectory.fromArray(self._eff1Ndata(), **deepcopy(self.meta))

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
    Behavior specific to 1d trajectories
    """
    def _get_dimension_labels(self):
        if 'label' in self.meta.keys():
            return self.meta['label']
        else:
            return None

    def plot_spatial(self, *args, **kwargs):
        raise NotImplementedError("Cannot plot spatial trajectory for 1d trajectory. Use plot_vstime()")

class Trajectory_2d(Trajectory):
    """
    Behavior specific to 2d trajectories
    """
    def _get_dimension_labels(self):
        if 'label' in self.meta.keys():
            return [self.meta['label'] + " (x)", \
                    self.meta['label'] + " (y)"]
        else:
            return None

    def plot_spatial(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys() and 'label' in self.meta.keys():
            kwargs['label'] = self.meta['label']

        return self._raw_plot_spatial(ax, (0, 1), **kwargs)

class Trajectory_3d(Trajectory):
    """
    Behavior specific to 3d trajectories
    """
    def _get_dimension_labels(self):
        if 'label' in self.meta.keys():
            return [self.meta['label'] + " (x)", \
                    self.meta['label'] + " (y)", \
                    self.meta['label'] + " (z)"]
        else:
            return None

    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys() and 'label' in self.meta.keys():
            kwargs['label'] = self.meta['label']

        return self._raw_plot_spatial(ax, dims, **kwargs)

# Now we're getting to the fully concrete level. It's most likely that there is
# no further specialization here
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
