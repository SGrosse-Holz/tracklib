import os,sys
import importlib

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

    Note: this implementation should be able to deal with missing data, i.e.
    np.nan's

    Note: the different possible types of trajectories (1 or 2 particles, 1, 2
    or 3 dimensions) are implemented as subclasses, thus facilitating
    customization

    Note: this class implements the Sequence interface, i.e. it can be used
    much like a list. Element access returns (N, d) arrays as data points.
    """
    def __init__(self, label=None):
        self.label = label
        self._data = None

        # Units
        self.dt = 1
        self.dx = 1
        self.parity = 'even'

    def __len__(self):
        return self._data.shape[1]

    def __getitem__(self, key):
        return self._data[:, key, :]

    @property
    def N(self):
        return self._data.shape[0]
    @property
    def T(self):
        return self._data.shape[1]
    @property
    def d(self):
        return self._data.shape[2]

    @classmethod
    def fromArray(cls, array, label=None, parity='even'):
        """
        Create Trajectory from numpy array. array should have shape (N, T, d),
        (T, d), or (T,), N = 1 or 2, d = 1, 2, 3.
        Note that we return instances of the child classes Trajectory_xNxd.
        """
        array = np.array(array)
        if len(array.shape) > 3:
            raise ValueError("Array of shape {} cannot be interpreted as trajectory".format(str(array.shape)))
        elif len(array.shape) == 2:
            array = np.expand_dims(array, 0)
        elif len(array.shape) == 1:
            array = np.expand_dims(array, (0, 2))

        try:
            obj = getattr(importlib.import_module(cls.__module__), "Trajectory_{:d}N{:d}d".format(array.shape[0], array.shape[2]))(label)
        except AttributeError:
            raise ValueError("Could not instantiate trajectory with (N, T, d) = {}".format(str(array.shape)))

        obj._data = array
        obj.parity = parity
        return obj

    def copyMeta(self, target):
        """
        Copy all the meta information (like label, dx, dt) to another object.
        """
        for key in self.__dict__.keys():
            if key is not '_data':
                setattr(target, key, self.__dict__[key])

    @abstractmethod
    def _eff1Ndata(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_dimension_labels(self):
        raise NotImplementedError()

    def plot_vstime(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys():
            kwargs['label'] = self._get_dimension_labels()

        tplot = np.arange(self.T)*self.dt
        return ax.plot(tplot, self._eff1Ndata()*self.dx, **kwargs)

    @abstractmethod
    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        raise NotImplementedError()

    def msd(self, memo=True, giveN=False):
        """
        Memoized method to calculate single trajectory MSD
        Use memo=False to circumvent memoization and explicitly (re-)calculate
        MSD
        """
        if not hasattr(self, '_msdN') or not memo:
            self._msdN = util.msd(self._eff1Ndata(), giveN=True)
            self._msdN = (self._msdN[0]*self.dx**2, self._msdN[1])

        if giveN:
            return self._msdN
        else:
            return self._msdN[0]

    def absTrajectory(self):
        """
        Give a new trajectory representing the magnitude of the current one,
        i.e. absolute value for 1d, 2-norm for 2d and 3d. Note that for
        multiple particles, this will take the magnitude of the individual
        particles' trajectories. To get the relative distance, use
        Trajectory_2N.relativeTrajectory().absTrajectory()
        """
        obj = Trajectory.fromArray(np.sqrt(np.sum(self._data**2, axis=2, keepdims=True)))
        self.copyMeta(obj)
        return obj

    def relativeTrajectory(self):
        raise NotImplementedError("relativeTrajectory() does not apply to {}".format(type(self).__name__))

    def diffTrajectory(self, dt=1):
        """
        Given a new trajectory of displacements of the current one
        Note that this switches parity.
        """
        obj = Trajectory.fromArray(self._data[:, dt:, :] - self._data[:, :-dt, :])
        self.copyMeta(obj)

        if obj.parity is 'even':
            obj.parity = 'odd'
        else:
            obj.parity = 'even'

        return obj

# Specialize depending on particle number or dimension, which changes behavior
# of some functions that can be overridden here
class N12Error(ValueError):
    pass

class Trajectory_1N(Trajectory):
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

        return ax.plot(self._data[0, :, dims[0]]*self.dx, \
                       self._data[0, :, dims[1]]*self.dx, \
                       **kwargs)


class Trajectory_2N(Trajectory):
    def _eff1Ndata(self):
        return self._data[0] - self._data[1]

    def relativeTrajectory(self):
        obj = Trajectory.fromArray(self._eff1Ndata())
        self.copyMeta(obj)
        return obj

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
        lines = ax.plot(self._data[0, :, dims[0]]*self.dx, \
                        self._data[0, :, dims[1]]*self.dx, \
                        **kwargs)
        kwargs['color'] = lines[0].get_color()

        if makeLegend:
            ax.plot(0, 0, label=label, **kwargs)

        # Plot second particle
        kwargs['linestyle'] = linestyles[1]
        lines.append(ax.plot(self._data[1, :, dims[0]]*self.dx, \
                             self._data[1, :, dims[1]]*self.dx, \
                             **kwargs))

        return lines

class Trajectory_1d(Trajectory):
    def _get_dimension_labels(self):
        return self.label

    def plot_spatial(self, *args, **kwargs):
        raise NotImplementedError("Cannot plot spatial trajectory for 1d trajectory. Use plot_vstime()")

class Trajectory_2d(Trajectory):
    def _get_dimension_labels(self):
        if self.label is None:
            return None
        else:
            return [self.label + " (x)", \
                    self.label + " (y)"]

    def plot_spatial(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys():
            kwargs['label'] = self.label

        return self._raw_plot_spatial(ax, (0, 1), **kwargs)

class Trajectory_3d(Trajectory):
    def _get_dimension_labels(self):
        if self.label is None:
            return None
        else:
            return [self.label + " (x)", \
                    self.label + " (y)", \
                    self.label + " (z)"]

    def plot_spatial(self, ax=None, dims=(0, 1), **kwargs):
        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs.keys():
            kwargs['label'] = self.label

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














