from copy import deepcopy

import numpy as np

class Trajectory:
    """
     + kwargs are written to meta
     + (N, T, d), (T, d), (T,) array for data
     + generic meta entries got converted to attributes
     + localization error is (d,), (N, d), (N, T, d)
    """

    ### Set up ###

    def __init__(self, data=None, t=None, *, localization_error=None, parity=None, **kwargs):
        if data is None:
            self.data = None
        else:
            data = np.array(data) # Note that this also copies the data
            if len(data.shape) > 3:
                raise ValueError("Array of shape {} cannot be interpreted as trajectory".format(str(data.shape)))
            elif len(data.shape) == 2:
                data = np.expand_dims(data, 0)
            elif len(data.shape) == 1:
                data = np.expand_dims(data, (0, 2))

            if t is None:
                self.data = data
            else:
                t = np.asarray(t)
                if len(t.shape) > 1:
                    raise ValueError(f"Argument t should be 1d array, but has shape {t.shape}")
                if len(t) != data.shape[1]:
                    raise ValueError(f"len(t) = {len(t)} is not equal to number of timepoints in data ({data.shape[1]})")

                if np.issubdtype(t.dtype, np.integer):
                    t = t - np.min(t)
                    data_patched = np.empty((data.shape[0], np.max(t)+1, data.shape[2]), dtype=float)
                    data_patched[:] = np.nan
                    data_patched[:, t, :] = data
                    self.data = data_patched
                else:
                    # This might be implemented at some point, the idea being that
                    # from given floating point timestamps one could identify a
                    # characteristic / median lag time and use that to rectify the
                    # timestamps into integer frames.
                    # While being useful for some applications, this does not
                    # produce a faithful representation of the input data, so maybe
                    # should not run by default without any comment / warning.
                    # Beyond the point above, handing a floating point array to the
                    # `t` argument also has a high likelihood of being
                    # unintentional, so throwing an error might be the best
                    # behavior.
                    raise NotImplementedError("Trajectory() does currently not support float time stamps")

        self.localization_error = localization_error
        self.parity = parity
        self.meta = kwargs

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
    def F(self):
        """ Number of valid frames """
        return np.sum(~np.any(np.isnan(self.data), axis=(0, 2)))

    @property
    def d(self):
        """ Number of dimensions """
        return self.data.shape[2]

    def __len__(self):
        return self.T

    def count_valid_frames(self):
        """
        Return the number of frames that have data.

        We regard a frame as unusable as soon as any data is missing. So this
        function counts the number of frames in the trajectory where none of
        the data is ``np.nan``.

        Returns
        -------
        int

        See also
        --------
        F
        """
        return self.F

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
        follows the numpy conventions (i.e. depends on the format of `!key`),
        while `d` is guaranteed to be present.
        """
        ret = self.data[:, key, :]
        # T squeezing is already done by numpy element-access
        try:
            ret = np.squeeze(ret, axis=0)
        except ValueError:
            pass
        return ret

    ### Modifiers ###

    def abs(self, order=None, keepmeta=None):
        """
        Modifier: 2-norm (or other norm, see `!order`)

        Parameters
        ----------
        order : float or None, optional
            order of the norm; see ``numpy.linalg.norm``'s `!ord` parameter.
            Defaults to 2-norm (i.e. Euclidean norm)
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

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
        traj = Trajectory(np.linalg.norm(self.data, ord=order, axis=2, keepdims=True))

        if keepmeta is not None:
            for key in keepmeta:
                traj.meta[key] = deepcopy(self.meta[key])

        if self.localization_error is not None:
            traj.localization_error = np.linalg.norm(self.localization_error, ord=order, axis=-1, keepdims=True)
        if self.parity is not None:
            traj.parity == 'even'

        return traj

    def diff(self, dt=1, keepmeta=None):
        """
        Modifier: displacements

        Calculate the displacements over `!dt` frames.

        Parameters
        ----------
        dt : integer
            the time lag to use for displacement calculation
            default: 1, i.e. frame to frame displacements
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

        Returns
        -------
        Trajectory

        See also
        --------
        abs, dims, relative
        """
        traj =  Trajectory(self.data[:, dt:, :] - self.data[:, :-dt, :])

        if keepmeta is not None:
            for key in keepmeta:
                traj.meta[key] = deepcopy(self.meta[key])

        if self.localization_error is not None:
            traj.localization_error = self.localization_error * np.sqrt(2)
        if self.parity is not None:
            traj.parity == 'even' if self.parity == 'odd' else 'odd'
        
        return traj

    def dims(self, key, keepmeta=None):
        """
        Modifier: select dimensions

        Parameters
        ----------
        key : list of int, or slice
            which dimensions to use. Attention: this cannot be a single `!int`. To
            get the ``i``-th spatial component, use ``traj.dims([i])``.
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

        Returns
        -------
        Trajectory

        See also
        --------
        abs, diff, relative
        """
        traj = Trajectory(self.data[:, :, key])

        if keepmeta is not None:
            for key in keepmeta:
                traj.meta[key] = deepcopy(self.meta[key])

        if self.localization_error is not None:
            traj.localization_error = self.localization_error.take(key, axis=-1)
        traj.parity == self.parity

        return traj

    def rescale(self, factor, keepmeta=None):
        """
        Modifier: rescale all data by a constant factor

        Parameters
        ----------
        factor : float
        
        Returns
        -------
        Trajectory
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

        See also
        --------
        abs, offset
        """
        traj = Trajectory(self.data * factor)

        if keepmeta is not None:
            for key in keepmeta:
                traj.meta[key] = deepcopy(self.meta[key])

        if self.localization_error is not None:
            traj.localization_error = self.localization_error * factor
        traj.parity = self.parity

        return traj

    def offset(self, off, keepmeta=None):
        """
        Modifier: shift the trajectory by some offset

        Parameters
        ----------
        off : float or array of shape ``(d,)``, ``(N, d)``, or ``(N, T, d)``
            the offset to add
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

        Returns
        -------
        Trajectory

        See also
        --------
        abs, rescale
        """
        off = np.array(off)
        ls = len(off.shape)
        if ls == 0: # happens if input is scalar
            off = np.expand_dims(off, (0, 1, 2))
        elif ls == 1:
            off = np.expand_dims(off, (0, 1))
        elif ls == 2:
            off = np.expand_dims(off, 1)

        traj = Trajectory(self.data + off)

        if keepmeta is not None:
            for key in keepmeta:
                traj.meta[key] = deepcopy(self.meta[key])

        traj.localization_error = self.localization_error
        traj.parity = self.parity

        return traj

    def relative(self, ref=None, keepmeta=None):
        """
        Modifier: Return `Trajectory` of pairwise distances

        Parameters
        ----------
        ref : int or None, optional
            which locus to use as reference. If ``None`` (the default) return
            sequential differences (i.e. ``2-1, 3-2, 4-3, ...``). Otherwise,
            returned distances are ``1-ref, 2-ref, ...``
        keepmeta : list of str or None, optional
            which entries from the `meta` dict to copy to the new trajectory

        Returns
        -------
        Trajectory

        Raises
        ------
        ValueError
            when called on trajectories with only a single locus

        See also
        --------
        abs, diff, dims
        """
        if self.N == 1:
            raise ValueError("relative() does not apply to single locus trajectories")

        loc = self.localization_error.copy() if self.localization_error is not None else np.array([0.])
        locN = len(loc.shape) > 1
        if not locN:
            loc = np.array(self.N*[loc])
        loc = loc**2

        if ref is None:
            traj = Trajectory(np.diff(self.data, axis=0))
            loc = np.sqrt(loc[:-1] + loc[1:])
        else:
            if not ref < self.N:
                raise ValueError(f"Cannot use locus {ref} as reference for {self.N}-locus trajectory")
            ind = [i for i in range(self.N) if i != ref]
            traj = Trajectory(self.data[ind] - self.data[[ref]])
            loc = np.sqrt(loc[ind] + loc[[ref]])

        if self.localization_error is None:
            traj.localization_error = None
        elif not locN:
            traj.localization_error = loc[0]
        else:
            traj.localization_error = loc

        traj.parity = self.parity

        return traj
