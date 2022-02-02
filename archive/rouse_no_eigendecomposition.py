"""
This module provides an implementation of the Rouse model.

Its only content is the `Model` class, so see there for more description

See also
--------
Model
"""

import numpy as np
import scipy.linalg
import scipy.integrate

class Model:
    r"""
    The Rouse model

    This model follows the linear Langevin equation

    .. math:: \dot{x}(t) = Ax(t) + F + \xi(t)\,,

    where :math:`x` is a vector of monomer positions and :math:`\xi` is
    Gaussian white noise of strength proportional `D` driving the system. The
    connectivity matrix :math:`A` describes which monomers are connected to
    which by harmonic bonds with spring constant `k`. The connectivity matrix
    is conveniently set up using `setup_free_chain` and `add_crosslinks`.
    Finally, :math:`F` can be used to incorporate an external force on
    individual monomers.

    This model implementation provides the following functionality:

    - setting up a spring network with arbitrary connectivity matrix
    - calculating the steady state ensemble.
    - propagating an arbitrary Gaussian ensemble forward in time
    - evolving a single conformation :math:`x` forward in time
    - analytically calculating MSD & ACF, also for arbitrary linear
      combinations of loci (e.g. MSD of the vector between two loci)
    - other auxiliary stuff, like giving equilibration time scales and contact
      probabilities

    Attributes
    ----------
    N : int
        number of monomers
    D : float
        diffusivity of an individual monomer (i.e. strength of the driving
        white noise)
    k : float
        default spring constant for harmonic bonds
    d : int
        spatial dimension
    A : (N, N) array, dtype=float, internal
        connectivity matrix. Initialized to a linear polymer by default
    F : (N, d) array, dtype=float, internal
        deterministic external force on each monomer. Zero by default

    Other Parameters
    ----------------
    setup_dynamics : bool
        whether to pre-calculate matrices needed for dynamics (propagation of
        ensembles / evolution of single conformation).
    add_bonds : list of bonds
        shortcut for adding bonds to the system before initializing dynamics.
        Corresponds to the `!links` argument of `add_crosslinks`, see there for
        more details.
    """
    # Implementation notes
    # --------------------
    # In a decision to favor time over space, we use simple numpy arrays
    # instead of sparse matrices (which might otherwise make sense for A).
    #
    # The check_dynamics mechanism is supposed to simply catch the most
    # straight-forward mistakes, i.e. it checks the parameter values and keeps track
    # of changes to the matrices through the member functions, but nothing else.
    #
    # This implementation uses S = 1 and AS = SA^T, i.e. we also assume symmetric A.
    #
    # D can be zero
    #
    # Numerics
    # --------
    # Since A is singular in many use cases, getting the numerics straight
    # requires some thought. Also some conceptual points such as "what's the
    # right steady state?" might become somewhat tricky. The numerical issue is
    # actually not limiting in our case: A is always real & symmetric, thus
    # orthogonally diagonalizable, which eliminates numerical issues with the
    # eigendecomposition (Moler & van Loan, 1978: "19 dubious ways ..."). This
    # should thus be the method of choice. Thus:
    #
    # - propagation requires M <-- B.M + G, C <-- B.C.B^T + Σ with B =
    #   exp(-k*A*dt). Following the above, we calculate B via EV decomposition.
    # - the "propagation matrices" G and Σ require integration of exp(A).
    #   Following the above, we again use EV decomposition. Note that small
    #   eigenvalues will have to be replaced by hand
    # - for the steady state, we have <x> = (kA)^{-1}.F and <x.x^T>_c =
    #   D/k*A^{-1}. These expressions correctly diverge when A is not
    #   invertible, as there is no steady state. In this case, however, it
    #   makes sense to fix all non-steady state degrees of freedom (i.e. the
    #   center of mass) at 0 and study the steady state for the remaining ones.
    #   This is exactly achieved by the Moore-Penrose pseudo-inverse of A,
    #   which we get from the EV decomposition by setting 1/λ = 0 for all
    #   vanishingly small eigenvalues λ.
    
    def __init__(self, N, D=1., k=1., d=3, setup_dynamics=True, add_bonds=None):
        self.N = N
        self.D = D
        self.k = k

        self._dynamics = {'needs_updating' : True}

        self.setup_free_chain(d)
        if add_bonds is not None:
            self.add_crosslinks(add_bonds)

        if setup_dynamics:
            self.update_dynamics()

    @property
    def d(self):
        return self.F.shape[1]

    def __eq__(self, other):
        for param in ['N', 'D', 'k']:
            if getattr(self, param) != getattr(other, param):
                return False
        for param in ['A', 'F']:
            if np.any(getattr(self, param) != getattr(other, param)):
                return False
        return True

    def __repr__(self):
        n_extra_bonds = np.count_nonzero(np.triu(self.A, k=2).flatten())
        rep = "rouse.Model(N={}, D={}, k={}, d={})".format(self.N, self.D, self.k, self.d)
        if n_extra_bonds > 0:
            rep += " with {} additional bonds".format(n_extra_bonds)
        return rep

####### Setting up a model and its dynamics (mostly for internal use)

    def setup_free_chain(self, d=3):
        """
        Set up connectivity matrix of a linear chain and zero external force

        Mostly used internally upon initialization

        Parameters
        ----------
        d : int
            spatial dimension

        See also
        --------
        add_crosslinks, add_tether
        """
        self._dynamics['needs_updating'] = True

        self.A = np.diagflat(self.N*[2.], k=0) \
                 + np.diagflat((self.N-1)*[-1.], k=-1) \
                 + np.diagflat((self.N-1)*[-1.], k= 1)
        self.A[0, 0] = self.A[-1, -1] = 1
        
        self.F = np.zeros((self.N, d))

    def add_crosslinks(self, links, k_rel=1.):
        """
        Add additional bonds to connectivity matrix

        Parameters
        ----------
        links : list of (monA, monB, k_rel) or (monA, monB) tuples
            bonds to add. Introduces an additional bond between ``monA`` and
            ``monB`` (integer indices < `N`) of strength ``k_rel*k``. Giving
            ``k_rel`` as third entry in the tuple overwrites use of the
            function argument `!k_rel` for this bond.
        k_rel : float > 0, optional
            strength of the new bonds, as multiple of the class attribute `!k`.

        See also
        --------
        add_tether
        """
        self._dynamics['needs_updating'] = True

        for link in links:
            myk_rel = k_rel if len(link) == 2 else link[2]
            self.A[link[0], link[0]] += myk_rel
            self.A[link[0], link[1]] -= myk_rel
            self.A[link[1], link[0]] -= myk_rel
            self.A[link[1], link[1]] += myk_rel

    def add_tether(self, mon=0, k_rel=1., point=(0., 0., 0.)):
        """
        Tether a monomer to a fixed point in space

        This simply introduces an additional harmonic bond between this monomer
        and the specified point. Note that you can give a large `!k_rel` to
        actually pin the monomer in place

        Parameters
        ----------
        mon : int
            the index of the monomer that should get the tether
        k_rel : float > 0, optional
            strength of the tether, in multiples of the default bond strength `!k`.
        point : (d,) array-like
            the point to tether to

        See also
        --------
        add_crosslinks
        """
        self._dynamics['needs_updating'] = True
        
        self.A[mon, mon] += k_rel
        self.F[mon] += k_rel * np.asarray(point)

    def update_dynamics(self, dt=1.):
        """
        Pre-calculate dynamic matrices

        This sets the model up for evaluation of model dynamics, such as
        evolving a conformation or propagating an ensemble. It should be called
        after any modification to model parameters, which is handled
        automatically for internal modifications (such as when `add_crosslinks`
        is called). If you modify anything externally, recommended behavior is
        to simply set ``model._dynamics['needs_updating'] = True``, which will
        ensure that stuff is recalculated as needed.

        Ultimately, this is just a wrapper for `update_invA`, `update_G`, and
        `update_Sig`. If you know what you're doing, you can also just call
        these individually as needed.

        Parameters
        ----------
        dt : float
            time step to prepare for

        See also
        --------
        check_dynamics
        """
        self._dynamics = {
                'needs_updating' : False,
                'N' : self.N,
                'D' : self.D,
                'k' : self.k,
                'dt' : dt,
                'emkAt' : scipy.linalg.expm(-self.k*self.A*dt)
                }

        self.update_invA()
        self.update_G()
        self.update_Sig()

    def update_invA(self):
        """
        Calculate inverse of connectivity matrix. Internal use mostly

        If `!A` is not invertible (which is common), the value of the inverse
        is set to ``None``.

        See also
        --------
        update_dynamics
        """
        if np.isclose(scipy.linalg.det(self.A), 0):
            self._dynamics['invA'] = None
        else:
            self._dynamics['invA'] = scipy.linalg.inv(self.A)

    def update_G(self, override_full_update=False):
        """
        Update the discrete version of `!F`. Internal use mostly

        Parameters
        ----------
        override_full_update : bool
            pretend that this was a full update. Useful when you changed only
            `!F` since the last update, to avoid recalculating the covariance
            matrix from `!A`.

        See also
        --------
        update_Sig
        """
        if not np.any(self.F):
            G = np.zeros_like(self.F)
        elif self._dynamics['invA'] is not None:
            G = (np.eye(self.N) - self._dynamics['emkAt']) \
                @ (self._dynamics['invA']/self.k) @ self.F
        else:
            def integrand(tau):
                return scipy.linalg.expm(-self.k*self.A*tau) @ self.F
            G = scipy.integrate.quad_vec(integrand, 0, self._dynamics['dt'])[0]
        
        self._dynamics['G'] = G

        if override_full_update: # pragma: no cover
            self._dynamics['needs_updating'] = True

    def update_Sig(self, override_full_update=False):
        """
        Update the propagation matrix. Internal use mostly

        Parameters
        ----------
        override_full_update : bool
            pretend that this was a full update. Useful when you changed only
            `!A` since the last update, to avoid recalculating `!G`

        See also
        --------
        update_G
        """
        if self.D > 0:
            if self._dynamics['invA'] is not None:
                B = self._dynamics['emkAt']
                Sig = (np.eye(self.N) - B@B) @ self._dynamics['invA'] * self.D/self.k
            else:
                def integrand(tau):
                    return scipy.linalg.expm(-2*self.k*self.A*tau)
                Sig = 2*self.D * scipy.integrate.quad_vec(integrand, 0, self._dynamics['dt'])[0]
            
            self._dynamics['LSig'] = scipy.linalg.cholesky(Sig, lower=True)
        else:
            Sig = np.zeros((self.N, self.N))
            self._dynamics['LSig'] = Sig

        self._dynamics['Sig'] = Sig

        if override_full_update: # pragma: no cover
            self._dynamics['needs_updating'] = True

    def check_dynamics(self, dt=None, run_if_necessary=True):
        """
        Check that dynamics are set up properly

        Parameters
        ----------
        dt : float or None
            the time step we should be set up for. Specify ``None`` to accept
            whatever was the value set up
        run_if_necessary : bool
            controls the behavior when dynamics are not set up. If set to
            ``True`` (default), `update_dynamics` is simply run with the
            settings we have. If set to ``False`` raise a ``RuntimeError`` if
            dynamics are not (correctly) set up for the current model.

        Raises
        ------
        RuntimeError
            if something is not set up properly

        See also
        --------
        update_dynamics
        """

        if dt is None:
            try:
                dt = self._dynamics['dt']
            except KeyError:
                raise RuntimeError("Call update_dynamics before running")

        try:
            if dt != self._dynamics['dt']:
                self._dynamics['needs_updating'] = True

            for key in ['N', 'D', 'k']:
                if self._dynamics[key] != getattr(self, key):
                    self._dynamics['needs_updating'] = True
        except KeyError:
            self._dynamics['needs_updating'] = True
        
        if self._dynamics['needs_updating']:
            if run_if_necessary:
                self.update_dynamics(dt)
            else:
                raise RuntimeError("Model changed since last call to update_dynamics()")

####### Propagation of an ensemble

    def steady_state(self):
        """
        Calculate the steady state ensemble

        Since our model is linear and driven by Gaussian noise, the steady
        state ensemble will always be Gaussian, and as such given by a mean
        vector and covariance matrix over all monomers.
        
        Typical models (e.g. a free polymer) will not have a true steady state,
        because the center of mass keeps diffusing. Still, the "internal"
        degrees of freedom reach a steady state, which might be useful. To that
        end, if there is no true steady state we tether monomer 0 to the
        coordinate origin to obtain the steady state. This gives the correct
        covariance structure for all internal degrees of freedom.

        Returns
        -------
        M : (N, d) np.ndarray, dtype=float
            mean position of each monomer
        C : (N, N) np.ndarray, dtype=float
            covariance between monomer coordinates in each dimension. Note that
            we assume independence of spatial dimensions, such that the full
            covariance ``C_ijab := <x_ia*x_jb>_c`` (with monomer indices i, j
            and spatial indices a, b) is written as the tensor product ``C_ijab
            = C_ij*δ_ab`` with Kronecker's δ. The matrix returned by this
            function is ``C_ij``.
        """
        # automatically introduce tether if there is no steady state
        if not 'invA' in self._dynamics.keys():
            self.update_invA()
        invA = self._dynamics['invA']
        if invA is None:
            self.A[0, 0] += 1.0
            invA = scipy.linalg.inv(self.A)
            self.A[0, 0] -= 1.0

        return (invA @ self.F / self.k,
                invA * self.D / self.k)

    def propagate_M(self, M, dt=None, check_dynamics=True):
        """
        Propagate the mean of a Gaussian ensemble

        Parameters
        ----------
        M : (N, d) np.ndarray, dtype=float
            mean conformation
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        M : (N, d) np.ndarray, dtype=float
            like input `!M`, but propagated by `!dt`.

        See also
        --------
        propagate, propagate_C, steady_state
        """
        if check_dynamics:
            self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        G = self._dynamics['G']
        return B @ M + G

    def propagate_C(self, C, dt=None, check_dynamics=True):
        """
        Propagate the covariance of a Gaussian ensemble

        Parameters
        ----------
        C : (N, N) np.ndarray, dtype=float
            current covariance
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        C : (N, N) np.ndarray, dtype=float
            like input `!C`, but propagated by `!dt`.

        See also
        --------
        propagate, propagate_M, steady_state
        """
        if check_dynamics:
            self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        Sig = self._dynamics['Sig']
        return B @ C @ B + Sig

    def propagate(self, M, C, dt=None, check_dynamics=True):
        """
        Propagate a Gaussian ensemble one time step

        This is simply a wrapper for `propagate_M` and `propagate_C`.

        Parameters
        ----------
        M : (N, d) np.ndarray, dtype=float
            current mean conformation
        C : (N, N) np.ndarray, dtype=float
            current covariance
        dt : float or None, optional
            the time step to propagate for. When ``None`` uses the time step
            set through ``update_dynamics`` (default, recommended)
        check_dynamics : bool
            whether to check for correct setup of precalculated matrices. Can
            be disabled for performance when its otherwise clear that the setup
            is correct

        Returns
        -------
        M, C
            like input, but propagated by `!dt`.

        See also
        --------
        propagate_M, propagate_C, steady_state
        """
        return propagate_M(M, dt, check_dynamics),
               propagate_C(C, dt, check_dynamics=False) # if needed, dynamics were already checked in M step

####### Evolution of a single conformation

    def conf_ss(self):
        """
        Draw a conformation from steady state

        Returns
        -------
        (N, d) np.ndarray, dtype=float
            the drawn conformation

        See also
        --------
        steady_state
        """
        M, C = self.steady_state()
        L = scipy.linalg.cholesky(C, lower=True)
        return M + L @ np.random.normal(size=(self.N, self.d))

    def evolve(self, conf, dt=None):
        """
        Evolve a conformation forward one time step

        Parameters
        ----------
        conf : (N, d) np.ndarray, dtype=float
            the conformation to start from
        dt : float or None
            the time step to evolve for. When ``None`` uses the time step set
            through ``update_dynamics`` (default, recommended)

        Returns
        -------
        conf
            like input, but evolved for `!dt`.

        See also
        --------
        conf_ss
        """
        self.check_dynamics(dt, run_if_necessary=True)
        B = self._dynamics['emkAt']
        L = self._dynamics['LSig']
        return B @ conf + L @ np.random.normal(size=conf.shape)

####### MSD and related stuff

    def _analytical_MSD(self, dts, w=None):
        r"""
        Calculate MSD directly from analytical expression. Internal use mostly.

        The analytical solution for the MSD at time lag :math:`\Delta t` is
        given by

        .. math:: MSD(\Delta t) \equiv \left\langle\left(x(t+\Delta t) - x(t)\right)^{\otimes 2}\right\rangle = 2(I + B)^{-1}\Sigma + G\otimes G

        with :math:`B \equiv \mathrm{e}^{-kA\Delta t}`, and :math:`G` and
        :math:`\Sigma` being the finite time step propagators calculated by the
        ``update_*`` functions.

        Note that this form (with +G^2 term) is really only valid in the
        absence of a steady state, where we can substitute ``<x> = 0`` due to
        translational invariance. If there was a steady state, this term would
        become -G^2, but in that case we use ACF anyways.

        For many use cases (namely when the dof of interest reach steady state)
        it is (way) faster to calculate the MSD via the ACF. This function
        therefore serves only as a last resort.

        Under regular circumstances, there should be no need to call this
        function directly, use `MSD` instead, which correctly uses the faster
        version whenever possible. See there for more documentation

        See also
        --------
        MSD
        """
        dts = np.sort(dts)
        if dts[0] < 0:
            raise ValueError("dt should be > 0")
        Bs = [scipy.linalg.expm(-self.k*self.A*dt) for dt in dts]

        # Calculate Sigmas
        def integrand(tau):
            return scipy.linalg.expm(-2*self.k*self.A*tau)
        res, err, full_info = scipy.integrate.quad_vec(
                integrand, 0, dts[-1],
                points = dts,
                full_output = True)
        Sigs = [2*self.D * np.sum(full_info.integrals[full_info.intervals[:, 1] <= dt],
                                  axis = 0)
                for dt in dts]

        # Calculate Gs
        if not np.any(self.F):
            Gs = [np.zeros_like(self.F) for dt in dts]
        else:
            def integrand(tau):
                return scipy.linalg.expm(-self.k*self.A*tau) @ self.F
            Gs = [scipy.integrate.quad_vec(integrand, 0, dt)[0] for dt in dts]

        # Note: to quickly check this formula, use the "dirty trick" of
        # assuming a steady state where there is none, i.e. replace <xx> with
        # (1-B^2)^(-1)Σ and eliminate factors. The final expression, as opposed
        # to the steady state itself, does not diverge as B --> 1 and can thus
        # be analytically continued, i.e. stays valid even if the steady state
        # we had to assume intermittently does not exist
        # Note: the argument for obtaining the +G^2 here is a bit sketchy...
        # TODO: check this expression numerically
        xxs = [2*scipy.linalg.inv(np.eye(self.N) + B) @ Sig + G[:, None]*G[None, :] for B, Sig, G in zip(Bs, Sigs, Gs)]

        if w is None:
            return self.d * np.array(xxs)
        else:
            return self.d * np.array([w @ xx @ w for xx in xxs])

    def MSD(self, dts, w=None):
        """
        Calculate MSD for given degrees of freedom

        This is simply an evaluation of analytical expressions. Which path
        exactly we take depends on the system, see Notes.

        Parameters
        ----------
        dts : (T,) array-like, dtype=float
            the time lags to evaluate the MSD at. Entries should be greater or
            equal to zero
        w : (N,) np.ndarray, dtype=float
            measurement vector. Use this to specify the linear combination of
            monomers whose MSD you are interested in. See Examples. If
            unspecified, the function returns the full covariance matrix at lag
            Δt.

        Returns
        -------
        (T,) np.ndarray or (T, N, N) np.array
            the MSD, either as scalar function evaluated at `!dts` or matrix
            valued (if no measurement vector `!w` was specified)

        Notes
        -----
        When the degrees of freedom we are interested in (that is, ``w @ x`` if
        `!w` is specified, otherwise simply ``x``) reach steady state, we can
        calculate the MSD as ``MSD(Δt) = 2*( ACF(0) - ACF(Δt) )``, for which we
        can use the `ACF` function. This is reasonably fast, since we "just"
        have to calculate matrix exponentials. The function `_analytical_MSD`
        in contrast gives the correct MSD also when there is really no steady
        state (e.g. a single monomer on a free chain), but it is expensive to
        evaluate, since it involves integrals that have to be approximated
        numerically. We therefore use this function (`MSD`) as a wrapper that
        checks whether a steady state exists and uses the appropriate
        calculation. Note that the result is the analytical solution in either
        case.

        Examples
        --------
        >>> model = Model(N=25)
        ... dts = np.array([0, 1, 2, 10]) # some sample lag times
        ...
        ... # Calculate full matrices. This is usually not as relevant
        ... msd = model.MSD(dts)
        ...
        ... # MSD of the middle monomer
        ... w = np.zeros(model.N)
        ... w[12] = 1
        ... msd = model.MSD(dts, w)
        ...
        ... # MSD of the end-to-end vector
        ... w = np.zeros(model.N)
        ... w[0]  =  1
        ... w[-1] = -1
        ... msd = model.MSD(dts, w)
        """
        if w is None or np.sum(w) != 0:
            # COM motion is relevant. Do we have a steady state?
            try:
                self.check_dynamics(run_if_necessary=False)
            except:
                self.update_invA()
            if self._dynamics['invA'] is None:
                return self._analytical_MSD(dts, w)

        return 2*( self.ACF([0], w) - self.ACF(dts, w) )

    def ACF(self, dts, w=None):
        # Note that this is valid also if there is no actual steady state for
        # the chain, but for the variable we're interested in, such as when
        # we're looking at end to end distances.
        dts = np.sort(dts)
        Bs = []
        i_start, i_end = 0, len(dts)
        if dts[0] < 0:
            raise ValueError("dt should be > 0")
        while i_start < i_end and dts[i_start] == 0:
            Bs += [np.eye(self.N)]
            i_start += 1
        while i_end > i_start and dts[i_end-1] == np.inf:
            i_end -= 1

        Bs += [scipy.linalg.expm(-self.k*self.A*dt) for dt in dts[i_start:i_end]]
        Bs += [np.zeros((self.N, self.N)) for _ in range(len(dts)-i_end)]

        # steady_state() automatically detects absence of true steady state and
        # then gives "internal" steady state by adding an extra bond to the
        # origin
        M, C = self.steady_state()
        C = C - M[:, None]*M[None, :]

        if w is None:
            Bs = np.array(Bs)
            return self.d * Bs @ C
        else:
            Cw = C @ w
            return self.d * np.array([(w @ B) @ Cw for B in Bs])

####### Length & Time scales

    def timescales(self):
        return {
                't_microscopic' : 1/self.k,
                't_Rouse' : (self.N / np.pi)**2 / self.k,                
                't_equilibration' : np.pi*self.N**2 / (4*self.k),
                }

    def Gamma(self):
        return 2*self.d*self.D / np.sqrt(np.pi*self.k)

    def Gamma_2loci(self):
        return 2*self.Gamma()

    def rms_Ree(self, L=None):
        if L is None:
            L = self.N-1
        return np.sqrt(self.d*self.D/self.k * L)

####### Auxiliary things

    def contact_probability(self):
        _, J = self.steady_state()
        Jii = np.tile(np.diagonal(J), (len(J), 1))
        with np.errstate(divide='ignore'): # the diagonal gives np.inf; that's fine
            return (Jii + Jii.T - 2*J)**(-self.d/2)
