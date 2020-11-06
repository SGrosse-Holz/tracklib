"""
This module provides a Rouse model with a switchable extra bond.

Notes
-----
**Nomenclature:** a 'trace' is a `!np.ndarray` of shape (T,). Consequently, a
'looptrace' is a `!np.ndarray` with ``shape = (T,)`` and ``dtype = bool``,
indicating at which points the extra bond is present.

Similarly to `Model.propagate`, `likelihood` is just an alias for one of two
methods of calculating the likelihood: `!_likelihood_filter` (default) uses a
Kalman filter to run through the trajectory and calculate the likelihood
iteratively, while `!_likelihood_direct` just evaluates the probability density
over the space of all trajectories of a given length. These two methods, while
not giving the exact same result, are equivalent in the sense that they differ
by a constant offset. You can change which method is used by reassigning
`rouse.likelihood`:

>>> rouse.likelihood = rouse._likelihood_filter # use Kalman filter (default)
... rouse.likelihood = rouse._likelihood_direct # use high-dimensional Rouse PDF

"""

import numpy as np
import scipy.linalg
import scipy.integrate

class Model:
    r"""
    The Rouse model with extra bond

    Main capabilities of this class: `propagate` an ensemble, or `evolve` a
    single conformation. In the first case, the PDF of the ensemble over
    conformation space is Gaussian, so it is fully specified by a mean
    conformation and a covariance matrix. These follow deterministic equations
    (exponential relaxation towards steady state), whose initial value problem
    has an explicit solution. In the second case, we start from an explicit
    conformation and follow the Rouse evolution for a definite time. This is
    also done by utilizing the explicit solution to these EOM.

    Remember to call `setup_dynamics` if changing parameters after
    initialization.

    Attributes
    ----------
    N : int
        number of monomers
    D : float
        diffusion constant of a single free monomer
    k : float
        strength of the backbone bonds
    k_extra : float
        strength of the extra bond

    Other Parameters
    ----------------
    setup : bool, optional
        whether to run `setup_dynamics` after initialization.
    extrabond : 2-tuple of int, optional
        the position of the extra bond. By default this is between the ends of
        the chain.

    Notes
    -----
    If ``k_extra < -k/(N-1)`` the connectivity matrix will have a negative
    eigenvalue. In this situation there is a repulsive bond between the ends
    that is stronger than the backbone, so it will tear the polymer apart. The
    model can still be useful in this limit, but if the extra bond is switched
    on too much, it will tear everything apart. This can also lead to negative
    covariance matrices.
    
    Relation between the Rouse model with physical constants and the parameters
    used here: if we write the equation of motion as

    .. math:: \gamma\dot{x} = \kappa Ax + \xi \,,\qquad \left\langle\xi(t)\xi(t')\right\rangle = 2\gamma k_\text{B}T\delta(t-t') \,,

    then we have

    .. math:: k = \frac{\kappa}{\gamma} \,,\qquad D = \frac{k_\text{B}T}{\gamma} \,.

    For a Rouse `Model` ``model``, the following operations are defined:

    ``model == other_model``
        comparison. Gives ``True`` if all the parameters are equal.
    ``repr(model)``
        give a string representation.

    There are two methods that can be used for propagation: solve the mean and
    covariance equations numerically, or use the known analytical solutions.
    The latter involves calculating matrix exponentials, but is a factor 8
    faster in tests. Both are implemented, as `!_propagate_ode` and
    `!_propagate_exp` respectively. `Model.propagate` is then simply assigned to
    one of them, `!_propagate_exp` by default.
    """

    def __init__(self, N, D, k, k_extra, setup=True, extrabond=(0, -1)):
        self.N = N
        self.D = D
        self.k = k
        self.k_extra = k_extra

        self.bondpos = extrabond
        if setup:
            self.setup_dynamics()
    
    def __eq__(self, other):
        for param in ['N', 'D', 'k', 'k_extra', 'bondpos']:
            if getattr(self, param) != getattr(other, param):
                return False
        return True
        
    def __repr__(self):
        if self.bondpos == (0, -1):
            return "rouse.Model(N={}, D={}, k={}, k_extra={})".format(self.N, self.D, self.k, self.k_extra)
        else:
            return "rouse.Model(N={}, D={}, k={}, k_extra={}, extrabond={})".format(self.N, self.D, self.k, self.k_extra, str(self.bondpos))
        
    def give_matrices(self, bond, tethered=False):
        """
        Assemble the Rouse matrices (connectivity and noise covariance).
        
        Parameters
        ----------
        bond : bool
            is the extra bond present?
        tethered : bool, optional
            set to True to tether monomer 0 to the origin, i.e. get a
            non-singular connectivity. This is important if we want to sample
            from steady state (an untethered chain has COM diffusion and thus
            no steady state).
            
        Returns
        -------
        A : (`N`, `N`) np.ndarray
            the connectivity matrix
        S : (`N`, `N`) np.ndarray
            covariance of the noise

        See also
        --------
        steady_state, conf_ss
        """
        A = np.diagflat(self.N*[-2.]) + np.diagflat((self.N-1)*[1.], k=1) + np.diagflat((self.N-1)*[1.], k=-1)
        A[ 0,  0] += 1
        A[-1, -1] += 1
        A = self.k * A
        
        if bond:
            b0 = self.bondpos[0]
            b1 = self.bondpos[1]
            A[b0, b0] -= self.k_extra
            A[b0, b1] += self.k_extra
            A[b1, b0] += self.k_extra
            A[b1, b1] -= self.k_extra
            
        if tethered:
            A[0, 0] -= self.k
            
        S = 2*self.D*np.eye(self.N)
        
        return A, S

    def setup_dynamics(self, dt=1.):
        """
        Pre-calculate some matrices needed for propagation/evolution.

        By default, this function is called by the constructor, so in most
        cases it has only to be invoked by the user if they change parameters
        in the model.
        
        Parameters
        ----------
        dt : float
            the timestep that we use for propagtion
        """
        def integrand(tau, t, A, S):
            ettA = scipy.linalg.expm((t-tau)*A)
            return ettA @ S @ ettA.T

        self._propagation_memo = {
                'model' : Model(self.N, self.D, self.k, self.k_extra, setup=False),
                'dt' : dt,
                }
        for bond in [True, False]:
            A, S = self.give_matrices(bond)
            Sig = scipy.integrate.quad_vec(lambda tau : integrand(tau, dt, A, S), 0, dt)[0]
            etA = scipy.linalg.expm(dt*A)
            self._propagation_memo[bond] = {
                    'etA' : etA,
                    'Sig' : Sig,
                    'LSig' : scipy.linalg.cholesky(Sig, lower=True),
                    }

    def check_setup_called(self, dt=None, run_if_necessary=False):
        """
        Check whether `setup_dynamics` has been called.

        Mostly internal use. Checks whether `setup_dynamics` has been called
        and if so, whether parameters have changed since then.

        This function does not have a return value, it simply raises a
        `!RuntimeError` if something does not work out.

        Parameters
        ----------
        dt : float
            the time step that we need to be set up for. Set to ``None``
            (default) to omit this check.
        run_if_necessary : bool, optional
            if ``True``, instead of raising an error, just run the setup. If ``dt
            is None``, will try to use the stored value from the last call to
            `setup_dynamics`, if there is one.

        Raises
        ------
        RuntimeError
            if anything changed and ``run_if_necessary == False``

        See also
        --------
        setup_dynamics

        Notes
        -----
        This can be used instead of directly calling `setup_dynamics` in cases
        where we are operating on a user specified model and might want to
        change some parameters, then call `setup_dynamics`, but don't
        necessarily know the timestep the user used in the setup.
        """
        if not hasattr(self, '_propagation_memo'):
            if run_if_necessary and dt is not None:
                self.setup_dynamics(dt)
            else:
                raise RuntimeError("Did not call setup_dynamics before trying to evaluate dynamics")
        elif not self == self._propagation_memo['model'] or (dt != self._propagation_memo['dt'] and dt is not None):
            if dt is None:
                dt = self._propagation_memo['dt']
            if run_if_necessary:
                self.setup_dynamics(dt)
            else:
                raise RuntimeError("Parameter values changed since last call to setup_dynamics()")

    ### Propagation of ensemble mean + sem covariance ###

    def steady_state(self, bond=False):
        r"""
        Return mean and covariance of the steady state.

        Returns
        -------
        M : (`N`,) `!np.ndarray`
            the mean for each monomer (which is zero, this is mostly for
            convenience).
        C : (`N`, `N`) `!np.ndarray`
            the long run covariance

        See also
        --------
        propagate, give_matrices

        Notes
        -----
        The long run covariance :math:`C` is given by

        .. math:: C = \frac{1}{2k}A^{-1}S \,.

        To ensure invertibility of the connectivity matrix, we tether one end
        of the chain to the origin.
        """
        A, S = self.give_matrices(bond, tethered=True)
        C = -0.5*scipy.linalg.inv(A) @ S
        return np.zeros(C.shape[0]), C
    
    def _propagate_ode(self, M0, C0, t, bond):
        """
        Propagation by numerically solving the ODEs
        """
        A, S = self.give_matrices(bond)
        
        def eq_mean(_, M):
            return A @ M
        def eq_cov(_, C):
            C = np.reshape(C, (self.N, self.N))
            dC = S + A @ C + C @ A.T
            return np.reshape(dC, (self.N*self.N,))
        
        M1_odesol = scipy.integrate.solve_ivp(eq_mean, (0, t), M0, \
                                       method='LSODA')
        M1 = M1_odesol.y[:, -1]
        C1_odesol = scipy.integrate.solve_ivp(eq_cov, (0, t), np.reshape(C0, (self.N*self.N,)), \
                                       method='LSODA') # RK45 gives non-positive solutions
        C1 = C1_odesol.y[:, -1]
        C1.shape = (self.N, self.N)

        return M1, C1

    def _propagate_exp(self, M0, C0, t, bond):
        """
        Propagation by using the analytical solutions
        """
        self.check_setup_called(t, run_if_necessary=True)
        etA = self._propagation_memo[bond]['etA']
        Sig = self._propagation_memo[bond]['Sig']
        return etA @ M0, etA @ C0 @ etA.T + Sig
        
    propagate = _propagate_exp
    propagate.__doc__ = """
        Propagate the ensemble ``(M0, C0)`` for a time t.

        This is simply assigned to one of `!_propagate_ode` or `!_propagate_exp`
        (default), which either solve the equations of motion numerically, or
        use the known analytical solution. Users can reassign this attribute:

        >>> model = Model(10, 1, 1, 1)
        ... model.propagate = model._propagate_ode # for numerical solution
        ... model.propagate = model._propagate_exp # for evaluation of analytical solution

        Parameters
        ----------
        M0 : (`N`,) np.ndarray
            the initial value for the mean conformation
        C0 : (`N`, `N`) np.ndarray
            the initial value for the covariance
        t : float
            the time to propagate for
        bond : boool
            whether the extra bond should be present or absent

        Returns
        -------
        M1 : (`N`,) np.ndarray
            the propagated mean
        C1 : (`N`, `N`) np.ndarray
            the propagated covariance

        See also
        --------
        steady_state
        """
    
    ### Evolution of single conformations ###
    
    def conf_ss(self, bond=False, d=3):
        """
        Draw a conformation from steady state.
        
        Parameters
        ----------
        bond : bool, optional
            whether the extra bond is present
        d : int, optional
            the number of spatial dimensions

        Returns
        -------
        (`N`, d) np.ndarray
            the sampled conformation

        See also
        --------
        evolve, conformations_from_looptrace

        Notes
        -----
        This uses the ``tethered`` keyword to `give_matrices` to ensure an
        invertible connectivity matrix.
        """
        _, C = self.steady_state(bond)
        L = scipy.linalg.cholesky(C, lower=True)
        return L @ np.random.normal(size=(self.N, d))

    def evolve(self, conf, bond=False, dt=None):
        """
        Evolve the conformation conf.

        Parameters
        ----------
        conf : (`N`, ...) np.ndarray
            the conformation(s) to evolve
        bond : bool, optional
            whether the extra bond should be present
        dt : float, optional
            the time step to take. Set this to ``None`` (the default) to use
            the time step given to `setup_dynamics`. This should be
            default usage, since recalculating the dynamic matrices at each
            step is expensive.

        Returns
        -------
        (`N`, ...) np.ndarray
            the evolved conformation

        See also
        --------
        conf_ss, conformations_from_looptrace, setup_dynamics
        """
        self.check_setup_called(dt=dt, run_if_necessary=True)

        B = self._propagation_memo[bond]['etA']
        L = self._propagation_memo[bond]['LSig']
        return B @ conf + L @ np.random.normal(size=conf.shape)

    def conformations_from_looptrace(self, looptrace, d=3):
        """
        A generator yielding conformations for a given looptrace.
        
        We start from steady state with/without the extra bond according to
        ``looptrace[0]`` and `evolve` from there. This means that
        ``looptrace[i]`` can be seen as the indicator for whether or not there
        is a bond present between time points ``i-1`` and ``i``.

        Parameters
        ----------
        looptrace : (T,) iterable of bool
            the looptrace to simulate for
        d : int, optional
            the dimensionality of the conformations to generate

        Yields
        ------
        (`N`, d) np.ndarray
            the conformation at the corresponding time step.

        See also
        --------
        conf_ss, evolve
        """
        conf = self.conf_ss(looptrace[0], d)
        for bond in looptrace:
            conf = self.evolve(conf, bond)
            yield conf

# We define the likelihood outside the Model class mainly for a conceptual
# reason: the trace, looptrace, and model enter the likelihood on equal
# footing, i.e. it is just as fair to talk about the likelihood of the model
# given the trace as vice versa. Or likelihood of looptrace given trace and
# model, etc.
def _likelihood_filter(trace, model, looptrace, noise):
    """
    Likelihood calculation using Kalman filter.
    """
    T = len(trace)
    try:
        w = model.measurement
    except AttributeError:
        w = np.zeros((model.N,))
        w[ 0] =  1
        w[-1] = -1

    model.check_setup_called()
    dt = model._propagation_memo['dt']

    M0, C0 = model.steady_state(looptrace[0])

    logL = np.empty((T,), dtype=float)
    logL[:] = np.nan
    for i, bond in enumerate(looptrace):
        M1, C1 = model.propagate(M0, C0, dt, bond)
        if np.isnan(trace[i]):
            M0 = M1
            C0 = C1
            continue
        
        # Update step copied from Christoph
        if noise > 0:
            InvSigmaPrior = scipy.linalg.inv(C1)
            InvSigma = np.tensordot(w, w, 0)/noise**2 + InvSigmaPrior
            SigmaPosterior = scipy.linalg.inv(InvSigma)
            MuPosterior = SigmaPosterior @ (w*trace[i] / noise**2 + InvSigmaPrior @ M1)
        else:
            SigmaPosterior = 0*C1
            MuPosterior = scipy.linalg.inv(np.tensordot(w, w, 0)) @ w*trace[i]
        
        # Same for likelihood calculation
        m = w @ M1
        s = C1[0, 0] + C1[-1, -1] - 2*C1[0, -1]
        if s < 0:
            raise RuntimeError("Prediction covariance negative: {}\nModel: {}".format(s, model))
#         logL[i] = np.log(scipy.stats.norm.pdf(trace[i], m, np.sqrt(s + noise**2)))
        logL[i] = -0.5*(trace[i] - m)**2 / (s+noise**2) - 0.5*np.log(2*np.pi*(s+noise**2))
        
        M0 = MuPosterior
        C0 = SigmaPosterior
        
    return np.nansum(logL)

def _likelihood_direct(trace, model, looptrace, noise):
    """
    Likelihood calculation using the full Rouse PDF.
    """
    T = len(trace)
    try:
        w = model.measurement
    except AttributeError:
        w = np.zeros((model.N,))
        w[ 0] =  1
        w[-1] = -1

    model.check_setup_called()

    # Get steady state to start from
    _, J = model.steady_state(looptrace[0])

    traceCov = np.zeros((len(trace), len(trace)))
    for i, bond in enumerate(looptrace):
        if i > 0:
            etA = model._propagation_memo[bond]['etA']
            Sig = model._propagation_memo[bond]['Sig']
            J = etA @ J @ etA.T + Sig

        AJw = J @ w
        traceCov[i, i] = w @ AJw
        for j in range(i+1, len(looptrace)):
            AJw = model._propagation_memo[looptrace[j]]['etA'] @ AJw
            traceCov[j, i] = w @ AJw

    traceCov = traceCov + traceCov.T - np.diagflat(np.diagonal(traceCov))

    # Add localization error
    traceCov += noise**2 * np.eye(len(trace))

    # Make this gap-robust
    ind = ~np.isnan(trace)
    traceCov = traceCov[ind, :][:, ind]
    trace = trace[ind]

    (detsign, logdet) = np.linalg.slogdet(traceCov)
    if detsign < 1:
        raise RuntimeError("Calculated covariance matrix has non-positive determinant")
    return -0.5 * trace @ scipy.linalg.inv(traceCov) @ trace - 0.5*(np.log(2*np.pi) + logdet)

likelihood = _likelihood_filter # faster
likelihood.__doc__ = """
    Calculate log likelihood for the given combination of
    trace, looping sequence, and model.

    Parameters
    ----------
    trace : (T,) array
        the observed trace.
    model : Model
        the model to use for likelihood calculation
    looptrace : (T,) array, dtype=bool
        for which steps there is a loop. ``looptrace[i]`` indicates that there
        is a loop during the evolution from ``trace[i-1]`` to ``trace[i]`` and
        ``looptrace[0]`` is the value used for initialization.
    noise : float
        the localization error on each point in the trace

    Returns
    -------
    logL : float
        the calculated log-likelihood

    See also
    --------
    Model

    Notes
    -----
    We assume that the ensemble starts from steady state.

    By default, we assume that the input trace is the relative coordinate of
    the two ends of the chain. To change this behavior, set
    `!model.measurement` to some `!model.N`-dimensional measurement vector. For
    the default behavior, this would be ``np.array([1, 0, ..., 0, -1])``.
    """
