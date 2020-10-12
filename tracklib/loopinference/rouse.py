import numpy as np
import scipy.linalg
import scipy.integrate

class Model:
    """
    Note: if k_extra < -k/(N-1) the connectivity matrix will have a negative
    eigenvalue. In this situation there is a repulsive bond between the ends
    that is stronger than the backbone, so it will tear the polymer apart. The
    model can still be useful in this limit, but if the extra bond is switched
    on too much, it will tear everything apart. This can also lead to negative
    covariance matrices.
    """
    def __init__(self, N, D, k, k_extra):
        """
        N : length of the chain
        D : diffusion constant of a free monomer
        k : spring constant of chain bonds
        k_extra : spring constant of the additional bond
        
        Relation to the physical Rouse model:
            γ*xdot = κ*B*x + ξ,   <ξ(t)xξ(t')> = 2*γ*k_B*T*δ(t-t')
        Then:
            k = κ/γ, D = k_B*T/γ
        """
        self.N = N
        self.D = D
        self.k = k
        self.k_extra = k_extra

    def __eq__(self, other):
        """
        Models are equal if all the parameters are equal.
        """
        for param in ['N', 'D', 'k', 'k_extra']:
            if getattr(self, param) != getattr(other, param):
                return False
        return True
        
    def __repr__(self):
        return "Rouse_model(N={}, D={}, k={}, k_extra={})".format(self.N, self.D, self.k, self.k_extra)
        
    def give_matrices(self, bond, tethered=False):
        """
        Assemble the matrices needed for evolution of the Rouse model
        
        Input
        -----
        bond : bool
            is the extra bond present?
        tethered : bool
            set to True to tether monomer 0 to the origin, i.e. get a non-singular A
            default: False
            
        Output
        ------
        A : the transition matrix for the deterministic part of the equation
        S : covariance of the noise
        """
        A = np.diagflat(self.N*[-2.]) + np.diagflat((self.N-1)*[1.], k=1) + np.diagflat((self.N-1)*[1.], k=-1)
        A[ 0,  0] += 1
        A[-1, -1] += 1
        A = self.k * A
        
        if bond:
            A[ 0,  0] -= self.k_extra
            A[ 0, -1] += self.k_extra
            A[-1,  0] += self.k_extra
            A[-1, -1] -= self.k_extra
            
        if tethered:
            A[0, 0] -= self.k
            
        S = 2*self.D*np.eye(self.N)
        
        return A, S
    
    def setup_propagation(self, dt=1):
        """
        Pre-calculate the matrices needed for propagation using the analytical
        solution
        """
        def integrand(tau, t, A, S):
            ettA = scipy.linalg.expm((t-tau)*A)
            return ettA @ S @ ettA.T

        self._propagation_memo = {
                'model' : Model(self.N, self.D, self.k, self.k_extra),
                'dt' : dt,
                }
        for bond in [True, False]:
            A, S = self.give_matrices(bond)
            Sig = scipy.integrate.quad_vec(lambda tau : integrand(tau, dt, A, S), 0, dt)[0]
            etA = scipy.linalg.expm(dt*A)
            self._propagation_memo[bond] = {'etA' : etA, 'Sig' : Sig}

    def check_propagation_memo_uptodate(self, dt=1):
        """
        Check whether the internal storage variable _propagation_memo exists
        and is up to date with the current parameters.
        """
        if not hasattr(self, '_propagation_memo'):
            raise RuntimeError("Call setup_propagation before using propagate()")
        elif not self == self._propagation_memo['model'] or dt != self._propagation_memo['dt']:
            raise RuntimeError("Parameter values changed since last call to setup_propagation()")
    
    def _propagate_ode(self, M0, C0, t, bond):
        """
        Propagate the ensemble specified by mean M0 and covariance C0
        for a time t and return the new mean and covariance

        Note: sometimes this leads to C1 not being positive definite!
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
        Propagate M0 and C0 using the analytical solution to the equations used
        by _propagate_ode.

        Note: call setup_propagation before using this. Failing to do so will
        raise a RuntimeError.
        Speedup over _propagate_ode is roughly 8x.
        """
        self.check_propagation_memo_uptodate(t)
        etA = self._propagation_memo[bond]['etA']
        Sig = self._propagation_memo[bond]['Sig']
        return etA @ M0, etA @ C0 @ etA.T + Sig
        
    propagate = _propagate_exp
    
# We define the likelihood outside the Model class mainly for a conceptual
# reason: the trace, looptrace, and model enter the likelihood on equal
# footing, i.e. it is just as fair to talk about the likelihood of the model
# given the trace as vice versa. Or likelihood of looptrace given trace and
# model, etc.
def _likelihood_filter(trace, looptrace, model, *, noise, w=None, times=None):
    """
    Calculate log likelihood for the given combination of
    trace, looping sequence, and model.

    Input
    -----
    trace : (T,) array
        the distance trace
    looptrace : (T,) array, dtype=bool
        for which steps there is a loop. looptrace[i] indicates that
        there is a loop during the evolution from trace[i-1] to trace[i]
        and looptrace[0] is the value used for initialization.
    model : Model
        the model to use for likelihood calculation
    noise : float
        the localization error on each trace
    w : (N,) array
        the measurement vector
        default: (1, 0, ..., 0, -1) (i.e. end-to-end distance)
    times : (T,) array
        the times at which the trace is evaluated
        default: np.arange(T)

    Output
    ------
    logL : float
        the calculated log-likelihood

    Notes
    -----
    We assume that the ensemble starts from steady state.
    This is filterData in Christoph's / Hugo's code
    """
    T = len(trace)
    if w is None:
        w = np.zeros((model.N,))
        w[ 0] =  1
        w[-1] = -1
    if times is None:
        times = np.arange(T)

    A, S = model.give_matrices(bond=looptrace[0], tethered=True)
    M0 = np.zeros((model.N,))
    C0 = -0.5 * scipy.linalg.inv(A) @ S

    logL = np.empty((T,), dtype=float)
    logL[:] = np.nan
    curtime = 2*times[0] - times[1] # we have steady state "before" the trace starts
    for i, nexttime in enumerate(times):
        M1, C1 = model.propagate(M0, C0, nexttime-curtime, bond=looptrace[i])
        
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
        curtime = nexttime
        
    return np.sum(logL)

def _likelihood_direct(trace, looptrace, model, *, noise, w=None, times=None):
    """
    Calculate the likelihood for a trajectory using the Rouse solutions
    TODO: explain this better
    """
    T = len(trace)
    if w is None:
        w = np.zeros((model.N,))
        w[ 0] =  1
        w[-1] = -1
    if times is not None:
        raise ValueError("Direct likelihood calculation does not work with custom times")
    model.check_propagation_memo_uptodate()

    # Get steady state to start from
    A, S = model.give_matrices(bond=looptrace[0], tethered=True)
    J = -0.5 * scipy.linalg.inv(A) @ S

    traceCov = np.zeros((len(trace), len(trace)))
    for i in range(len(trace)):
        if i > 0:
            etA = model._propagation_memo[looptrace[i]]['etA']
            Sig = model._propagation_memo[looptrace[i]]['Sig']
            J = etA @ J @ etA.T + Sig

        AJw = J @ w
        traceCov[i, i] = w @ AJw
        for j in range(i+1, len(trace)):
            AJw = model._propagation_memo[looptrace[j]]['etA'] @ AJw
            traceCov[j, i] = w @ AJw

    traceCov = traceCov + traceCov.T - np.diagflat(np.diagonal(traceCov))

    # Add localization error
    traceCov += noise**2 * np.eye(len(trace))

    (detsign, logdet) = np.linalg.slogdet(traceCov)
    if detsign < 1:
        raise RuntimeError("Calculated covariance matrix has negative determinant")
    return -0.5 * trace @ scipy.linalg.inv(traceCov) @ trace - 0.5*(np.log(2*np.pi) + logdet)

likelihood = _likelihood_filter # faster
