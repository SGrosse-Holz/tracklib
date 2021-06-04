"""
This module provides an implementation of the Rouse model.

In a decision to favor time over space, we use simple numpy arrays instead of
sparse matrices (which might make sense for A).

The check_dynamics mechanism is supposed to simply catch the most
straight-forward mistakes, i.e. it checks the parameter values and keeps track
of changes to the matrices through the member functions, but nothing else.

This implementation uses S = 1 and AS = SA^T, i.e. we also assume symmetric A.

D can be zero
"""
# Implementation Note: this is a synergy of tracklib's old rouse.py and
# rouselib's rousesim.py

import numpy as np
import scipy.linalg
import scipy.integrate

LOG_SQRT_2_PI = 0.5*np.log(2*np.pi)

class Model:
    """
    The Rouse model

    ...
    """
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
        rep = "rouse.Model(N={}, D={}, k={})".format(self.N, self.D, self.k)
        if n_extra_bonds > 0:
            rep += " with {} additional bonds".format(n_extra_bonds)
        return rep

####### Setting up a model and its dynamics

    def setup_free_chain(self, d=3):
        self._dynamics['needs_updating'] = True

        self.A = np.diagflat(self.N*[2.], k=0) \
                 + np.diagflat((self.N-1)*[-1.], k=-1) \
                 + np.diagflat((self.N-1)*[-1.], k= 1)
        self.A[0, 0] = self.A[-1, -1] = 1
        
        self.F = np.zeros((self.N, d))

    def add_crosslinks(self, links, k_rel=1.):
        """
        Provide bonds as (loc0, loc1, k_rel) or (loc0, loc1). The latter case
        then uses k_rel.
        """
        self._dynamics['needs_updating'] = True

        for link in links:
            myk_rel = k_rel if len(link) == 2 else link[2]
            self.A[link[0], link[0]] += myk_rel
            self.A[link[0], link[1]] -= myk_rel
            self.A[link[1], link[0]] -= myk_rel
            self.A[link[1], link[1]] += myk_rel

    def add_tether(self, mon=0, k_rel=1., point=(0., 0., 0.)):
        self._dynamics['needs_updating'] = True
        
        self.A[mon, mon] += k_rel
        self.F[mon] += k_rel * point

    def update_dynamics(self, dt=1.):
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
        if np.isclose(scipy.linalg.det(self.A), 0):
            self._dynamics['invA'] = None
        else:
            self._dynamics['invA'] = scipy.linalg.inv(self.A)

    def update_G(self, override_full_update=False):
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

    def steady_state(self, additional_tether=True):
        if additional_tether:
            self.A[0, 0] += float(additional_tether)
            invA = scipy.linalg.inv(self.A)
            self.A[0, 0] -= float(additional_tether)
        else:
            if not 'invA' in self._dynamics.keys():
                self.update_invA()
            invA = self._dynamics['invA']
            if invA is None:
                raise RuntimeError("Free chain does not have a steady state")

        return (invA @ self.F / self.k,
                invA * self.D / self.k)

    def propagate(self, M, C, dt=None):
        self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        G = self._dynamics['G']
        Sig = self._dynamics['Sig']
        return B @ M + G, B @ C @ B + Sig

    def propagate_M(self, M, dt=None, check_dynamics=True):
        if check_dynamics:
            self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        G = self._dynamics['G']
        return B @ M + G

    def propagate_C(self, C, dt=None, check_dynamics=True):
        if check_dynamics:
            self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        Sig = self._dynamics['Sig']
        return B @ C @ B + Sig

####### Evolution of a single conformation

    def conf_ss(self, additional_tether=True):
        M, C = self.steady_state(additional_tether)
        L = scipy.linalg.cholesky(C, lower=True)
        return M + L @ np.random.normal(size=(self.N, self.d))

    def evolve(self, conf, dt=None):
        self.check_dynamics(dt)
        B = self._dynamics['emkAt']
        L = self._dynamics['LSig']
        return B @ conf + L @ np.random.normal(size=conf.shape)

####### Likelihood evaluation

    @staticmethod # This actually doesn't have anything to do with the model at hand
    def update_ensemble_with_observation_1d(M, C, x, s, w):
        # Important: M is (N,) only, no d!
        s2 = s*s

        invC = scipy.linalg.inv(C)
        invCpost = invC + w[:, np.newaxis]*w[np.newaxis, :] / s2
        Cpost = scipy.linalg.inv(invCpost)
        Mpost = Cpost @ (invC @ M + w*x / s2)

        m = w @ M
        c = w @ C @ w
        cs = c+s2
        logL = -0.5*((x-m)*(x-m)/cs + np.log(cs)) - LOG_SQRT_2_PI

        return Mpost, Cpost, logL

    @staticmethod
    def update_ensemble_with_observation_nd(M, C, x, s, w):
        # now: M is (N, d), C is d*[(N, N)], x is (d,), s is (d,), w is (N,)
        # We take advantage of the fact that the update on C depends only on
        # the localization error, i.e. if some errors are the same, we can skip
        # some computation:)
        uni_s, uni_ind, rev_ind = np.unique(s, return_index=True, return_inverse=True)
        uni_s2 = uni_s*uni_s

        assert all([np.all(C[d] == C[i]) for d, i in enumerate(rev_ind)])

        uni_invC = [scipy.linalg.inv(C[i]) for i in uni_ind]
        uni_Cpost = [scipy.linalg.inv(invC + w[:, np.newaxis]*w[np.newaxis, :] / s2) for invC, s2 in zip(uni_invC, uni_s2)]

        Mpost = np.stack([uni_Cpost[i] @ (uni_invC[i] @ M[:, d] + w*x[d] / uni_s2[i]) \
                          for d, i in enumerate(rev_ind)], axis=1)
        Cpost = [uni_Cpost[i] for i in rev_ind]

        m = w @ M
        cs = np.array([w @ myC @ w + s*s for myC, s in zip(C, s)])
        logL = -0.5*np.sum((x-m)*(x-m)/cs + np.log(cs)) - len(cs)*LOG_SQRT_2_PI

        return Mpost, Cpost, logL

####### Auxiliary things

    def contact_probability(self, additional_tether=True):
        _, J = self.steady_state(additional_tether)
        Jii = np.tile(np.diagonal(J), (len(J), 1))
        return (Jii + Jii.T - 2*J)**(-3/2)

    def analytical_MSD(self, dts, w=None):
        try:
            self.check_dynamics(run_if_necessary=False)
        except:
            self.update_invA()
        invA = self._dynamics['invA']

        dts = np.sort(dts)

        Bs = [scipy.linalg.expm(-self.k*self.A*dt) for dt in dts]

        if invA is not None:
            Sigs = [(np.eye(self.N) - B @ B) @ invA * self.D/self.k for B in Bs]
        else:
            def integrand(tau):
                return scipy.linalg.expm(-2*self.k*self.A*tau)
            res, err, full_info = scipy.integrate.quad_vec(
                    integrand, 0, dts[-1],
                    points = dts,
                    full_output = True)
            Sigs = [2*self.D * np.sum(
                                full_info.integrals[full_info.intervals[:, 1] <= dt],
                                axis = 0)
                    for dt in dts]

        xxs = [2*scipy.linalg.inv(np.eye(self.N) + B) @ Sig for B, Sig in zip(Bs, Sigs)]

        if w is None:
            return self.d * np.array(xxs)
        else:
            return self.d * np.array([w @ xx @ w for xx in xxs])

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

    def rms_Ree(self, N=None):
        if N is None:
            N = self.N
        return self.d*self.D/self.k * N
