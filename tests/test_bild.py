import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
np.seterr(all='raise')
from matplotlib import pyplot as plt
import scipy.stats

import unittest
from unittest.mock import patch

from context import tracklib as tl
bild = tl.analysis.bild

# Extend unittest.TestCase's capabilities to deal with numpy arrays
class myTestCase(unittest.TestCase):
    def assert_array_equal(self, array1, array2):
        try:
            np.testing.assert_array_equal(array1, array2)
            res = True
        except AssertionError as err: # pragma: no cover
            res = False
            print(err)
        self.assertTrue(res)

class TestUtilLoopingtrace(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.arange(9).reshape(3, 3))
        self.lt = bild.Loopingtrace.forTrajectory(self.traj, thresholds=[7, 9])

    def test_forTrajectory(self):
        lt = bild.Loopingtrace.forTrajectory(self.traj, nStates=5, thresholds=[7, 9])
        self.assertEqual(lt.n, 3)
        self.assert_array_equal(lt.state, [0, 1, 2])
        self.assertEqual(lt.state.dtype, int)
        self.assertEqual(lt.t.dtype, int)

        lt = bild.Loopingtrace.forTrajectory(self.traj)
        self.assertEqual(lt.state.dtype, int)

    def test_fromStates(self):
        lt = bild.Loopingtrace.fromStates([1, 2, 1, np.nan, 0], nStates=3)
        self.assertEqual(lt.state.dtype, int)
        self.assertEqual(len(lt.state), 4)
        self.assert_array_equal(lt.t, np.array([0, 1, 2, 4]))

    def test_copy(self):
        new = self.lt.copy()
        self.lt.state[2] = 5
        self.lt.t[1] = 10
        self.assertEqual(new.state[2], 2)
        self.assertEqual(new.t[1], 1)

    def test_sequence(self):
        self.assertEqual(len(self.lt), 3)
        self.assertEqual(self.lt[2], 2)
        self.lt[2] = 0
        self.assertEqual(self.lt[2], 0)

        self.lt[:] = [0, 0, 0]
        self.assert_array_equal(self.lt[:], np.array([0, 0, 0]))

    def test_equality(self):
        lt0 = bild.Loopingtrace.fromStates([1, 1, 0, 0, 1, np.nan, 0])

        lt1 = bild.Loopingtrace.fromStates([1, 1, 0, 0, 1, np.nan, 0])
        self.assertTrue(lt0 == lt1)
        self.assertEqual(lt0, lt1)

        lt1 = bild.Loopingtrace.fromStates([1, 2, 0, 0, 1, np.nan, 0])
        self.assertFalse(lt0 == lt1)
        self.assertNotEqual(lt0, lt1)

        lt1 = bild.Loopingtrace.fromStates([1, 1, 0, 0, 1, 0, 0])
        self.assertFalse(lt0 == lt1)

        lt1 = bild.Loopingtrace.fromStates([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        self.assertFalse(lt0 == lt1)


    def test_plottable(self):
        plt.plot(*self.lt.plottable(), color='r')
        plt.show()

    def test_full_valid(self):
        traj = tl.Trajectory.fromArray([1, 2, np.nan, 4])
        lt = bild.Loopingtrace.forTrajectory(traj, thresholds=[3])
        self.assert_array_equal(lt.full_valid(), np.array([0, 0, 1, 1]))

    def test_loops(self):
        lt = bild.Loopingtrace.fromStates([1, 1, 1, 0, np.nan, 0, 0, 1, 1])
        self.assert_array_equal(lt.loops(), np.array([[0, 3, 1], [3, 7, 0], [7, 9, 1]]))
        self.assert_array_equal(lt.loops(return_='index'), np.array([[0, 3, 1], [3, 6, 0], [6, 8, 1]]))

# class TestParametricFamily(myTestCase):
# This is literally just an initializer, nothing to test here

class TestPriors(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]))
        self.lt = bild.Loopingtrace.forTrajectory(self.traj, thresholds=[3])

    def test_uniform(self):
        prior = bild.priors.UniformPrior()
        log_pmf = -3*np.log(2)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

    def test_geometric(self):
        prior = bild.priors.GeometricPrior(logq=-1, nStates=2)
        log_pmf = -1 - 2*np.log(1+np.exp(-1)) - np.log(2)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

        fam = bild.priors.GeometricPrior.family(nStates=3)
        prior = fam.get(-1)
        log_pmf = -1 - 2*np.log(1+np.exp(-1)*2) - np.log(3)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

class TestModels(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.lt = bild.Loopingtrace.forTrajectory(self.traj, thresholds=[3])

    def test_base(self):
        model = bild.models.MultiStateRouse(20, 1, 5, d=1)
        traj = tl.Trajectory.fromArray([1, 2, 3])

        # Check base implementation
        lt = bild.models.MultiStateModel.initial_loopingtrace(model, traj)
        self.assertEqual(len(lt), 3)

    def test_Rouse(self):
        model = bild.models.MultiStateRouse(20, 1, 5, d=1)
        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([1, 0, 0]))

        traj = model.trajectory_from_loopingtrace(bild.Loopingtrace.fromStates([0, 0, 0, 1, 1, 1]),
                                                  localization_error=0.1)
        self.assertEqual(len(traj), 6)

    def test_Factorized(self):
        model = bild.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ], d=1)

        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([0, 0, 1]))

        model.clear_memo()
        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([0, 0, 1]))

        traj = model.trajectory_from_loopingtrace(bild.Loopingtrace.fromStates([0, 0, 0, 1, 1, 1]))
        self.assertEqual(len(traj), 6)

    def test_fitting_factorized(self):
        modelfam = bild.ParametricFamily((0.1, 10), [(1e-10, None), (1e-10, None)])
        modelfam.get = lambda s0, s1 : bild.models.FactorizedModel([
            scipy.stats.maxwell(scale=s0),
            scipy.stats.maxwell(scale=s1),
            ], d=1)

        true_params = (1, 4)
        data = tl.TaggedSet()
        mod_gen = modelfam.get(*true_params)
        for state in 3*[0, 1]:
            lt = bild.Loopingtrace.fromStates(100*[state])
            data.add(mod_gen.trajectory_from_loopingtrace(lt))

        fitres = bild.models.fit(data, modelfam, maxfun=200)
        self.assertTrue(fitres.success)
        for true_param, est_param in zip(true_params, fitres.x):
            self.assertAlmostEqual(true_param, est_param, delta=0.3)

    def test_fitting_rouse(self):
        modelfam = bild.ParametricFamily((0.1,), [(1e-10, None)])
        modelfam.get = lambda k : bild.models.MultiStateRouse(20, 1, k, d=3, localization_error=0.1)

        true_params = (1,)
        data = tl.TaggedSet()
        mod_gen = modelfam.get(*true_params)
        for state in 3*[0, 1]:
            lt = bild.Loopingtrace.fromStates(100*[state])
            data.add(mod_gen.trajectory_from_loopingtrace(lt))

        fitres = bild.models.fit(data, modelfam, maxfun=200)
        self.assertTrue(fitres.success)
        for true_param, est_param in zip(true_params, fitres.x):
            self.assertAlmostEqual(true_param, est_param, delta=0.3)

class TestGeometricPriorScheme(myTestCase):
    def setUp(self):
        self.scheme = bild.mcmc.GeometricPriorScheme(nStates=3, stepsize=0.5)

    def test_all(self):
        self.assertEqual(self.scheme.nStates, 3)
        prior = self.scheme(-1)
        self.assertIsInstance(prior, bild.priors.GeometricPrior)
        self.assertAlmostEqual(self.scheme.stepping_probability(-1, -1), 0.5413, delta=1e-3)
        self.assertGreater(0, next(self.scheme.gen_proposal_sample_from()))
        self.assertAlmostEqual(np.mean(list(self.scheme.gen_proposal_sample_from(-1, nSample=100))), -1.5, delta=0.3)

class TestFullMCMC(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.model = bild.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ], d=1)
        self.priorscheme = bild.mcmc.GeometricPriorScheme(nStates=2)

        self.sampler = bild.mcmc.FullMCMC(initialize_from='random')
        self.sampler.setup(self.traj, self.model, self.priorscheme)

    def test_configure(self):
        self.sampler.configure(min_approaches_to_best_sample=23)
        self.assertEqual(self.sampler.config['min_approaches_to_best_sample'], 23)
        self.assertNotEqual(self.sampler.config['check_stopping_every'], -1)

    def test_stopping(self):
        self.sampler.configure(min_approaches_to_best_sample=5)
        run_all_different = tl.util.mcmc.MCMCRun(np.linspace(0, 5, 10), [([], -1) for _ in range(10)])
        run_all_same = tl.util.mcmc.MCMCRun(np.linspace(0, 5, 10), 10*[([], -1)])

        self.assertTrue(self.sampler.callback_stopping(run_all_different))
        self.assertFalse(self.sampler.callback_stopping(run_all_same))

    def test_lt_stepping_probability_and_proposal_sample(self):
        def num2binlist(num, nbits):
            ls = list(bin(num + (1<<nbits)))[3:] # bin() gives '0b1...'
            return [int(oi) for oi in ls]

        nbit = 6
        lts = [bild.Loopingtrace.fromStates(num2binlist(num, nbit), nStates=2) for num in range(1<<nbit)]

        for lt_from_ind in np.random.choice(len(lts), 5, replace=False):
            self.assertAlmostEqual(np.sum([self.sampler.lt_stepping_probability(lts[lt_from_ind], lt) for lt in lts]), 1, delta=1e-10)

        lt_prop = lts[np.random.choice(len(lts))]
        p_next = np.array([self.sampler.lt_stepping_probability(lt_prop, lt) for lt in lts])
        ind_best_next = np.argmax(p_next)
        lt_best_next = lts[ind_best_next]
        N = 100/p_next[ind_best_next]
        if N > 10000: # pragma: no cover
            raise RuntimeError(f"Don't want to sample {N} profiles")
        N_best_next = np.count_nonzero([lt == lt_best_next for lt in self.sampler.lt_gen_proposal_sample_from(lt_prop, nSample=int(N))])
        self.assertAlmostEqual(N_best_next, 100, delta=30) # 3 sigma

    def test_run(self):
        self.sampler.configure(iterations=10, burn_in=5)
        run = self.sampler.run()
        self.assertEqual(len(run.samples), 5)

        self.sampler.initialize_from = 'random'
        run = self.sampler.run()
        self.assertEqual(len(run.samples), 5)

class Test_main(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])

        self.model = bild.models.MultiStateRouse(N=20, D=1, k=5, d=1)
        self.MCMCconfig = {'iterations' : 10, 'burn_in' : 0}

    def test_main(self):
        bild.main(self.traj, self.model, self.MCMCconfig)
        self.assertIn('bild', self.traj.meta.keys())

        ret_traj = bild.main(self.traj, self.model, self.MCMCconfig, return_='traj')
        self.assertIs(ret_traj, self.traj)

        ret_dict = bild.main(self.traj, self.model, self.MCMCconfig, return_='dict')
        self.assertEqual(ret_dict.keys(), self.traj.meta['bild'].keys()) # cheat, we have the dict
                                                                         # from previous runs

        bild.main(self.traj, self.model, self.MCMCconfig, return_='None', return_mcmcrun=True)
        self.assertIn('mcmcrun', self.traj.meta['bild'].keys())

# class Test_plot(myTestCase):
#     @patch("builtins.print")
#     def setUp(self, mock_print):
#         self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
# 
#         model = bild.models.RouseModel(N=20, D=1, k=5)
#         priorfam = bild.priors.GeometricPrior.family()
#         MCMCconfig = {'iterations' : 10, 'burn_in' : 5}
# 
#         np.random.seed(119) # chosen such that at least the first run doesn't collapse
#         bild.main(self.traj, model, priorfam, MCMCconfig, max_iterations=7)
# 
#     def test_butterfly(self):
#         bild.plot.butterfly(self.traj)
#         plt.show()
# 
# class Test_postproc(myTestCase):
#     def setUp(self):
#         self.traj = tl.Trajectory.fromArray(np.array([1, 1, 1, 5, 5, 5, np.nan, 7, 5, 4, 5, 1, 1]), localization_error=[0.1])
#         self.model = bild.models.RouseModel(N=20, D=1, k=5, k_extra=1, looppositions=[(0, 0), (0, -1)])
#         self.prior = bild.priors.GeometricPrior(logq=0, nStates=2)
# 
#         self.lt = self.model.initial_loopingtrace(self.traj)
#         # this is [1 1 1 0 0 0 0 0 0 0 1 1], which is also the optimal one (as
#         # determined by exhaustive sampling)
# 
#     def test_clusterflip(self):
#         lt_cf = self.lt.copy()
#         lt_cf[1] = 0
#         lt_cf[4:7] = 1
#         lt_cf[8] = 1
#         lt_opt = bild.postproc.optimize_clusterflip(lt_cf, self.traj, self.model, self.prior)
#         self.assertEqual(lt_opt, self.lt)
# 
#         lt_cf[2] = 0 # Now the boundaries are messed up
#         lt_opt = bild.postproc.optimize_clusterflip(lt_cf, self.traj, self.model, self.prior)
#         self.assertNotEqual(lt_opt, self.lt)
# 
#     def test_nbit(self):
#         lt_n = self.lt.copy()
#         lt_n[2] = 0
#         lt_n[4:6] = 1
#         lt_n[9:11] = [1, 0]
#         lt_opt = bild.postproc.optimize_nbit(lt_n, self.traj, self.model, self.prior, n=3)
#         self.assertEqual(lt_opt, self.lt)
# 
#     def test_boundary(self):
#         lt_b = self.lt.copy()
#         lt_b[3] = 1
#         lt_b[10] = 0
#         lt_opt = bild.postproc.optimize_boundary(lt_b, self.traj, self.model, self.prior)
#         self.assertEqual(lt_opt, self.lt)
# 
#         lt_b[0] = 0
#         lt_opt = bild.postproc.optimize_boundary(lt_b, self.traj, self.model, self.prior)
#         self.assertEqual(lt_opt, self.lt)
# 
#         # Boundary moves can abolish an interval
#         lt_b[6] = 1
#         lt_opt = bild.postproc.optimize_boundary(lt_b, self.traj, self.model, self.prior)
#         self.assertEqual(lt_opt, self.lt)
# 
#         # But not create new boundaries
#         lt_b[:5] = 1
#         lt_b[5:] = 0
#         lt_opt = bild.postproc.optimize_boundary(lt_b, self.traj, self.model, self.prior)
#         self.assertNotEqual(lt_opt, self.lt)
# 
#         lt_b[:] = 0
#         lt_opt = bild.postproc.optimize_boundary(lt_b, self.traj, self.model, self.prior)
#         self.assertNotEqual(lt_opt, self.lt)

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
