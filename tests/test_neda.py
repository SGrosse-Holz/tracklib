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
neda = tl.analysis.neda

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
        self.lt = neda.Loopingtrace.forTrajectory(self.traj, thresholds=[7, 9])

    def test_forTrajectory(self):
        lt = neda.Loopingtrace.forTrajectory(self.traj, nStates=5, thresholds=[7, 9])
        self.assertEqual(lt.n, 3)
        self.assert_array_equal(lt.state, [0, 1, 2])
        self.assertEqual(lt.state.dtype, int)
        self.assertEqual(lt.t.dtype, int)

        lt = neda.Loopingtrace.forTrajectory(self.traj)
        self.assertEqual(lt.state.dtype, int)

    def test_fromStates(self):
        lt = neda.Loopingtrace.fromStates([1, 2, 1, np.nan, 0], nStates=3)
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

    def test_plottable(self):
        plt.plot(*self.lt.plottable(), color='r')
        plt.show()

    def test_full_valid(self):
        traj = tl.Trajectory.fromArray([1, 2, np.nan, 4])
        lt = neda.Loopingtrace.forTrajectory(traj, thresholds=[3])
        self.assert_array_equal(lt.full_valid(), np.array([0, 0, 1, 1]))

    def test_loops(self):
        lt = neda.Loopingtrace.fromStates([1, 1, 1, 0, 0, 0, 0, 1, 1])
        self.assert_array_equal(lt.loops(), np.array([[0, 3, 1], [3, 7, 0], [7, 9, 1]]))

# class TestParametricFamily(myTestCase):
# This is literally just an initializer, nothing to test here

class TestPriors(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]))
        self.lt = neda.Loopingtrace.forTrajectory(self.traj, thresholds=[3])

    def test_uniform(self):
        prior = neda.priors.UniformPrior()
        log_pmf = -3*np.log(2)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

    def test_geometric(self):
        prior = neda.priors.GeometricPrior(logq=-1, nStates=2)
        log_pmf = -1 - 2*np.log(1+np.exp(-1)) - np.log(2)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

        fam = neda.priors.GeometricPrior.family(nStates=3)
        prior = fam.get(-1)
        log_pmf = -1 - 2*np.log(1+np.exp(-1)*2) - np.log(3)
        self.assertEqual(prior.logpi(self.lt), log_pmf)
        self.assert_array_equal(prior.logpi_vectorized([self.lt]), np.array([log_pmf]))

class TestModels(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.lt = neda.Loopingtrace.forTrajectory(self.traj, thresholds=[3])

    def test_Rouse(self):
        model = neda.models.RouseModel(20, 1, 5, k_extra=1)
        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([1, 0, 0]))

        traj = model.trajectory_from_loopingtrace(neda.Loopingtrace.fromStates([0, 0, 0, 1, 1, 1]))
        self.assertEqual(len(traj), 6)

    def test_Factorized(self):
        model = neda.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ])

        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([0, 0, 1]))

        model.clear_memo()
        logL = model.logL(self.lt, self.traj)
        lt = model.initial_loopingtrace(self.traj)
        self.assertTrue(logL > -100 and logL < 0)
        self.assert_array_equal(lt.state, np.array([0, 0, 1]))

        traj = model.trajectory_from_loopingtrace(neda.Loopingtrace.fromStates([0, 0, 0, 1, 1, 1]))
        self.assertEqual(len(traj), 6)

    def test_fitting_factorized(self):
        modelfam = neda.ParametricFamily((0.1, 10), [(1e-10, None), (1e-10, None)])
        modelfam.get = lambda s0, s1 : neda.models.FactorizedModel([
            scipy.stats.maxwell(scale=s0),
            scipy.stats.maxwell(scale=s1),
            ])

        true_params = (1, 4)
        data = tl.TaggedSet()
        mod_gen = modelfam.get(*true_params)
        for state in 3*[0, 1]:
            lt = neda.Loopingtrace.fromStates(100*[state])
            data.add(mod_gen.trajectory_from_loopingtrace(lt))

        fitres = neda.models.fit(data, modelfam, maxfun=200)
        self.assertTrue(fitres.success)
        for true_param, est_param in zip(true_params, fitres.x):
            self.assertAlmostEqual(true_param, est_param, delta=0.5)

    def test_fitting_rouse(self):
        modelfam = neda.ParametricFamily((0.1,), [(1e-10, None)])
        modelfam.get = lambda k : neda.models.RouseModel(20, 1, k)

        true_params = (1,)
        data = tl.TaggedSet()
        mod_gen = modelfam.get(*true_params)
        for state in 3*[0, 1]:
            lt = neda.Loopingtrace.fromStates(100*[state])
            data.add(mod_gen.trajectory_from_loopingtrace(lt))

        fitres = neda.models.fit(data, modelfam, maxfun=200)
        self.assertTrue(fitres.success)
        for true_param, est_param in zip(true_params, fitres.x):
            self.assertAlmostEqual(true_param, est_param, delta=0.5)

class TestTriMCMC(myTestCase):
    def setUp(self):
        self.lt = neda.Loopingtrace.fromStates([1, 1, 1, 0, 0, 0, 0, 1, 1])

    def test_stepping_probability(self):
        def p_step(tri, states):
            return tri.stepping_probability(self.lt,
                                            neda.Loopingtrace.fromStates(states),
                                            )

        tri = neda.mcmc.TriMCMC(weights=[1, 1, 1])

        self.assertEqual(p_step(tri, self.lt.state), 0)
        self.assertEqual(p_step(tri, [1, 1, 0, 1, 1, 1, 0, 1, 1]), 0)

        p_last_interval = 0
        p_last_interval += p_step(tri, [1, 1, 1, 0, 0, 0, 0, 0, 0])
        p_last_interval += p_step(tri, [1, 1, 1, 0, 0, 0, 0, 0, 1])
        p_last_interval += p_step(tri, [1, 1, 1, 0, 0, 0, 0, 1, 0])
        self.assertAlmostEqual(p_last_interval, 1/3)

        tri = neda.mcmc.TriMCMC(weights=[1, 0, 0])
        self.assertAlmostEqual(p_step(tri, [1, 1, 1, 1, 1, 1, 1, 1, 1]), 1/3)

        tri = neda.mcmc.TriMCMC(weights=[0, 1, 0])
        self.assertEqual(p_step(tri, [1, 1, 1, 0, 1, 1, 0, 1, 1]), 0)

        p_mid_interval = 0
        p_mid_interval += p_step(tri, [1, 1, 1, 0, 0, 0, 1, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 0, 0, 1, 1, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 0, 1, 1, 1, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 1, 1, 1, 0, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 1, 1, 0, 0, 1, 1])
        p_mid_interval += p_step(tri, [1, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertAlmostEqual(p_mid_interval, 1/3)

        tri = neda.mcmc.TriMCMC(weights=[0, 0, 1])
        self.assertEqual(p_step(tri, [1, 0, 1, 0, 1, 1, 0, 1, 1]), 0)
        self.assertEqual(p_step(tri, [1, 1, 1, 0, 0, 0, 1, 1, 0]), 0)
        self.assertEqual(p_step(tri, [1, 1, 0, 0, 0, 0, 1, 1, 1]), 0)

    def test_proposal_sample(self):
        tri = neda.mcmc.TriMCMC(weights=[1, 1, 1])

        lt_from = neda.Loopingtrace.fromStates([1, 1, 1, 0, 0, 0, 0, 1, 1])
        for lt_to in tri.gen_proposal_sample_from(lt_from, nSample=10):
            self.assertGreater(tri.stepping_probability(lt_from, lt_to), 0)
            self.assertGreater(tri.stepping_probability(lt_to, lt_from), 0)

        lt_from = neda.Loopingtrace.fromStates([1, 0, 1, 0, 0, 0, 1, 1, 1])
        for lt_to in tri.gen_proposal_sample_from(lt_from, nSample=10):
            self.assertGreater(tri.stepping_probability(lt_from, lt_to), 0)
            self.assertGreater(tri.stepping_probability(lt_to, lt_from), 0)

        lt_from = neda.Loopingtrace.fromStates([1, 0, 1, 1, 0, 1, 1, 0, 1])
        for lt_to in tri.gen_proposal_sample_from(lt_from, nSample=10):
            self.assertGreater(tri.stepping_probability(lt_from, lt_to), 0)
            self.assertGreater(tri.stepping_probability(lt_to, lt_from), 0)

        lt_from = neda.Loopingtrace.fromStates([1, 0, 1, 1, 0])
        for lt_to in tri.gen_proposal_sample_from(lt_from):
            self.assertGreater(tri.stepping_probability(lt_from, lt_to), 0)
            self.assertGreater(tri.stepping_probability(lt_to, lt_from), 0)

class TestMCMCRun(myTestCase):
    def setUp(self):
        sample0 = [0]
        sample1 = [1]
        sample2 = [2]

        self.samples = 5*[sample0] + 2*[sample1] + 3*[sample2]
        self.logLs = np.array(10*[-10] + 5*[-5] + 2*[-2] + 3*[-3])
        self.myrun = neda.mcmc.MCMCRun(self.logLs, self.samples)

    def test_logLs_trunc(self):
        self.assert_array_equal(self.myrun.logLs_trunc(), self.logLs[10:20])

    def test_best_sample_L(self):
        best, bestL = self.myrun.best_sample_L()
        self.assertEqual(bestL, -2)
        self.assertListEqual(best, [1])

    def test_acceptance_rate(self):
        acc = 2/9
        self.assertEqual(self.myrun.acceptance_rate('sample_equality'), acc)
        self.assertEqual(self.myrun.acceptance_rate('sample_identity'), acc)
        self.assertEqual(self.myrun.acceptance_rate('likelihood_equality'), acc)

    def test_evaluate(self):
        tmp = self.myrun.evaluate(lambda sample : 2*sample)
        self.assertIs(tmp[0], tmp[1])
        self.assertTupleEqual(np.array(tmp).shape, (10, 2))

class TestMCMCSchemes(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.model = neda.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ])
        self.prior = neda.priors.GeometricPrior(-1, nStates=2)

        self.tpw = neda.mcmc.TPWMCMC()
        self.tpw.setup(self.traj, self.model, self.prior)
        self.tpw.configure(iterations=10, burn_in=5)

    def test_acceptance_probability(self):
        self.assertEqual(self.tpw.acceptance_probability(1, 2), 1)
        self.assertEqual(self.tpw.acceptance_probability(1, 0), np.exp(-1))

    def test_run(self):
        myrun = self.tpw.run()
        self.assertEqual(len(myrun.samples), 5)
        self.assertEqual(len(myrun.logLs), 10)

        neda.mcmc.TPWMCMC.propose_update = neda.mcmc.MCMCScheme.propose_update
        tpw = neda.mcmc.TPWMCMC()
        tpw.setup(self.traj, self.model, self.prior)
        tpw.configure(iterations=10, burn_in=5)

        myrun = tpw.run()
        self.assertEqual(len(myrun.samples), 5)
        self.assertEqual(len(myrun.logLs), 10)

    def test_stepping_probability(self):
        myrun = self.tpw.run()
        for i in range(1, len(myrun.samples)):
            self.assertGreaterEqual(self.tpw.stepping_probability(myrun.samples[i-1], myrun.samples[i]), 0)

    def test_gen_proposal_sample(self):
        myrun = self.tpw.run()
        lt = myrun.samples[0]

        for newlt in self.tpw.gen_proposal_sample_from(lt, nSample=1):
            self.assertEqual(np.count_nonzero(newlt.state != lt.state), 1)
        for newlt in self.tpw.gen_proposal_sample_from(lt, nSample=5):
            self.assertEqual(np.count_nonzero(newlt.state != lt.state), 1)

        cnt = 0
        for newlt in self.tpw.gen_proposal_sample_from(lt):
            cnt += 1
            self.assertEqual(np.count_nonzero(newlt.state != lt.state), 1)
        self.assertEqual(cnt, 3)

class test_Environment(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.model = neda.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ])
        self.MCMCconfig = {
                'iterations' : 10,
                'burn_in'    :  5,
                }

        self.env = neda.Environment(self.traj, self.model, self.MCMCconfig)
        self.prior = neda.priors.GeometricPrior(-1, nStates=2)

        # random seeds are set such that the first run does not collapse, the
        # second does
        np.random.seed(10)
        self.mcmcrun = self.env.runMCMC(self.prior)
        np.random.seed(15)
        self.mcmcrun_collapsed = self.env.runMCMC(self.prior)

    def test_runMCMC(self):
        self.assertEqual(len(self.mcmcrun.logLs), 10)
        self.assertEqual(len(self.mcmcrun.samples), 5)
        self.assertEqual(len(self.mcmcrun_collapsed.logLs), 10)
        self.assertEqual(len(self.mcmcrun_collapsed.samples), 5)

    def test_normal(self):
        p = self.env.posterior_density(self.mcmcrun, self.prior, self.mcmcrun.samples[0])
        self.assertGreater(p, 0)
        self.assertLess(p, np.inf)

        ev = self.env.evidence(self.prior, self.mcmcrun)
        self.assertLess(ev, np.max(self.mcmcrun.logLs))

    @patch("builtins.print")
    def test_collapsed(self, mock_print):
        p = self.env.posterior_density(self.mcmcrun_collapsed, self.prior, self.mcmcrun.samples[0])
        self.assertTrue(np.isinf(p))
        # mock_print.assert_called() # this is not important and might be changed

        ev = self.env.evidence(self.prior, self.mcmcrun_collapsed)
        self.assertEqual(ev, np.max(self.mcmcrun.logLs))
        # mock_print.assert_called() # this is not important and might be changed

    def test_evidence_diff(self):
        dev = self.env.evidence_differential(self.prior, self.prior, self.mcmcrun)
        self.assertEqual(dev, 0)

        dev = self.env.evidence_differential(neda.priors.GeometricPrior(-2), self.prior, self.mcmcrun)
        self.assertNotEqual(dev, 0)

class Test_main(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])

        self.model = neda.models.RouseModel(N=20, D=1, k=5)
        self.priorfam = neda.priors.GeometricPrior.family()
        self.MCMCconfig = {'iterations' : 10, 'burn_in' : 5}

    @patch("builtins.print")
    def test_main(self, mock_print):
        np.random.seed(119) # chosen such that at least the first run doesn't collapse
        neda.main(self.traj, self.model, self.priorfam, self.MCMCconfig, max_iterations=7)
        self.assertIn('neda', self.traj.meta.keys())

        np.random.seed(119)
        ret_traj = neda.main(self.traj, self.model, self.priorfam, self.MCMCconfig,
                             max_iterations=7, return_='traj')
        self.assertIs(ret_traj, self.traj)

        np.random.seed(119)
        ret_dict = neda.main(self.traj, self.model, self.priorfam, self.MCMCconfig,
                             max_iterations=7, return_='dict')
        self.assertEqual(ret_dict.keys(), self.traj.meta['neda'].keys()) # cheat, we have the dict
                                                                         # from previous runs

class Test_plot(myTestCase):
    @patch("builtins.print")
    def setUp(self, mock_print):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])

        model = neda.models.RouseModel(N=20, D=1, k=5)
        priorfam = neda.priors.GeometricPrior.family()
        MCMCconfig = {'iterations' : 10, 'burn_in' : 5}

        np.random.seed(119) # chosen such that at least the first run doesn't collapse
        neda.main(self.traj, model, priorfam, MCMCconfig, max_iterations=7)

    def test_butterfly(self):
        neda.plot.butterfly(self.traj)
        plt.show()

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
