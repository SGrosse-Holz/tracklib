import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
np.seterr(all='raise') # pay attention to details
from matplotlib import pyplot as plt

import unittest
from unittest.mock import patch

from context import tracklib as tl

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

class Test0Trajectory(myTestCase):
    def test_fromArray(self):
        traj = tl.Trajectory.fromArray(np.zeros((10,)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_1N1d)

        traj = tl.Trajectory.fromArray(np.zeros((10, 2)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_1N2d)

        traj = tl.Trajectory.fromArray(np.zeros((1, 10, 3)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_1N3d)

        traj = tl.Trajectory.fromArray(np.zeros((2, 10, 1)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_2N1d)

        traj = tl.Trajectory.fromArray(np.zeros((2, 10, 2)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_2N2d)

        traj = tl.Trajectory.fromArray(np.zeros((2, 10, 3)))
        self.assertIsInstance(traj, tl.trajectory.Trajectory_2N3d)

        traj = tl.Trajectory.fromArray(np.zeros((2, 5, 2)), t=[1, 2, 4, 5, 7])
        self.assertEqual(traj.T, 7)
        self.assertTrue(np.all(np.isnan(traj[[2, 5]])))

        with self.assertRaises(ValueError):
            traj = tl.Trajectory.fromArray(np.zeros((1, 1, 1, 1)))
        with self.assertRaises(ValueError):
            traj = tl.Trajectory.fromArray(np.zeros((5, 1, 1)))

class Test1Trajectory(myTestCase):
    def setUp(self):
        self.T = 10
        self.Ns = [1, 1, 1, 2, 2, 2]
        self.ds = [1, 2, 3, 1, 2, 3]
        self.trajs = [tl.Trajectory.fromArray(np.zeros((N, self.T, d)), localization_error=np.ones((d,)), parity='even') \
                      for N, d in zip(self.Ns, self.ds)]
        self.trajs[5].meta['localization_error'] = np.ones((2, 3)) # To check that shape

    def test_valid_frames(self):
        traj = tl.Trajectory.fromArray(np.zeros((2, 5, 2)), t=[1, 2, 4, 5, 7])
        self.assertEqual(traj.valid_frames(), 5)

        traj.data[0, 2, :] = 0
        self.assertEqual(traj.valid_frames(), 5)

        traj.data[1, 2, :] = 0
        self.assertEqual(traj.valid_frames(), 6)

    def test_interface(self):
        """
        test all the methods whose signature is identical for all the
        subclasses
        """
        for traj, N, d in zip(self.trajs, self.Ns, self.ds):
            # Basic
            self.assertEqual(len(traj), self.T)
            self.assertEqual(traj.N, N)
            self.assertEqual(traj.T, self.T)
            self.assertEqual(traj.d, d)

            if N == 1:
                self.assertTupleEqual(traj[3:5].shape, (2, d))
                self.assert_array_equal(traj[2], np.zeros((d,)))
            else:
                self.assertTupleEqual(traj[3:5].shape, (N, 2, d))
                self.assert_array_equal(traj[2], np.zeros((N, d)))

            # Plotting
            lines = traj.plot_vstime()
            self.assertEqual(len(lines), d)

            if d > 1:
                lines = traj.plot_spatial(label='test plot')
                self.assertEqual(len(lines), N)

                lines = traj.plot_spatial(linestyle='-')
                if N == 2:
                    lines = traj.plot_spatial(linestyle=['-', '--'])
                else:
                    with self.assertRaises(ValueError):
                        lines = traj.plot_spatial(linestyle=['-', '--'])
            else:
                with self.assertRaises(NotImplementedError):
                    lines = traj.plot_spatial(label='test plot')

            # Modifiers
            traj.meta['MSD'] = np.array([5])
            times2 = traj.rescale(2)
            self.assert_array_equal(traj.data*2, times2.data)
            self.assert_array_equal(traj.meta['MSD']*4, times2.meta['MSD'])
            plus1 = traj.offset(1)
            self.assert_array_equal(traj.data+1, plus1.data)
            plus1 = traj.offset([1])
            self.assert_array_equal(traj.data+1, plus1.data)
            plus1 = traj.offset([[1]])
            self.assert_array_equal(traj.data+1, plus1.data)

            if N == 2:
                rel = traj.relative()
                self.assertTupleEqual(rel.data.shape, (1, self.T, d))
                mag = rel.abs()
                self.assertTupleEqual(mag.data.shape, (1, self.T, 1))
                dif = rel.diff(dt=2)
                self.assertTupleEqual(dif.data.shape, (1, self.T-2, d))
                dim = traj.dims([0])
                self.assertTupleEqual(dim.data.shape, (2, self.T, 1))
            elif N == 1:
                mag = traj.abs()
                self.assertTupleEqual(mag.data.shape, (1, self.T, 1))
                with self.assertRaises(NotImplementedError):
                    rel = traj.relative()
                dif = traj.diff(dt=3)
                self.assertTupleEqual(dif.data.shape, (1, self.T-3, d))
                if d >= 2:
                    dim = traj.dims([0, 1])
                    self.assertTupleEqual(dim.data.shape, (1, self.T, 2))

class Test0TaggedSet(unittest.TestCase):
    def test_init(self):
        ls = tl.TaggedSet()

        ls = tl.TaggedSet(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))
        self.assertListEqual(ls._data, [1, 2, 3])
        self.assertListEqual(ls._tags, [{"a", "b"}, {"a"}, {"b", "c"}])

        ls = tl.TaggedSet([1, 2, 3], hasTags=False)
        
    def test_add(self):
        ls = tl.TaggedSet()
        ls.add(1)
        ls.add(2, 'a')
        ls.add(3, {'b', 'c'})
        
class Test1TaggedSet(unittest.TestCase):
    def setUp(self):
        self.ls = tl.TaggedSet(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))

    def test_len(self):
        self.assertEqual(len(self.ls), 3)
        
    def test_iteration(self):
        for ind, val in enumerate(self.ls):
            self.assertEqual(val, self.ls._data[ind])

        for ind, (val, tags) in enumerate(self.ls(giveTags=True)):
            self.assertEqual(val, self.ls._data[ind])
            self.assertSetEqual(tags, self.ls._tags[ind])

        all_values = {1, 2, 3}
        for val in self.ls(randomize=True):
            all_values.remove(val)
        self.assertSetEqual(all_values, set())

    def test_elementaccess(self):
        self.assertEqual(self.ls[1], 2)

    def test_mergein(self):
        newls = tl.TaggedSet(zip([4, 5, 6], [["d"], [], "e"]))
        self.ls.mergein(newls, additionalTags='new')
        self.assertListEqual(self.ls._data, [1, 2, 3, 4, 5, 6])
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c', 'd', 'e', 'new'})

        self.ls |= newls
        self.assertEqual(len(self.ls), 9)
        
        self.ls.makeSelection(tags='new')
        self.assertSetEqual(set(self.ls), {4, 5, 6})

    def test_makeTagsSet(self):
        self.assertSetEqual(tl.TaggedSet.makeTagsSet("foo"), {"foo"})
        self.assertSetEqual(tl.TaggedSet.makeTagsSet(["foo"]), {"foo"})
        with self.assertRaises(ValueError):
            tl.TaggedSet.makeTagsSet(1)

    def test_selection(self):
        for _ in range(5):
            self.ls.makeSelection(nrand=1, random_seed=6542)
            self.assertEqual(len(self.ls), 1)
        for _ in range(5):
            self.ls.makeSelection(prand=0.5)
            self.assertEqual(len(self.ls), 1)

        self.ls.makeSelection(tags="a")
        self.assertSetEqual({*self.ls}, {1, 2})

        self.ls.makeSelection(tags=["a", "b"], logic=all)
        self.assertSetEqual({*self.ls}, {1})

        self.ls.makeSelection(tags=["a", "b"], logic=any)
        self.assertSetEqual({*self.ls}, {1, 2, 3})

        def sel(datum, tags):
            return datum >= 2
        self.ls.makeSelection(selector=sel)
        self.assertSetEqual({*self.ls}, {2, 3})

        self.ls.refineSelection(tags='c')
        self.assertSetEqual({*self.ls}, {3})

        sel = self.ls.saveSelection()
        self.assertListEqual(sel, [False, False, True])

        self.ls.makeSelection()
        self.assertEqual(len(self.ls), 3)
        self.ls.restoreSelection(sel)
        self.assertEqual(len(self.ls), 1)
        self.assertSetEqual({*self.ls}, {3})

        copied = self.ls.copySelection()
        copied.makeSelection()
        self.assertEqual(len(copied), 1)

    def test_deleteSelection(self):
        self.ls.makeSelection(tags='a')
        self.ls.deleteSelection()
        self.assertEqual(len(self.ls), 1)
        self.assertEqual(self.ls[0], 3)

    def test_addTags(self):
        self.ls.addTags('moo')
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c', 'moo'})

    def test_tagset(self):
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c'})

    def test_apply(self):
        def fun(i):
            return i+1
        self.ls.apply(fun)
        self.assertListEqual(self.ls._data, [2, 3, 4])

    def test_process(self):
        # need a TaggedSet of mutable objects
        mutls = tl.TaggedSet(zip(['hello', 'World', '!'], [["a", "b"], "a", ["b", "c"]]))
        mutls.makeSelection(tags="a")
        newls = mutls.process(lambda word : word+'_moo')

        self.assertListEqual(mutls._data, ['hello', 'World', '!'])
        self.assertListEqual(newls._data, ['hello_moo', 'World_moo'])

    def test_map_unique(self):
        def funTrue(x):
            return True
        def fun2(x):
            return 2*x

        self.assertTrue(self.ls.map_unique(funTrue))
        with self.assertRaises(RuntimeError):
            self.ls.map_unique(fun2)
        self.assertIsNone(tl.TaggedSet().map_unique(funTrue))

class TestClean(myTestCase):
    def setUp(self):
        # Split with threshold = 2 => trajectories of lengths [2, 3, 3]
        self.traj = tl.Trajectory.fromArray([1., 2, 4.1, 4.5, 3, -0.5, -1, -0.7])

        # Split with threshold = 3 => 3 trajectories with lengths [4, 5, 6]
        self.ds = tl.TaggedSet()
        self.ds.add(tl.Trajectory.fromArray([0, 0.75, 0.5, 0.3, 5.4, 5.5, 5.3, -2.0, 5.4]))
        self.ds.add(tl.Trajectory.fromArray([1.2, 1.4, np.nan, np.nan, 10.0, 10.2]))

    def test_split_trajectory(self):
        split_trajs = tl.clean.split_trajectory_at_big_steps(self.traj, 2)
        self.assert_array_equal(np.sort([len(traj) for traj in split_trajs]), np.array([2, 3, 3]))

    def test_split_dataset(self):
        split_ds = tl.clean.split_dataset_at_big_steps(self.ds, 3)
        self.assert_array_equal(np.sort([len(traj) for traj in split_ds]), np.array([4, 5, 6]))

class TestIOLoad(myTestCase):
    def test_evalSPT(self):
        filename = "testdata/evalSPT.csv"
        # file contents:
        # 1.0	2.3	1	5	
        # 1.5	2.1	2	5	
        # 2.1	1.5	3	5	
        # 1.9	1.7	5	5	
        # 0.5	9.3	-5	10	
        # 0.4	8.5	-4	10	
        # 1.2	9.1	-6	10	

        ds = tl.io.load.evalSPT(filename, tags={'test'})

        ds.makeSelection(selector = lambda traj, _ : len(traj) <= 3)
        self.assert_array_equal(ds[0][:], np.array([[1.2, 9.1], [0.5, 9.3], [0.4, 8.5]]))

        ds.makeSelection(selector = lambda traj, _ : len(traj) > 3)
        self.assert_array_equal(ds[0][:], np.array([[1.0, 2.3], [1.5, 2.1], [2.1, 1.5], [np.nan, np.nan], [1.9, 1.7]]))

    def test_csv(self):
        filename = "testdata/twoLocus.csv"
        # file contents:
        # line,id,noise,x,y,frame,x2,y2,meta_mean,meta_unique
        # 1,32,10,0.6,0.8,10,0.5,1.0,1,5
        # 2,32,1.5,0.7,0.5,9,0.7,0.8,2,5
        # 3,20,5.3,1.2,1.3,11,-0.8,-1.2,3,6

        # First two are just dummies
        with self.assertRaises(ValueError): # too many columns specified
            ds = tl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 'y', 'z', 't', 'x2', 'y2', 'z2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'mean', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

        with self.assertRaises(AssertionError): # y2 does not have a y to go with it
            ds = tl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 't', 'x2', 'y2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'mean', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

        ds = tl.io.load.csv(
                filename, [None, 'id', 'noise', 'x', 'y', 't', 'x2', 'y2', 'meta_mean', 'meta_unique'],
                meta_post={'meta_mean' : 'mean', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

        ds.makeSelection(selector = lambda traj, _ : len(traj) >= 2)
        self.assert_array_equal(ds[0][:], np.array([[[0.7, 0.5], [0.6, 0.8]], [[0.7, 0.8], [0.5, 1.0]]]))
        self.assert_array_equal(ds[0].meta['noise'], np.array([1.5, 10]))
        self.assertEqual(ds[0].meta['meta_mean'], 1.5)
        self.assertEqual(ds[0].meta['meta_unique'], 5)

        ds.makeSelection(selector = lambda traj, _ : len(traj) < 2)
        self.assert_array_equal(ds[0][:], np.array([[[1.2, 1.3]], [[-0.8, -1.2]]]))
        self.assert_array_equal(ds[0].meta['noise'], np.array([5.3]))
        self.assertEqual(ds[0].meta['meta_mean'], 3)
        self.assertEqual(ds[0].meta['meta_unique'], 6)

        with self.assertRaises(RuntimeError):
            ds = tl.io.load.csv(
                    filename, [None, 'id', 'noise', 'x', 'y', 't', 'x2', 'y2', 'meta_mean', 'meta_unique'],
                    meta_post={'meta_mean' : 'unique', 'meta_unique' : 'unique'}, delimiter=',', skip_header=1)

class TestIOWrite(myTestCase):
    def setUp(self):
        self.ds = tl.TaggedSet()
        self.ds.add(tl.Trajectory.fromArray([0, 0.75, 0.5, 0.3, 5.4, 5.5, 5.3, -2.0, 5.4]))
        self.ds.add(tl.Trajectory.fromArray([1.2, 1.4, np.nan, np.nan, 10.0, 10.2]))

    def test_csv(self):
        filename = "testdata/test_write.csv"
        tl.io.write.csv(self.ds, filename)

        with open(filename, 'r') as f:
            self.assertTrue(f.read() == 'id\tframe\tx\n0\t0\t0.0\n0\t1\t0.75\n0\t2\t0.5\n0\t3\t0.3\n0\t4\t5.4\n0\t5\t5.5\n0\t6\t5.3\n0\t7\t-2.0\n0\t8\t5.4\n1\t0\t1.2\n1\t1\t1.4\n1\t4\t10.0\n1\t5\t10.2\n')

        ds = tl.TaggedSet()
        ds.add(tl.Trajectory.fromArray(np.arange(20).reshape((2, 5, 2))))
        tl.io.write.csv(ds, filename)

        with open(filename, 'r') as f:
            self.assertTrue(f.read() == 'id\tframe\tx\ty\tx2\ty2\n0\t0\t0\t1\t10\t11\n0\t1\t2\t3\t12\t13\n0\t2\t4\t5\t14\t15\n0\t3\t6\t7\t16\t17\n0\t4\t8\t9\t18\t19\n')

class TestUtilUtil(myTestCase):
    def test_distribute_noiselevel(self):
        self.assert_array_equal(tl.util.util.distribute_noiselevel(14, [5, 10, 15]),
                                [1, 2, 3])

    def test_log_derivative(self):
        x = [1, 2, 5, np.nan, 30]
        y = np.array(x)**1.438
        xnew, dy = tl.util.util.log_derivative(y, x, resampling_density=1.5)
        self.assertTrue(np.all(np.abs(dy - 1.438) < 1e-7))

class TestUtilSweep(myTestCase):
    @staticmethod
    def countfun(ds, what, factor=1):
        cnt = 0
        for traj in ds:
            if np.isnan(what):
                cnt += np.sum(np.isnan(traj[:]))
            else:
                cnt += np.sum(traj[:]*factor == what)
        return cnt

    def setUp(self):
        ds = tl.TaggedSet()
        ds.add(tl.Trajectory.fromArray([0, 0.75, 0.5, 0.3, 5.4, 5.5, 5.3, -2.0, 5.4]))
        ds.add(tl.Trajectory.fromArray([1.2, 1.4, np.nan, np.nan, 10.0, 10.2]))

        self.sweep = tl.util.Sweeper(ds, self.countfun)
        _ = tl.util.Sweeper(ds, self.countfun, copy=False)

    def test_preprocess(self):
        self.sweep.preprocess(lambda traj : traj.abs())
    
    def test_run(self):
        res = self.sweep.run({'what' : [5.5, np.nan, 10.0], 'factor' : 1})
        self.assertDictEqual(res, {'what' : [5.5, np.nan, 10.0], 'factor' : [1, 1, 1], 'result' : [1, 2, 1]})

        self.sweep.processes = 2
        res = self.sweep.run({'what' : [10.8, 0.5], 'factor' : [1, 2]})
        self.assertDictEqual(res, {'what' : [10.8, 10.8, 0.5, 0.5], 'factor' : [1, 2, 1, 2], 'result' : [0, 2, 1, 0]})

        self.sweep.processes = 1
        res = self.sweep.run([{'what' : 10.0}, {'what' : 1.5, 'factor' : 2}])
        self.assertDictEqual(res, {'what' : [10.0, 1.5], 'factor' : [None, 2], 'result' : [1, 1]})

class TestMCMCRun(myTestCase):
    def setUp(self):
        sample0 = [0]
        sample1 = [1]
        sample2 = [2]

        self.samples = 5*[sample0] + 2*[sample1] + 3*[sample2]
        self.logLs = np.array(10*[-10] + 5*[-5] + 2*[-2] + 3*[-3])
        self.myrun = tl.util.mcmc.MCMCRun(self.logLs, self.samples)

    def test_init_noparams(self):
        myrun = tl.util.mcmc.MCMCRun()
        self.assertEqual(len(myrun.samples), 0)
        myrun.samples.append(1)
        myrun = tl.util.mcmc.MCMCRun()
        self.assertEqual(len(myrun.samples), 0)

    def test_logLs_trunc(self):
        self.assert_array_equal(self.myrun.logLs_trunc(), self.logLs[10:20])

    def test_best_sample_logL(self):
        best, bestL = self.myrun.best_sample_logL()
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

class TestUtilmcmc(myTestCase):
    @patch('builtins.print')
    def test_mcmc(self, mock_print):
        # Simply use the example from the docs
        class normMCMC(tl.util.mcmc.Sampler):
            callback_tracker = 0

            def __init__(self, stepsize=0.1):
                self.stepsize = stepsize

            def propose_update(self, current_value):
                proposed_value = current_value + np.random.normal(scale=self.stepsize)
                logp_forward = -0.5*(proposed_value - current_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
                logp_backward = -0.5*(current_value - proposed_value)**2/self.stepsize**2 - 0.5*np.log(2*np.pi*self.stepsize**2)
                return proposed_value, logp_forward, logp_backward
                
            def logL(self, value):
                return -0.5*value**2 - 0.5*np.log(2*np.pi)
            
            def callback_logging(self, current_value, best_value):
                self.callback_tracker += np.abs(current_value) + np.abs(best_value)
                print("test")

            def callback_stopping(self, myrun):
                return len(myrun.logLs) > 7
        
        mc = normMCMC()
        mc.configure(iterations=10,
                     burn_in=5,
                     log_every=2,
                     show_progress=False,)
        myrun = mc.run(1)

        self.assertEqual(mock_print.call_count, 10)
        self.assertEqual(len(myrun.logLs), 10)
        self.assertEqual(len(myrun.samples), 5)
        self.assertGreater(mc.callback_tracker, 0)

        mc.config['check_stopping_every'] = 2
        myrun = mc.run(1)

        self.assertEqual(len(myrun.logLs), 8)
        self.assertEqual(len(myrun.samples), 3)

class TestModelsRouse(myTestCase):
    def setUp(self):
        self.model = tl.models.Rouse(5, 1, 1)
        self.model_nosetup = tl.models.Rouse(5, 1, 1, setup_dynamics=False)

    def test_setup(self):
        mod = tl.models.Rouse(5, 1, 1, setup_dynamics=False)
        mod.F[-1] = [1, 0, 0]
        mod.update_dynamics()
        mod.add_tether(0, 1, 0)
        mod.update_dynamics()

    def test_operators(self):
        self.assertTrue(self.model == self.model_nosetup)
        self.model_nosetup.F[-1] = 1
        self.assertFalse(self.model == self.model_nosetup)
        self.assertTrue(self.model != self.model_nosetup) # we have != as mixin for == :)
        self.assertFalse(self.model == tl.models.Rouse(6, 1, 1, setup_dynamics=False))

        self.assertEqual(repr(self.model), "rouse.Model(N=5, D=1, k=1)")
        self.assertEqual(repr(tl.models.Rouse(5, 1, 1, add_bonds=[(2, 4), (1, 3, 0.5)])),
                         "rouse.Model(N=5, D=1, k=1) with 2 additional bonds")

    def test_check_dynamics(self):
        self.model.check_dynamics(dt=1)

        with self.assertRaises(RuntimeError):
            self.model_nosetup.check_dynamics()

        self.model_nosetup.check_dynamics(dt=1, run_if_necessary=True)
        self.model_nosetup.k = 2
        with self.assertRaises(RuntimeError):
            self.model_nosetup.check_dynamics(dt=1, run_if_necessary=False)

    def test_steady_state(self):
        with self.assertRaises(RuntimeError):
            self.model_nosetup.steady_state(additional_tether=False)

        M, C = self.model_nosetup.steady_state()
        self.assert_array_equal(M, np.zeros_like(M))

    def test_propagate(self):
        _, C0 = self.model.steady_state()
        M0 = np.array(3*[np.linspace(0, 1, self.model.N)]).T
        M1, C1 = self.model.propagate(M0, C0, 0.1)

        self.assertTrue(np.allclose(M1, M0, atol=1))
        self.assertTrue(np.allclose(C1, C0, atol=1))

        self.assertTrue(np.allclose(M1, self.model.propagate_M(M0)))
        self.assertTrue(np.allclose(C1, self.model.propagate_C(C0)))

        self.model.D = 0
        M2, C2 = self.model.propagate(M0, np.zeros_like(C0), 0.1)

        self.assertEqual(self.model._dynamics['D'], self.model.D)
        self.assertTrue(np.allclose(M2, M0, atol=1))
        self.assert_array_equal(C2, np.zeros_like(C2))

    def test_conf_ss(self):
        conf = self.model.conf_ss()
        self.assertTupleEqual(conf.shape, (self.model.N,3))

    def test_evolve(self):
        conf = self.model.conf_ss()
        conf = self.model.evolve(conf)
        self.assertTupleEqual(conf.shape, (5,3))

#     def test_update_ensemble_with_observation_1d(self):
#         # these are generic formulae, so have nothing to do with a specific model
#         l, m, c = tl.models.rouse.Model.update_ensemble_with_observation_1d(
#                         np.array([1]), np.array([[1]]), np.array([0]), 1, np.array([1]))
#         self.assertAlmostEqual(l, -1/4-1/2*np.log(4*np.pi), delta=1e-10)
#         self.assertEqual(m[0], 0.5)
#         self.assertEqual(c[0, 0], 0.5)
# 
#         for _ in range(10):
#             M = np.random.rand(1)
#             C = np.eye(1)
#             x = np.random.rand(1) - 1
#             s = np.random.rand() + 1
#             w = np.array([1.])
# 
#             l0, m0, c0 = tl.models.rouse.Model.update_ensemble_with_observation_1d(M, C, x, s, w)
#             l1, m1, c1 = tl.models.rouse.Model.Kalman_update_1d(M, C, x, s, w)
#             print(l0, l1, m0, m1, c0, c1)
#             self.assertAlmostEqual(l0, l1, delta=1e-10)
#             self.assertAlmostEqual(m0, m1, delta=1e-10)
#             self.assertAlmostEqual(c0, c1, delta=1e-10)

class TestModelsStatgauss(myTestCase):
    def test_sampleMSD(self):
        traces = tl.models.statgauss.sampleMSD(np.linspace(0, 5, 10), n=5, subtractMean=True)
        self.assertTupleEqual(traces.shape, (10, 5))
        self.assertTrue(np.allclose(np.mean(traces, axis=0), np.zeros(5)))

    def test_dataset_and_control(self):
        ds = tl.models.statgauss.dataset(np.linspace(0, 5, 10), Ts=[10, 5, 6, None, 4])
        self.assertEqual(len(ds), 5)

        cont = tl.models.statgauss.control(ds, msd=np.linspace(0, 5, 10))
        for original, control in zip(ds, cont):
            self.assertEqual(len(original), len(control))

        cont = tl.models.statgauss.control(ds, msd=lambda t : t)
        for original, control in zip(ds, cont):
            self.assertEqual(len(original), len(control))

        try:
            cont = tl.models.statgauss.control(ds)
        except RuntimeError: # pragma: no cover
                             # the above might or might not work, just want to
                             # check the internal MSD calculation
            pass

class TestAnalysisChi2(myTestCase):
    def setUp(self):
        self.msd = np.linspace(0, 5, 100)
        self.ds1 = tl.models.statgauss.dataset(self.msd, Ts=10*[None])
        self.ds1[2].data[:, 5, :] = np.nan # Just to ensure nan-robustness

        self.ds2 = tl.models.statgauss.dataset(self.msd, N=2, Ts=10*[None])
        self.ds2[5].data[:, 4, :] = np.nan # Just to ensure nan-robustness

    def test_chi2(self):
        dof = tl.analysis.chi2.chi2vsMSD(self.ds1, n=5, msd=self.msd)
        tl.analysis.chi2.summary_plot(self.ds1, dof)

        dof = tl.analysis.chi2.chi2vsMSD(self.ds2, n=5, msd=self.msd)
        tl.analysis.chi2.summary_plot(self.ds2, dof)

class TestAnalysisMSD(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray([1, 2, 3, 4, np.nan, 6])
        self.ds = tl.models.statgauss.dataset(msd=np.linspace(0, 5, 10), Ts=10*[None])

    def test_MSDtraj(self):
        tl.analysis.msd.MSDtraj(self.traj)
        msd = self.traj.meta['MSD']['data']
        self.assert_array_equal(msd, np.array([0, 1, 4, 9, 16, 25]))
        self.assert_array_equal(self.traj.meta['MSD']['N'], np.array([5, 3, 3, 2, 1, 1]))

        del self.traj.meta['MSD']
        msd = tl.analysis.MSD(self.traj, TA=False, recalculate=True)
        self.assert_array_equal(msd, np.array([0, 1, 4, 9, np.nan, 25]))
        self.assert_array_equal(self.traj.meta['MSD']['N'], np.array([1, 1, 1, 1, 0, 1]))

    def test_MSDdataset(self):
        msd, N = tl.analysis.msd.MSDdataset(self.ds, giveN=True)
        self.assertEqual(len(msd), 10)
        self.assert_array_equal(N, len(self.ds)*np.linspace(10, 1, 10))

        msd, var = tl.analysis.msd.MSDdataset(self.ds, givevar=True)
        self.assert_array_equal(np.isnan(var), 10*[False])

        self.assert_array_equal(msd, tl.analysis.MSD(self.ds))

    def test_MSD_functions(self):
        md = tl.analysis.MSD(self.traj, TA=False, function='D', writeto='MD')
        self.assertIs(md, self.traj.meta['MD']['data'])
        self.assert_array_equal(md, [0, 1, 2, 3, np.nan, 5])

        acf = tl.analysis.MSD(self.traj, TA=False, function='SP', writeto='ACF')
        self.assert_array_equal(acf, [1, 2, 3, 4, np.nan, 6])

        mcd = tl.analysis.MSD(self.traj, TA=False,
                              function=lambda xm, xn : np.sum((xm-xn)**3, axis=-1),
                              writeto='MCD')
        self.assert_array_equal(mcd, [0, 1, 8, 27, np.nan, 125])

    def test_dMSD(self):
        t, scal = tl.analysis.msd.dMSD(self.ds)
        t, scal = tl.analysis.msd.dMSD(self.traj)

    def test_scaling(self):
        alpha = tl.analysis.msd.scaling(self.traj, n=5)
        self.assertAlmostEqual(alpha, 2, places=5)
        self.assertTrue(np.isnan(tl.analysis.msd.scaling(tl.Trajectory.fromArray([np.nan, np.nan]), n=1)))

class TestAnalysisVACF(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray([1, 2, 3, 4, np.nan, 6])
        self.ds = tl.models.statgauss.dataset(msd=np.linspace(0, 5, 10), Ts=10*[None])

    def test_VACFtraj(self):
        vacf = tl.analysis.vacf.VACFtraj(self.traj)
        self.assert_array_equal(vacf, self.traj.meta['VACF'])
        self.assert_array_equal(vacf, np.array([1, 1, 1, np.nan, np.nan]))
        self.assert_array_equal(self.traj.meta['VACFmeta']['N'], np.array([3, 2, 1, 0, 0]))

        self.assert_array_equal(vacf, tl.analysis.VACF(self.traj))

    def test_VACFdataset(self):
        vacf, N = tl.analysis.vacf.VACFdataset(self.ds, giveN=True, dt=2)
        self.assertEqual(len(vacf), 4)
        self.assert_array_equal(N, len(self.ds)*np.array([8, 6, 4, 2]))

        vacf, var = tl.analysis.vacf.VACFdataset(self.ds, givevar=True, dt=2)
        self.assert_array_equal(np.isnan(var), 4*[False])

        self.assert_array_equal(vacf, tl.analysis.VACF(self.ds, dt=2))

class TestAnalysisKLD(myTestCase):
    def setUp(self):
        self.ds = tl.models.statgauss.dataset(msd=np.linspace(0, 5, 10), Ts=10*[None])
        for traj in self.ds:
            traj.meta['parity'] = 'even'

    def test_perezcruz(self):
        Dest = tl.analysis.kld.perezcruz(self.ds, n=2, k=5, dt=1)
        self.assertIsInstance(Dest, float)

        for traj in self.ds:
            traj.meta['parity'] = 'odd'
        Dest = tl.analysis.kld.perezcruz(self.ds, n=2, k=5, dt=1)
        self.assertIsInstance(Dest, float)

class TestAnalysisPlots(myTestCase):
    def setUp(self):
        self.ds = tl.models.statgauss.dataset(msd=np.linspace(0, 5, 10), Ts=[None, 5, 6, 7, None, 10, 1, 5])
        self.ds2 = tl.models.statgauss.dataset(msd=np.linspace(0, 5, 10), N=2, Ts=[None, 5, 6, 7, None, 10, 1, 5])

    def test_length_dist(self):
        _ = tl.analysis.plots.length_distribution(self.ds)

    def test_msd_overview(self):
        lines = tl.analysis.plots.msd_overview(self.ds)
        lines = tl.analysis.plots.msd_overview(self.ds, (2, 'seconds'), label='test')
        lines = tl.analysis.plots.msd_overview(self.ds, dt=5)
        self.assertEqual(len(lines), len(self.ds)+1)

    def test_spatial(self):
        lines = tl.analysis.plots.trajectories_spatial(self.ds, fallback_color='#abcdef', color='red')
        self.assertEqual(len(lines), len(self.ds))

        lines = tl.analysis.plots.trajectories_spatial(self.ds2, color=['red', 'blue']) # just testing kwarg
        lines = tl.analysis.plots.trajectories_spatial(self.ds2)
        self.assertEqual(len(lines), 2*len(self.ds2))

        self.ds2.addTags("tag")
        lines = tl.analysis.plots.trajectories_spatial(self.ds2, colordict={'tag' : 'k'}, linestyle=['--', ':'])

    def test_distance_dist(self):
        _ = tl.analysis.plots.distance_distribution(self.ds)
        _ = tl.analysis.plots.distance_distribution(self.ds2)

from test_bild import *

if __name__ == '__main__':
    unittest.main(module=__file__[:-3])
