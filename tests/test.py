import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import numpy as np
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
        except AssertionError as err:
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
                lines = traj.plot_spatial()
                self.assertEqual(len(lines), N)

                lines = traj.plot_spatial(linestyle='-')
                if N == 2:
                    lines = traj.plot_spatial(linestyle=['-', '--'])
                else:
                    with self.assertRaises(ValueError):
                        lines = traj.plot_spatial(linestyle=['-', '--'])

            # Modifiers
            times2 = traj.rescale(2)
            self.assert_array_equal(traj.data*2, times2.data)
            plus1 = traj.offset(1)
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

    def test_elementaccess(self):
        self.assertEqual(self.ls[1], 2)

    def test_mergein(self):
        newls = tl.TaggedSet(zip([4, 5, 6], [["d"], [], "e"]))
        self.ls.mergein(newls, additionalTags='new')
        self.assertListEqual(self.ls._data, [1, 2, 3, 4, 5, 6])
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c', 'd', 'e', 'new'})

        self.ls &= newls
        self.assertEqual(len(self.ls), 9)
        
        self.ls.makeSelection(tags='new')
        self.assertSetEqual(set(self.ls), {4, 5, 6})

    def test_makeTagsSet(self):
        self.assertSetEqual(tl.TaggedSet.makeTagsSet("foo"), {"foo"})
        self.assertSetEqual(tl.TaggedSet.makeTagsSet(["foo"]), {"foo"})
        with self.assertRaises(ValueError):
            tl.TaggedSet.makeTagsSet(1)

    def test_selection(self):
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

#     def test_homogeneous(self):
#         self.assertTrue(self.ls.isHomogeneous())
#         lsinh = tl.TaggedSet(zip([1, 2., 3], [["a", "b"], "a", ["b", "c"]]))
#         self.assertFalse(lsinh.isHomogeneous())
#         st = type('st', (int,), {})
#         lsinh = tl.TaggedSet(zip([1, st(5), 3], [["a", "b"], "a", ["b", "c"]]))
#         self.assertTrue(lsinh.isHomogeneous(int, allowSubclass=True))
#         self.assertFalse(lsinh.isHomogeneous(int, allowSubclass=False))
# 
#     def test_getHom(self):
#         imag = self.ls.getHom('imag')
#         self.assertEqual(imag, 0)
#         with self.assertRaises(RuntimeError):
#             self.ls._data[0] = 1j
#             imag = self.ls.getHom('imag')

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

class TestUtilUtil(myTestCase):
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

class TestUtilmcmc(myTestCase):
    @patch('builtins.print')
    def test_mcmc(self, mock_print):
        # Simply use the example from the docs
        class normMCMC(tl.util.mcmc.Sampler):
            callback_tracker = 0

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
        
        mc = normMCMC()
        mc.configure(iterations=10, burn_in=5, log_every=2, show_progress=False)
        logL, vals = mc.run(1)

        self.assertEqual(mock_print.call_count, 10)
        self.assertEqual(len(logL), 10)
        self.assertEqual(len(vals), 5)
        self.assertGreater(mc.callback_tracker, 0)

        mc.config['best_only'] = True
        logL, val = mc.run(1)
        self.assertIsInstance(logL, float)
        self.assertIsInstance(val, float)

class TestModelsRouse(myTestCase):
    def setUp(self):
        self.model = tl.models.Rouse(5, 1, 1, 1)

    def test_operators(self):
        self.assertTrue(self.model == tl.models.Rouse(5, 1, 1, 1, setup=False))
        self.assertFalse(self.model == tl.models.Rouse(6, 1, 1, 1, setup=False))

        self.assertEqual(repr(self.model), "rouse.Model(N=5, D=1, k=1, k_extra=1)")
        self.assertEqual(repr(tl.models.Rouse(5, 1, 1, 1, extrabond=(2, 4))), "rouse.Model(N=5, D=1, k=1, k_extra=1, extrabond=(2, 4))")

    def test_matrices(self):
        A, S = self.model.give_matrices(False)
        self.assert_array_equal(A, np.array([
            [-1,  1,  0,  0,  0],
            [ 1, -2,  1,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  1, -2,  1],
            [ 0,  0,  0,  1, -1]]))
        self.assert_array_equal(S, 2*np.eye(5))

        A, S = self.model.give_matrices(True)
        self.assert_array_equal(A, np.array([
            [-2,  1,  0,  0,  1],
            [ 1, -2,  1,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  1, -2,  1],
            [ 1,  0,  0,  1, -2]]))
        self.assert_array_equal(S, 2*np.eye(5))

        A, S = self.model.give_matrices(False, tethered=True)
        self.assert_array_equal(A, np.array([
            [-2,  1,  0,  0,  0],
            [ 1, -2,  1,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  1, -2,  1],
            [ 0,  0,  0,  1, -1]]))
        self.assert_array_equal(S, 2*np.eye(5))

    def test_check_setup(self):
        self.model.check_setup_called(dt=1)

        mod = tl.models.Rouse(5, 1, 1, 1, setup=False)
        with self.assertRaises(RuntimeError):
            mod.check_setup_called()

        mod.check_setup_called(dt=1, run_if_necessary=True)
        mod.k = 2
        with self.assertRaises(RuntimeError):
            mod.check_setup_called(dt=1)

    def test_propagate(self):
        _, C0 = self.model.steady_state(False)
        M0 = np.linspace(0, 1, self.model.N)

        self.model.propagate = self.model._propagate_ode
        M1, C1 = self.model.propagate(M0, C0, 1, True)

        self.model.propagate = self.model._propagate_exp
        M2, C2 = self.model.propagate(M0, C0, 1, True)

        self.assertTrue(np.allclose(M1, M2, atol=0.01))
        self.assertTrue(np.allclose(C1, C2, atol=0.01))

    def test_evolve(self):
        conf = self.model.conf_ss(False, d=2)
        self.assertTupleEqual(conf.shape, (5, 2))
        conf = self.model.evolve(conf, True)
        self.assertTupleEqual(conf.shape, (5, 2))

    def test_confs_from_looptrace(self):
        confs = list(self.model.conformations_from_looptrace([False, True, False, True]))
        self.assertEqual(len(confs), 4)

    def test_likelihood(self):
        looptrace = [False, False, True, True]
        trace = np.array([conf[-1][0] - conf[0][0] for conf in self.model.conformations_from_looptrace(looptrace)])
        trace[2] = np.nan

        logL = tl.models.rouse.likelihood(trace, self.model, looptrace, noise=1)
        self.assertIsInstance(logL, float)
        logL = tl.models.rouse._likelihood_filter(trace, self.model, looptrace, noise=1)
        self.assertIsInstance(logL, float)
        logL = tl.models.rouse._likelihood_direct(trace, self.model, looptrace, noise=1)
        self.assertIsInstance(logL, float)

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

class TestAnalysisKLILoopSequence(myTestCase):
    def setUp(self):
        self.lt = tl.analysis.kli.LoopSequence(T=100, numInt=10)

    def test_setup(self):
        self.assertEqual(len(self.lt.t), 9)
        self.assertEqual(len(self.lt.isLoop), 10)

    def test_fromLooptrace(self):
        mylt = tl.analysis.kli.LoopSequence.fromLooptrace([True, False, False, True, True, True, False])
        self.assert_array_equal(mylt.t, np.array([1, 3, 6]))
        self.assert_array_equal(mylt.isLoop, np.array([True, False, True, False]))

    def test_toLooptrace(self):
        ltrace = self.lt.toLooptrace()
        self.assertEqual(len(ltrace), self.lt.T)
        self.assertIs(ltrace.dtype, np.dtype('bool'))
    
    def test_numLoops(self):
        mylt = tl.analysis.kli.LoopSequence.fromLooptrace([True, False, False, True, True, True, False])
        self.assertEqual(mylt.numLoops(), 2)

    def test_plottable(self):
        plt.plot(*self.lt.plottable())

class TestAnalysisKLI(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray([[1, 0], [2, 0.5], [3, 0.2], [4, 0.7]], localization_error=np.array([1, 1]))
        self.model = tl.models.Rouse(5, 1, 1, 1)
    
    def test_traj_likelihood(self):
        logL = tl.analysis.kli.traj_likelihood(self.traj, self.model)
        self.assertIsInstance(logL, float)

    def test_fit_RouseParams(self):
        ds = tl.TaggedSet([self.traj], hasTags=False)
        fitres = tl.analysis.kli.fit_RouseParams(ds, self.model, unknown_params=['D'])
        self.assertTrue(fitres.success)
        fitres = tl.analysis.kli.fit_RouseParams(ds, self.model, unknown_params=['k'])
        self.assertTrue(fitres.success)

    def test_LoopSequenceMCMC(self):
        # Note: the actual sampling is tested on mcmc.Sampler directly. Here we
        # only check that the overridden methods work
        mc = tl.analysis.kli.LoopSequenceMCMC()
        mc.setup(self.traj, self.model)
        mc.stepsize = 0.1 # Workaround: usually this is set in mc.run()
        seq = tl.analysis.kli.LoopSequence(T=len(self.traj), numInt=2)

        newSeq, pf, pb = mc.propose_update(seq)
        self.assertIsInstance(newSeq, tl.analysis.kli.LoopSequence)
        self.assertIsInstance(pf, float)
        self.assertIsInstance(pb, float)

        logL = mc.logL(seq)
        self.assertIsInstance(logL, float)

    def test_LoopTraceMCMC(self):
        # Note: the actual sampling is tested on mcmc.Sampler directly. Here we
        # only check that the overridden methods work
        mc = tl.analysis.kli.LoopTraceMCMC()
        mc.setup(self.traj, self.model)
        mc.stepsize = 0.1 # Workaround: usually this is set in mc.run()
        ltrace = tl.analysis.kli.LoopSequence(T=len(self.traj), numInt=2).toLooptrace()

        newltrace, pf, pb = mc.propose_update(ltrace)
        self.assertEqual(len(newltrace), len(self.traj))
        self.assertIsInstance(pf, float)
        self.assertIsInstance(pb, float)

        logL = mc.logL(ltrace)
        self.assertIsInstance(logL, float)

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
        msd = tl.analysis.msd.MSDtraj(self.traj)
        self.assert_array_equal(msd, self.traj.meta['MSD'])
        self.assert_array_equal(msd, np.array([0, 1, 4, 9, 16, 25]))
        self.assert_array_equal(self.traj.meta['MSDmeta']['N'], np.array([5, 3, 3, 2, 1, 1]))

        self.assert_array_equal(msd, tl.analysis.MSD(self.traj))

    def test_MSDdataset(self):
        msd, N = tl.analysis.msd.MSDdataset(self.ds, giveN=True)
        self.assertEqual(len(msd), 10)
        self.assert_array_equal(N, len(self.ds)*np.linspace(10, 1, 10))

        self.assert_array_equal(msd, tl.analysis.MSD(self.ds))

    def test_dMSD(self):
        t, scal = tl.analysis.msd.dMSD(self.ds)
        t, scal = tl.analysis.msd.dMSD(self.traj)

    def test_scaling(self):
        alpha = tl.analysis.msd.scaling(self.traj, n=5)
        self.assertAlmostEqual(alpha, 2, places=5)
        self.assertTrue(np.isnan(tl.analysis.msd.scaling(tl.Trajectory.fromArray([np.nan, np.nan]), n=1)))

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
        self.assertEqual(len(lines), len(self.ds)+1)

    def test_spatial(self):
        lines = tl.analysis.plots.trajectories_spatial(self.ds)
        self.assertEqual(len(lines), len(self.ds))

        lines = tl.analysis.plots.trajectories_spatial(self.ds2)
        self.assertEqual(len(lines), 2*len(self.ds2))

        self.ds2.addTags("tag")
        lines = tl.analysis.plots.trajectories_spatial(self.ds2, colordict={'tag' : 'k'}, linestyle=['--', ':'])

    def test_distance_dist(self):
        _ = tl.analysis.plots.distance_distribution(self.ds)
        _ = tl.analysis.plots.distance_distribution(self.ds2)

if __name__ == '__main__':
    unittest.main(module=__file__[:-3])
