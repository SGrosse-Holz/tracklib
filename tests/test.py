import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

import joblib
import tempfile
import numpy as np

import unittest
from unittest.mock import patch

from context import tracklib
from tracklib import *
import tracklib.util as util

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

# Test cases are run alphabetically, so we can specify
# test levels simply as digit in the class name
# The main purpose of this is to get hierarchical output
# if anything fails.

class Test0TaggedList(unittest.TestCase):
    def test_init(self):
        ls = TaggedList()
        
    def test_append(self):
        ls = TaggedList()
        ls.append(1)
        ls.append(2, 'a')
        ls.append(3, ['b', '_all'])
        
    def test_generate(self):
        ls = TaggedList.generate(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))
        self.assertListEqual(ls._data, [1, 2, 3])
        self.assertListEqual(ls._tags, [{"a", "b", "_all"}, {"a", "_all"}, {"b", "c", "_all"}])
        
class Test1TaggedList(unittest.TestCase):
    def setUp(self):
        self.ls = TaggedList.generate(zip([1, 2, 3], [["a", "b"], "a", ["b", "c"]]))
        
    def test_iteration(self):
        for ind, val in enumerate(self.ls):
            self.assertEqual(val, self.ls._data[ind])

    def test_homogeneous(self):
        self.assertTrue(self.ls.isHomogeneous())
        lsinh = TaggedList.generate(zip([1, 2., 3], [["a", "b"], "a", ["b", "c"]]))
        self.assertFalse(lsinh.isHomogeneous())
        st = type('st', (int,), {})
        lsinh = TaggedList.generate(zip([1, st(5), 3], [["a", "b"], "a", ["b", "c"]]))
        self.assertTrue(lsinh.isHomogeneous(int, allowSubclass=True))
        self.assertFalse(lsinh.isHomogeneous(int, allowSubclass=False))

    def test_selection(self):
        self.ls.makeSelection("a")
        self.assertSetEqual({*self.ls}, {1, 2})

        self.ls.makeSelection(["a", "b"], logic=all)
        self.assertSetEqual({*self.ls}, {1})

        self.ls.makeSelection(["a", "b"], logic=any)
        self.assertSetEqual({*self.ls}, {1, 2, 3})

    def test_len(self):
        self.assertEqual(len(self.ls), 3)

    def test_makeTagsSet(self):
        self.assertSetEqual(TaggedList.makeTagsSet("foo"), {"foo"})
        self.assertSetEqual(TaggedList.makeTagsSet(["foo"]), {"foo"})
        with self.assertRaises(ValueError):
            TaggedList.makeTagsSet(1)

    def test_makeTagsList(self):
        self.assertListEqual(TaggedList.makeTagsList("foo"), ["foo"])
        self.assertListEqual(TaggedList.makeTagsList({"foo"}), ["foo"])
        with self.assertRaises(ValueError):
            TaggedList.makeTagsList(1)

    def test_tagset(self):
        self.assertSetEqual(self.ls.tagset(), {'a', 'b', 'c'})
        self.assertSetEqual(self.ls.tagset(omit_all=False), {'_all', 'a', 'b', 'c'})

    def test_byTag(self):
        self.ls.makeSelection('c')
        self.assertSetEqual({*self.ls.byTag()}, {3})
        self.assertSetEqual({*self.ls.byTag('a')}, {1, 2})
        self.assertSetEqual({*self.ls.byTag(['a', 'b'], logic=all)}, {1})
        self.assertSetEqual({*self.ls.byTag(['a', 'b'], logic=any)}, {1, 2, 3})
        
    def test_subset(self):
        sub = self.ls.subsetByTag('a')
        self.assertListEqual(sub._data, [1, 2])

class Test0Trajectory(myTestCase):
    def test_fromArray(self):
        traj = Trajectory.fromArray(np.zeros((10,)))
        self.assertTupleEqual(traj._data.shape, (1, 10, 1))
        self.assertIs(traj.label, None)

        traj = Trajectory.fromArray(np.zeros((10, 2)))
        self.assertTupleEqual(traj._data.shape, (1, 10, 2))

        traj = Trajectory.fromArray(np.zeros((1, 10, 3)))
        self.assertTupleEqual(traj._data.shape, (1, 10, 3))

        traj = Trajectory.fromArray(np.zeros((2, 10, 1)))
        self.assertTupleEqual(traj._data.shape, (2, 10, 1))

        traj = Trajectory.fromArray(np.zeros((2, 10, 2)))
        self.assertTupleEqual(traj._data.shape, (2, 10, 2))

        traj = Trajectory.fromArray(np.zeros((2, 10, 3)))
        self.assertTupleEqual(traj._data.shape, (2, 10, 3))

        with self.assertRaises(ValueError):
            traj = Trajectory.fromArray(np.zeros((10, 4)))
        with self.assertRaises(ValueError):
            traj = Trajectory.fromArray(np.zeros((3, 10, 1)))
        with self.assertRaises(ValueError):
            traj = Trajectory.fromArray(np.zeros((1, 1, 1, 1)))
        with self.assertRaises(ValueError):
            traj = Trajectory.fromArray(np.zeros((1, 10, 3)))
            traj.plot_spatial(dims=(0, 3))
        with self.assertRaises(ValueError):
            traj = Trajectory.fromArray(np.zeros((2, 10, 3)))
            traj.plot_spatial(dims=(0, 3))

class Test1Trajectory(myTestCase):
    def setUp(self):
        self.T = 10
        self.Ns = [1, 1, 1, 2, 2, 2]
        self.ds = [1, 2, 3, 1, 2, 3]
        self.trajs = [Trajectory.fromArray(np.zeros((N, self.T, d))) for N, d in zip(self.Ns, self.ds)]

    def test_interface(self):
        """
        test all the methods whose signature is identical for all the
        subclasses
        """
        for traj, N, d in zip(self.trajs, self.Ns, self.ds):
            self.assertEqual(len(traj), self.T)
            self.assertEqual(traj.N, N)
            self.assertEqual(traj.T, self.T)
            self.assertEqual(traj.d, d)

            self.assertTupleEqual(traj[3:5].shape, (N, 2, d))
            self.assert_array_equal(traj[2], np.zeros((N, d)))

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

            msd = traj.msd()
            self.assertTupleEqual(msd.shape, (self.T,))
            (_, Nmsd) = traj.msd(giveN=True)
            self.assertTupleEqual(Nmsd.shape, (self.T,))

            if N == 2:
                rel = traj.relativeTrajectory()
                self.assertTupleEqual(rel._data.shape, (1, self.T, d))
                mag = rel.absTrajectory()
                self.assertTupleEqual(mag._data.shape, (1, self.T, 1))
            elif N == 1:
                mag = traj.absTrajectory()
                self.assertTupleEqual(mag._data.shape, (1, self.T, 1))
                with self.assertRaises(NotImplementedError):
                    rel = traj.relativeTrajectory()

class TestUtil(myTestCase):
    def test_msd(self):
        # Testing 1d trajectories
        traj = [1, 2, 3, 4]
        self.assert_array_equal(util.msd(traj), np.array([0, 1, 4, 9]))

        (msd, N) = util.msd(traj, giveN=True)
        self.assert_array_equal(N, np.array([4, 3, 2, 1]))

        (msd, N) = util.msd(np.array([1, 2, np.nan, 4]), giveN=True)
        self.assert_array_equal(msd, np.array([0, 1, 4, 9]))
        self.assert_array_equal(N, np.array([3, 1, 1, 1]))

        with self.assertRaises(ValueError):
            util.msd([[1, 2, 3, 4]])

        # Correct handling of multiple trajectories
        traj = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]).swapaxes(0, 2)
        msd = util.msd(traj)
        self.assert_array_equal(msd, np.array([[0, 1, 4], [0, 1, 4], [0, 1, 4]]).swapaxes(0, 1))

    def test_sampleMSD(self):
        msd = np.sqrt(np.arange(10))
        trajs = util.sampleMSD(msd, n=2)
        self.assertTupleEqual(trajs.shape, (10, 2))
        acf = np.zeros((10,))
        acf[0] = 1
        trajs = util.sampleMSD(acf, n=2, isCorr=True)
        self.assertTupleEqual(trajs.shape, (10, 2))

class TestAnalysis(myTestCase):
    def setUp(self):
        tags = ["foo", ["foo", "bar"], ["bar"], {"foobar", "bar"}, "foo"]
        self.ntraj = len(tags)
        self.N = 1
        self.T = 10
        self.d = 2
        msd = np.sqrt(np.arange(self.T))
        trajs = util.sampleMSD(msd, n=self.N*self.d*len(tags), subtractMean=False)

        def gen():
            for i, mytags in enumerate(tags):
                mytracelist = [trajs[:, ((i*self.N + n)*self.d):((i*self.N + n+1)*self.d)] \
                               for n in range(self.N)]
                yield (Trajectory.fromArray(mytracelist), mytags)
        self.ds = TaggedList.generate(gen())

    def test_setup(self):
        self.assertEqual(len(self.ds), 5)
        for traj in self.ds:
            self.assertEqual(traj.N, self.N)
            self.assertEqual(traj.T, self.T)
            self.assertEqual(traj.d, self.d)

    def test_msd(self):
        msd = analysis.MSD(self.ds)
        self.assertTupleEqual(msd.shape, (self.T,))
        (_, N) = analysis.MSD(self.ds, giveN=True)
        self.assertTupleEqual(N.shape, (self.T,))
        
    def test_hist_lengths(self):
        h = analysis.hist_lengths(self.ds)
        self.assert_array_equal(h[0], np.array(self.ntraj))

    def test_plot_msds(self):
        lines = analysis.plot_msds(self.ds)
        self.assertEqual(len(lines), self.ntraj+1)

        self.ds.makeSelection({'foo'})
        lines = analysis.plot_msds(self.ds, label='ensemble')
        self.assertEqual(len(lines), 4)

    def test_plot_trajectories(self):
        lines = analysis.plot_trajectories(self.ds)
        self.assertEqual(len(lines), self.ntraj)
        self.ds.makeSelection("foo")
        lines = analysis.plot_trajectories(self.ds)
        self.assertEqual(len(lines), 3)

    def test_MSDcontrol(self):
        # Note: the MSD of generated trajectories (such as our sample) can lead
        # to exceptions bc it is noisy. Therefore, use a definitely clean one
        msd = np.sqrt(np.arange(len(self.ds._data[0])))
        control = analysis.MSDcontrol(self.ds, msd)
        self.assertEqual(len(control), len(self.ds))
        self.ds.makeSelection("foo")
        control = analysis.MSDcontrol(self.ds, msd)
        self.assertEqual(len(control), 3)
        with self.assertRaises(RuntimeError):
            control = analysis.MSDcontrol(self.ds, -msd)

    def test_KLD_PC(self):
        KLD = analysis.KLD_PC(self.ds, n=2, k=5)
        
class TestAnalysisKLDestimator(myTestCase):
    def setUp(self):
        tags = ["foo", ["foo", "bar"], ["bar"], {"foobar", "bar"}, "foo"]
        self.ntraj = len(tags)
        self.N = 2
        self.T = 100
        self.d = 2
        msd = np.sqrt(np.arange(self.T))
        trajs = util.sampleMSD(msd, n=self.N*self.d*len(tags), subtractMean=False)

        def gen():
            for i, mytags in enumerate(tags):
                mytracelist = [trajs[:, ((i*self.N + n)*self.d):((i*self.N + n+1)*self.d)] \
                               for n in range(self.N)]
                yield (Trajectory.fromArray(mytracelist), mytags)
        self.ds = TaggedList.generate(gen())
        self.est = analysis.KLDestimator(self.ds)
        self.est.setup(bootstraprepeats=2, processes=2, n=[2, 5], k=2, dt=1)

    def test_useRelative(self):
        self.est.useRelative()
        self.assertTupleEqual(self.est.ds._data[0]._data.shape, (1, self.T, self.d))

    def test_useDistance(self):
        self.est.useDistance()
        self.assertTupleEqual(self.est.ds._data[0]._data.shape, (1, self.T, 1))

    def test_run(self):
        res = self.est.run()
        self.assertEqual(len(res), 4)

    @patch("tracklib.analysis.multiprocessing.Pool")
    def test_noparallel(self, mockmap):
        self.est.setup(processes=1)
        res = self.est.run()
        mockmap.assert_not_called()

if __name__ == '__main__':
    unittest.main()
