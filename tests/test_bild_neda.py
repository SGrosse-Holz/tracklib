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

class test_Environment(myTestCase):
    def setUp(self):
        self.traj = tl.Trajectory.fromArray(np.array([1, 2, np.nan, 4]), localization_error=[0.5])
        self.model = bild.models.FactorizedModel([
            scipy.stats.maxwell(scale=1),
            scipy.stats.maxwell(scale=4),
            ], d=1)
        sampler = bild.mcmc.FullMCMC()
        sampler.configure(iterations=10, burn_in=5)

        priorfam = bild.ParametricFamily((-1e-5,), ([None, -1e-10]))
        priorfam.get = lambda logq : bild.priors.GeometricPrior(logq=logq, nStates=2)
        self.env = bild.neda.Environment(self.traj, self.model, sampler, priorfam)

        # random seeds are set such that the first run does not collapse, the
        # second does
        np.random.seed(6)
        self.mcmcrun = self.env.runMCMC(-1)
        np.random.seed(0)
        self.mcmcrun_collapsed = self.env.runMCMC(-1)

    def test_runMCMC(self):
        self.assertEqual(len(self.mcmcrun.logLs), 10)
        self.assertEqual(len(self.mcmcrun.samples), 5)
        self.assertEqual(len(self.mcmcrun_collapsed.logLs), 10)
        self.assertEqual(len(self.mcmcrun_collapsed.samples), 5)

    def test_normal(self):
        p = self.env.posterior_density(self.mcmcrun, self.mcmcrun.samples[0][0])
        self.assertGreater(p, 0)
        self.assertLess(p, np.inf)

        ev = self.env.evidence(self.mcmcrun)
        # self.assertLess(ev, np.max(self.mcmcrun.logLs_trunc())) # in theory, but numerical error, especially in a small sample like this

    @patch("builtins.print")
    def test_collapsed(self, mock_print):
        p = self.env.posterior_density(self.mcmcrun_collapsed, self.mcmcrun_collapsed.samples[0][0])
        self.assertTrue(np.isinf(p))
        # mock_print.assert_called() # this is not important and might be changed

        ev = self.env.evidence(self.mcmcrun_collapsed)
        self.assertEqual(ev, np.max(self.mcmcrun_collapsed.logLs_trunc()))
        # mock_print.assert_called() # this is not important and might be changed

if __name__ == '__main__': # pragma: no cover
    unittest.main(module=__file__[:-3])
