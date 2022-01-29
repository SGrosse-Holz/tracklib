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
msdfit = tl.analysis.msdfit

"""
exec "norm jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kddO]kV?__all__j>>
"""
__all__ = [
    'TestSplineFit',
]

# We test mostly the library implementations, since the base class `Fit` is
# abstract
# We also test on so few data that all the results are useless and we don't
# attempt to check for correctness of the fit. This code is supposed to be
# technical tests and not benchmarks, so it should run fast.

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

class TestDiffusive(myTestCase):
    def setUp(self):
        def traj():
            return tl.Trajectory.fromArray(np.cumsum(np.random.normal(size=(10, 3)), axis=0))

        self.data = tl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    def testSpline(self):
        fit = msdfit.lib.SplineFit(self.data, ss_order=1, n=4)
        res = fit.run(verbosity=0, maxfev=500)

        for i in [0, 1]:
            self.assertGreater(res['params'][i], 0)
            self.assertLess(res['params'][i], 1)

        res2 = fit.run(init_from={'params' : np.array([1e-8, 0.5, 0, 0, 0, 0]), 'logL' : 0})

    def testNPX(self):
        pass # TODO

class TestRouseLoci(myTestCase):
    def setUp(self):
        model = tl.models.rouse.Model(10)
        tracked = [2, 7]
        def traj():
            conf = model.conf_ss()
            traj = []
            for _ in range(10):
                conf = model.evolve(conf)
                traj.append(conf[tracked[0] - tracked[1]])

            return tl.Trajectory.fromArray(traj)

        self.data = tl.TaggedSet((traj() for _ in range(10)), hasTags=False)

    @patch('builtins.print')
    def testSpline(self, mock_print):
        fit = msdfit.lib.SplineFit(self.data, ss_order=0, n=4)
        res = fit.run()

        for i in [0, 1]:
            self.assertGreater(res['params'][i], 0)
            self.assertLess(res['params'][i], 2)

        # Test refining a spline fit
        fit2 = msdfit.lib.SplineFit(self.data, ss_order=0, n=6,
                                    previous_spline_fit_and_result = (fit, res),
                                    )

        with self.assertRaises(RuntimeError):
            res2 = fit2.run(optimization_steps=('gradient',), maxfev=10)


    def testRouse(self):
        fit = msdfit.lib.TwoLocusRouseFit(self.data)
        fit.fix_values += [(0, -np.inf), (3, -np.inf), (6, -np.inf)] # no localization error
        res = fit.run(full_output=True, optimization_steps=(dict(method='Nelder-Mead', options={'fatol' : 0.1, 'xatol' : 0.01}),))[-1][0]

        self.assertEqual(res['params'][0], -np.inf)

if __name__ == '__main__':
    unittest.main(module=__file__[:-3])
