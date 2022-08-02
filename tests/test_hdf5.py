import os,sys
try: # prevent any plotting
    del os.environ['DISPLAY']
except KeyError:
    pass

from pathlib import Path
import numpy as np
np.seterr(all='raise') # pay attention to details

import unittest
from unittest.mock import patch

from context import tracklib as tl
hdf5 = tl.io.hdf5

"""
  exec "norm 0jjd}O" | let @a="\n'" | exec "g/^class Test/norm w\"Ayt(:let @a=@a.\"',\\n'\"" | norm i__all__ = ["ap}kddO]kV?__all__j>>
"""
__all__ = [
    'TestHDF5',
]

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

class TestHDF5(myTestCase):
    def test_simple(self):
        data = {
            'array' : np.array([1, 2, 3]),
            'noarray' : np.array(3.5),
            'bool' : True,
            'int' : 5,
            'float' : 5.3,
            'complex' : 1+3j,
            'str' : "Hello World",
            'Trajectory' : tl.Trajectory.fromArray([1, 2, 4, 5], meta_test='moo'),
            'TaggedSet' : tl.TaggedSet(),
            'empty_tuple' : tuple(),
            'None' : None,
        }
        data['TaggedSet'].add(5, ['moo', 'foo', 'bar'])
        data['TaggedSet'].add(8.7)
        data['TaggedSet'].add(tl.Trajectory.fromArray([1.5, 3.8]), 'traj')

        filename = 'test.hdf5'
        tl.io.write.hdf5(data, filename)

        # Test partial reading
        self.assertTrue(tl.io.load.hdf5(filename, '/{bool}'))
        self.assertEqual(tl.io.load.hdf5(filename, '{float}'), data['float'])
        # self.assertEqual(tl.io.load.hdf5(filename, '

        data_read = tl.io.load.hdf5(filename)
        self.assertEqual(data.keys(), data_read.keys())

        for key in data:
            if key == 'Trajectory':
                traj = data[key]
                traj_read = data_read[key]
                self.assert_array_equal(traj.data, traj_read.data)
                for meta_key in traj.meta:
                    self.assertEqual(traj.meta[meta_key], traj_read.meta[meta_key])
            elif key == 'TaggedSet':
                self.assertListEqual(data[key]._tags, data_read[key]._tags)
                self.assertListEqual(data[key]._selected, data_read[key]._selected)

                data[key].makeSelection(tags='moo')
                data_read[key].makeSelection(tags='moo')
                self.assertEqual(data[key][0], data_read[key][0])
                
                data[key].makeSelection(tags='traj')
                data_read[key].makeSelection(tags='traj')
                self.assert_array_equal(data[key][0].data, data_read[key][0].data)
            elif key == 'array':
                self.assert_array_equal(data[key], data_read[key])
            elif key == 'empty_tuple':
                self.assertEqual(type(data[key]), type(data_read[key]))
                self.assertEqual(len(data_read[key]), 0)
            elif key == 'None':
                self.assertIs(data_read[key], None)
            else:
                self.assertEqual(data[key], data_read[key])

    def test_errors(self):
        filename = Path('.') / 'hdf5_dummy.hdf5'

        class Test:
            pass
        with self.assertRaises(RuntimeError):
            tl.io.write.hdf5(Test(), filename)

        tl.io.write.hdf5(5, filename, group='/test/{test}')
        res = tl.io.load.hdf5(filename)
        self.assertEqual(res['test']['test'], 5)

    def test_ls(self):
        filename = 'test.hdf5' # is this bad style? relying on a file written in another test...
        ls = hdf5.ls(filename)
        self.assertIn('TaggedSet', ls)
        self.assertIn('[array]', ls)
        self.assertIn('{bool = True}', ls)
        self.assertIn('{_HDF5_ORIG_TYPE_ = dict}', ls)

        ls = hdf5.ls(filename, '/Trajectory')
        self.assertIn('[data]', ls)
        self.assertIn('meta', ls)

        ls = hdf5.ls(filename, depth=2)
        self.assertIn('TaggedSet/[_selected]', ls)
        self.assertIn('Trajectory/meta', ls)
        self.assertIn('empty_tuple/{_HDF5_ORIG_TYPE_ = tuple}', ls)

        self.assertTrue(hdf5.ls(filename, '{bool}'))
        self.assertTrue(hdf5.ls(filename, '/{bool}'))
        self.assertEqual(hdf5.ls(filename, 'empty_tuple/{_HDF5_ORIG_TYPE_}'), 'tuple')

if __name__ == '__main__':
    unittest.main(module=__file__[:-3])
