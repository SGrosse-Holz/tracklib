from . import load
from . import write
from . import hdf5

# Register HDF5 I/O for tracklib classes
# Pack everything into a dummy namespace, so we don't pollute the module
# namespace
# This snippet has to be here because it would result in circular imports if
# placed in either trajectory.py/taggedset.py or io/hdf5.py
class stupid_namespace:
    from . import hdf5
    from tracklib import TaggedSet, Trajectory

    for cls in [TaggedSet, Trajectory]:
        hdf5.reader_writer_registry["602354027_tracklib."+cls.__name__] = (cls, hdf5.write_generic_class, hdf5.read_generic_class)

    # Historically used identifiers (should continue to support these):
    # 602354027_tracklib.Trajectory_1N1d
    # 602354027_tracklib.Trajectory_1N2d
    # 602354027_tracklib.Trajectory_1N3d
    # 602354027_tracklib.Trajectory_2N1d
    # 602354027_tracklib.Trajectory_2N2d
    # 602354027_tracklib.Trajectory_2N3d
    for N in [1, 2]:
        for d in [1, 2, 3]:
            hdf5.reader_writer_registry[f"602354027_tracklib.Trajectory_{N}N{d}d"] = (Trajectory, hdf5.write_generic_class, hdf5.read_generic_class)

del stupid_namespace
