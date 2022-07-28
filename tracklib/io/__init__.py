from . import load
from . import write

# Register HDF5 I/O for tracklib classes
# Pack everything into a dummy namespace, so we don't pollute the module
# namespace
class stupid_namespace:
    from . import hdf5
    from tracklib import TaggedSet
    from tracklib.trajectory import (Trajectory_1N1d, Trajectory_1N2d, Trajectory_1N3d,
                                     Trajectory_2N1d, Trajectory_2N2d, Trajectory_2N3d,
                                     )

    for cls in [Trajectory_1N1d, Trajectory_1N2d, Trajectory_1N3d,
                Trajectory_2N1d, Trajectory_2N2d, Trajectory_2N3d,
                TaggedSet,
                ]:
        hdf5.reader_writer_registry["602354027_tracklib."+cls.__name__] = (cls, hdf5.write_generic_class, hdf5.read_generic_class)

del stupid_namespace
