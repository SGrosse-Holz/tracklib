"""
Some functions for writing trajectories / data sets to file
"""
import numpy as np
import scipy.io
import csv as csv_mod
import h5py

from . import hdf5 as hdf5_mod

def csv(data, filename, header=True, delimiter='\t'):
    """
    A quick-and-dirty csv-writer. Might be updated eventually.

    Parameters
    ----------
    data : `TaggedSet` of `Trajectory`
        the data set to write to file
    filename : str
        the file to write to
    header : bool, optional
        whether to write a header line with column names
    delimiter : chr, optional
        which character to use as delimiter

    Notes
    -----
    The columns in the file will be ``['id', 'frame', 'x', 'y', 'z', 'x2',
    'y2', 'z2']``, where of course only those coordinates present in the data
    set will be written.

    Missing frames, i.e. those where all of the coordinates are ``np.nan`` will
    simply be omitted.

    Since `TaggedSet` and `Trajectory` have more structure than can reasonably
    represented in ``.csv`` files, this function has no aspirations of writing
    the whole structure to file. It can write only the "core" data, i.e. the
    actual trajectories.
    """
    N = data.map_unique(lambda traj : traj.N)
    d = data.map_unique(lambda traj : traj.d)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv_mod.writer(csvfile, delimiter=delimiter)

        if header:
            head = ['id', 'frame']
            coords = ['x', 'y', 'z'][:d]
            if N == 2:
                coords += [c + '2' for c in coords]
            writer.writerow(head + coords)

        for traj_id, traj in enumerate(data):
            for frame in range(traj.T):
                if np.all(np.isnan(traj.data[:, frame, :])):
                    continue

                head = [traj_id, frame]
                coords = []
                for n in range(N):
                    coords += traj.data[n, frame, :].tolist()

                writer.writerow(head + coords)

def mat(data, filename):
    """
    Write a dataset to MATLAB's .mat format

    This will produce a cell array containing the individual trajectories as
    structs. All the meta-data is passed along as well. The tags associated
    with the trajectory will be written to an entry ``'tracklib_tags'``.

    Parameters
    ----------
    data : TaggedSet of Trajectory
        the data set to write
    filename : str
        the file to write to
    """
    trajs = np.empty(len(data), dtype=object)
    for i, (traj, tags) in enumerate(data(giveTags=True)):
        traj.tracklib_tags = list(tags)
        trajs[i] = traj

    scipy.io.savemat(filename, {'trajs' : trajs})

def hdf5(data, filename, group=None, name=None):
    """
    Write to HDF5 file

    Parameters
    ----------
    data : TaggedSet or dict
        the stuff to write
    filename : str or pathlib.Path
        where to write to
    group : str
        where in the file to write the data. If unspecified, the file will be
        truncated and content written to the root node. If you want to write to
        an attribute of an existing or to-be-created group, specify this as
        ``group/{attr}``.
    name : str
        the name of the new entry to create for storing the data. If ``None``
        (default) store directly to `!group`.
    """
    mode = 'a' # append to existing file or create new if nonexistent
    if group is None:
        mode = 'w' # overwrite file if exists
        group = '/'
    elif name is None:
        # group is specified as group/name
        parts = group.split('/')
        group = '/'.join(parts[:-1])
        name = parts[-1]

        group = group if len(group) > 0 else '/'
        name = name if len(name) > 0 else None

    # After the above:
    #  - `group` will be a valid string
    #  - `name` might be None

    with h5py.File(str(filename), mode) as f:
        try:
            f.create_group(group)
        except ValueError: # if group exists, specifically '/'
            pass

        hdf5_mod.write(data, name, f[group])
