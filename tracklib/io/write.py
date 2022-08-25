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

        if traj.localization_error is None:
            traj.localization_error = np.nan
        if traj.parity is None:
            traj.parity = np.nan

        trajs[i] = traj

    scipy.io.savemat(filename, {'trajs' : trajs})

def hdf5(data, filename, group=None):
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
        truncated and content written to the root node.

    Notes
    -----
    Caution is advised, since this function will silently overwrite existing
    data (this is most often the desired behavior).
    """
    mode = 'a' # append to existing file or create new if nonexistent
    if group is None:
        mode = 'w' # overwrite file if exists
        group = '/'
        name = None
    else:
        if group.endswith('/'):
            group = group[:-1]

        parts = group.split('/')
        group = '/'.join(parts[:-1])
        name = parts[-1]

        group = group if len(group) > 0 else '/'
        assert len(name) > 0

    # After the above:
    #  - `group` will be a valid string
    #  - `name` might be None (only if group was None)

    with h5py.File(str(filename), mode) as f:
        try:
            f.create_group(group)
        except ValueError: # if group exists, specifically '/'
            pass

        # Silent overwrite
        if name is not None and name in f[group]:
            del f[group][name]

        hdf5_mod.write(data, name, f[group])

def hdf5_subTaggedSet(data, filename, group, refTaggedSet=None):
    """
    Write a subset of an already stored `!TaggedSet` to file

    Sometimes it is handy to store subsets of data in a directly loadable way
    (i.e. as its own `!TaggedSet` object). This would duplicate data and thus
    increase file size, so this function takes advantage of hdf5's hard links
    to store the properly pruned `!TaggedSet` by just linking to the
    corresponding entries in the full data set, which should be located in the
    same file at the `!refTaggedSet` address.

    Parameters
    ----------
    data : TaggedSet
        a `!TaggedSet` with some selection applied. The full data set
        (potentially with a different selection, this does not matter) should
        already be written to `!filename` under the path `!refTaggedSet`.
    filename : str or pathlib.Path
        the file to store things in
    group : str
        the location in the file where to store the new entry
    refTaggedSet : str
        where the full data set is stored in the file.

    Notes
    -----
    Caution is advised, since this function will silently overwrite existing
    data (this is most often the desired behavior).

    This function is intended for storing selections (subsets) of `!TaggedSets`
    such that they can be read from file as complete `!TaggedSets` themselves.
    Usecases include having a big dataset, out of which you routinely need only
    a specific part. If your subset is identified by the tag ``subset`` in the
    big dataset, this is equivalent to

    >>> from tracklib import io
    ...
    ... big_data = io.load.hdf5('file_with_big_data.h5', 'data')
    ... big_data.makeSelection(tags='subset')
    ... data = big_data.copySelection()
    ...
    ... # This can be reduced to a single line by saving the selection beforehand
    ... # when saving the data:
    ... big_data.makeSelection()
    ... io.write.hdf5(big_data, 'file_with_big_data_and_subset.h5', 'data')
    ... big_data.makeSelection(tags='subset')
    ... io.write.hdf5_subTaggedSet(big_data, 'file_with_big_data_and_subset.h5',
    ...                            'data_subset', refTaggedSet='/data')
    ...
    ... # so now when loading the data, we can just do
    ... data = io.load.hdf5('file_with_big_data_and_subset.h5', 'data_subset')

    Note that basically we just shifted the process of making the selection
    from loading to writing. This however can come in very handy when the
    selection process is more involved than a simple tag, or you distribute
    your data to others, who will appreciate an easy way to load just the
    relevant data.
    """
    # Check arguments
    if type(refTaggedSet) != str:
        raise ValueError("Please specify where the full dataset is saved in the file")

    # Dissect group as group/name
    parts = group.split('/')
    group = '/'.join(parts[:-1])
    name = parts[-1]

    group = group if len(group) > 0 else '/'

    if len(name) == 0:
        raise ValueError("Please specify a name (group) for your new entry")

    # We rely on the internal structure of TaggedSet, so check that we're up to
    # date about that
    vars_and_types = {'_data' : list, '_tags' : list, '_selected' : list}
    assert vars(data).keys() == vars_and_types.keys()
    for var, typ in vars_and_types.items():
        assert type(getattr(data, var)) == typ

    # Prepare the data to be saved
    # _data is just assembled as hdf5 paths here, since these will be hard links
    pseudo_TaggedSet = dict()
    pseudo_TaggedSet['_data'] = [refTaggedSet+f'/_data/{i}' for i, sel in enumerate(data._selected) if sel]
    pseudo_TaggedSet['_tags'] = [t for t, sel in zip(data._tags, data._selected) if sel]
    pseudo_TaggedSet['_selected']= len(data)*[True]

    # Write
    # Note that we never check that the softlinks actually work / make sense.
    # The only thing we do is copy stuff from the original, so at least it has
    # to exist
    with h5py.File(str(filename), 'a') as f:
        # First things first: make sure that the hard links work out
        refData = f[refTaggedSet+'/_data']
        if isinstance(refData, h5py.Dataset) or "0" not in refData:
            raise ValueError("Cannot save subset of TaggedSet of built-in type (these are not saved as groups, i.e. we cannot link to individual entries)")

        ls = pseudo_TaggedSet['_data']
        for i, path in enumerate(ls):
            ls[i] = f[path]

        # Set up new entry
        try:
            f.create_group(group)
        except ValueError:
            pass # group exists, that's okay

        # Silent overwrite
        if name in f[group]:
            del f[group][name]

        # Write everything to it
        hdf5_base = f[group].create_group(name)
        for key in f[refTaggedSet]:
            if key in pseudo_TaggedSet:
                hdf5_mod.write(pseudo_TaggedSet[key], key, hdf5_base)
            else: # pragma: no cover
                hdf5_base[key] = f[refTaggedSet][key] # hard link

        # Specifically, this copies _HDF5_ORIG_TYPE_
        for key, value in f[refTaggedSet].attrs.items():
            hdf5_base.attrs[key] = value
