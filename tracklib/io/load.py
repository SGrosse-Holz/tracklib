"""
Loading data from common formats into the Trajectory and TaggedSet structures
used throughout the library
"""

import os,sys

import numpy as np

from tracklib import Trajectory, TaggedSet

def csv(filename, columns=['x', 'y', 't', 'id'], tags=set(), meta_post={}, **kwargs):
    """
    Load data from a .csv file.

    This uses `!np.genfromtxt`, and all kwargs are forwarded to it. The most
    important ones are ``'delimiter'`` and ``'skip_header'``.
    
    Parameters
    ----------
    filename : string or file-like object
        the file to be read
    columns : list
        how to interpret the columns in the file. Use any of these identifiers:
        ``{'x', 'y', 'z', 'x2', 'y2', 'z2', 't', 'id', None}``, where ``'t'``
        (mandatory) is the frame number, ``'id'`` (mandatory) the trajectory
        id, and the remaining ones can be used to indicate spatial components
        of single or double-locus trajectories. Use ``None`` to indicate a
        column that should be ignored.  Columns beyond the list given here will
        be ignored in any case. Finally, the data for any str identifier not
        matching one of the above will be written to a corresponding entry in
        the trajectory's `meta` dict.
    tags : str, list of str or set of str, optional
        the tag(s) to be associated with trajectories from this file
    meta_post : dict, optional
        post-processing options for the `meta` data. Keys should be `meta`
        field names, values can be "unique" or "mean". With the former, all the
        values in the corresponding column should be the same, and only that
        value (instead of the whole array) will be written into the meta field.
        With the latter we simply take the mean of the array.

    Returns
    -------
    TaggedSet
        the loaded data set

    Examples
    --------
    This function can be used to load data from `!pandas.DataFrame` tables, if
    they conform to the format described above:

    >>> import io
    ... import pandas as pd
    ... import tracklib as tl
    ...
    ... # Set up a DataFrame containing some dummy data
    ... # Caveat to pay attention to: the order of the columns is important!
    ... df = pd.DataFrame()
    ... df['frame_no'] = [1, 2, 3]
    ... df['trajectory_id'] = [4, 4, 4]
    ... df['coord1'] = [1, 2, 3]
    ... df['coord2'] = [4, 5, 6]
    ...
    ... csv_stream = io.StringIO(df.to_csv())
    ... dataset = tl.io.load.csv(csv_stream,
    ...                          [None, 't', 'id', 'x', 'y'], # first column will be index
    ...                          delimiter=',',               # pandas' default
    ...                          skip_header=1,               # pandas prints a header line
    ...                         )
    """
    col_inds = {}
    for i, key in enumerate(columns):
        if type(key) == str:
            col_inds[key] = i

    keys = col_inds.keys()
    assert 'id' in keys
    assert 't' in keys
    
    # Get shape of trajectory and check that the given keys make sense
    if 'z' in keys:
        d = 3
        assert 'y' in keys
        assert 'x' in keys
    elif 'y' in keys:
        d = 2
        assert 'x' in keys
    elif 'x' in keys:
        d = 1
    else: # pragma: no cover
        raise ValueError("No valid coordinates found in specification: {}".format(columns))
    N = 1
    if 'x2' in keys or 'y2' in keys or 'z2' in keys:
        N = 2
        for key in keys & {'x', 'y', 'z'}:
            assert key + '2' in keys
        for key in keys & {'x2', 'y2', 'z2'}:
            assert key[:1] in keys

    # Sort into useful order
    col_ind_list = [col_inds['id'], col_inds['t']]
    for key in ['x', 'y', 'z', 'x2', 'y2', 'z2']:
        try:
            col_ind_list.append(col_inds[key])
        except KeyError:
            pass
    metakeys = []
    for key in keys - ['id', 't', 'x', 'y', 'z', 'x2', 'y2', 'z2']:
        col_ind_list.append(col_inds[key])
        metakeys.append(key)

    # Read data
    try:
        data = np.genfromtxt(filename, **kwargs)[:, col_ind_list]
    except IndexError:
        raise ValueError("'columns' contains more entries than the file has columns")
    ids = set(data[:, 0])

    # Generate individual Trajectories
    def gen():
        for myid in ids:
            mydata = data[np.where(data[:, 0] == myid)[0], :]
            myt = mydata[:, 1].astype(int)
            myt -= np.min(myt)
            trajdata = np.empty((N, np.max(myt)+1, d), dtype=float)
            trajdata[:] = np.nan
            for n in range(N):
                trajdata[n, myt, :] = mydata[:, (n*d + 2):((n+1)*d + 2)]

            meta = {}
            for i, key in enumerate(metakeys):
                meta_dat = mydata[:, 2 + N*d + i]
                if key in meta_post.keys():
                    if meta_post[key] == 'unique':
                        ms = set(meta_dat)
                        if len(ms) > 1:
                            raise RuntimeError("Data in column '{}' is not unique for trajectory with id {}".format(key, myid))
                        meta[key] = ms.pop()
                    elif meta_post[key] == 'mean':
                        meta[key] = np.mean(meta_dat)
                else:
                    meta[key] = np.empty(np.max(myt)+1)
                    meta[key][:] = np.nan
                    meta[key][myt] = mydata[:, 2 + N*d + i]

            yield (Trajectory.fromArray(trajdata, **meta), tags)

    return TaggedSet(gen())

def evalSPT(filename, tags=set()):
    """
    Load data in the format used by evalSPT

    This is a shortcut for ``csv(filename, ['x', 'y', 't', 'id'], tags)``.

    See also
    --------
    csv
    """
    return csv(filename, ['x', 'y', 't', 'id'], tags)
