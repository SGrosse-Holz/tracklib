"""
Loading data from common formats into the Trajectory and TaggedSet structures
used throughout the library
"""

import os,sys

import numpy as np

from tracklib import Trajectory, TaggedSet

def csv(filename, columns=['x', 'y', 't', 'id'], tags=None, meta_post={}, **kwargs):
    """
    Load data from a .csv file.

    This uses ``np.genfromtxt``, and all kwargs are forwarded to it. By
    default, we assume the delimiter ``','`` and utf8 encoding for string data,
    but these can of course be changed. Refer to ``numpy.genfromtxt``.
    
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
    This function can be used to load data from ``pandas.DataFrame`` tables, if
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
        if type(key) == str: # make sure to exclude None's
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

    # data_keys = ['x', 'x2', 'y', 'y2', ...]
    data_keys = sorted(keys & {'x', 'y', 'z', 'x2', 'y2', 'z2'})
    N = 1
    if any('2' in key for key in keys):
        N = 2
        for key in data_keys:
            assert key[0] in data_keys
            assert key[0]+'2' in data_keys

    # ['id', 't', {data}, {meta}]
    # this is mostly to keep track of which keys exist
    sorted_keys = ['id', 't', data_keys,
                   keys - {'id', 't', *data_keys}]

    # Read data
    gft_kwargs = dict(delimiter=',', dtype=None, encoding='utf8')
    gft_kwargs.update(kwargs)
    data = np.genfromtxt(filename, **gft_kwargs)

    # This feels suboptimal... maybe there's a better way?
    data_cols = [col_inds[key] for key in sorted_keys[2]]
    meta_cols = [col_inds[key] for key in sorted_keys[3]]
    try:
        # sorted_data = [id-array, t-array, data-array, list of meta-arrays]
        sorted_data = [
            np.array([line[col_inds['id']] for line in data]),
            np.array([line[col_inds['t' ]] for line in data]).astype(int),
            np.array([[line[col] for col in data_cols]
                      for line in data
                     ]).astype(float), # shape: (-1, N*d), sorted x, x2, y, ...
            [np.array([line[col] for line in data]) for col in meta_cols],
        ]
    except IndexError:
        raise ValueError("Too many columns for file. Did you use the right delimiter?")
    del data
    ids = set(sorted_data[0])

    # Assemble data set
    out = TaggedSet()
    for myid in ids:
        ind = sorted_data[0] == myid
        mydata = np.moveaxis(sorted_data[2][ind].reshape((-1, d, N)), 2, 0)
        myt = sorted_data[1][ind]
        myt -= np.min(myt)

        trajdata = np.empty((N, np.max(myt)+1, d), dtype=float)
        trajdata[:] = np.nan
        trajdata[:, myt, :] = mydata

        meta = {}
        for i, key in enumerate(sorted_keys[3]):
            mymeta = sorted_data[3][i][ind]
            if key in meta_post:
                post = meta_post[key]
                if post == 'unique':
                    ms = set(mymeta)
                    if len(ms) > 1:
                        raise RuntimeError("Data in column '{}' is not unique for trajectory with id {}".format(key, myid))
                    meta[key] = ms.pop()
                elif post == 'mean':
                    meta[key] = np.mean(mymeta)
                elif post == 'nanmean':
                    meta[key] = np.nanmean(mymeta.astype(float))
                else: # pragma: no cover
                    raise ValueError(f"invalid meta post-proc: {post}")
            else: # assume that we have floats and fill with nan's
                meta[key] = np.empty(np.max(myt)+1, dtype=float)
                meta[key][:] = np.nan
                meta[key][myt] = mymeta

        out.add(Trajectory.fromArray(trajdata, **meta), tags)

    return out

def evalSPT(filename, tags=set()):
    """
    Load data in the format used by evalSPT

    This is a shortcut for ``csv(filename, ['x', 'y', 't', 'id'], tags,
    delimiter='\t')``.

    See also
    --------
    csv
    """
    return csv(filename, ['x', 'y', 't', 'id'], tags, delimiter='\t')
