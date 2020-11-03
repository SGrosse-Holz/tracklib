"""
Loading data from common formats into the Trajectory and TaggedSet structures
used throughout the library
"""

import os,sys

import numpy as np

from tracklib import Trajectory, TaggedSet

def evalSPT(filename, tags=set()):
    """
    Load a file in the format used by evalSPT.
    
    Parameters
    ----------
    filename : string
        the file to be read
    tags : str, list of str or set of str, optional
        the tag(s) to be associated with trajectories from this file

    Returns
    -------
    TaggedSet
        the loaded data set

    Notes
    -----
    **Format specification:** the file handed to this function should be a text
    file, delimited by tabs, and without header line. The columns are
    interpreted as ``[x, y, t, id, ...]``, where ``...`` indicates that
    additional columns in the data will be ignored.  The time is assumed to be
    given in frames, i.e. should be an integer, but can start at an arbitrary
    number.. The `!id` column should identify trajectories, i.e. all the data
    belonging to one trajectory should be consecutive and have the same `!id`.
    `!id` do not have to start at ``1``, or be consecutive integers.
    """
    def data2gen(data, tags):
        for mydata in np.split(data, np.where(np.diff(data[:, 3]) != 0)[0]+1):
            t = mydata[:, 2].astype(int)
            t -= np.min(t)
            traj = np.empty((np.max(t)+1, 2))
            traj[:, :] = np.nan
            traj[t, :] = mydata[:, :2]
            yield (Trajectory.fromArray(traj), tags)

    try:
        data = np.genfromtxt(filename, delimiter='\t')
    except Exception as err:
        print("Error loading file '{}'".format(filename))
        raise err
    return TaggedSet(data2gen(data, tags))

# def twoLocusCSV(filename, tags):
#     def data2gen(data, tags):
#         for mydata in np.split(data, np.where(np.diff(data[:, 1]) != 0)[0]+1):
#             t = mydata[:, 2gt
