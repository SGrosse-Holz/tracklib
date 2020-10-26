# This module contains utility functions for loading data from common formats
# into the TaggedSet structure used by this library.

import os,sys

import numpy as np

from tracklib import Trajectory, TaggedSet

def evalSPT(filename, tags):
    """
    Load a file in the format used by evalSPT. See below for precise
    specification.

    Input
    -----
    filename : string
        the file to be read
    tags : str, list of str or set of str
        the tag(s) to be associated with trajectories from this file

    Output
    ------
    A TaggedSet containing the loaded trajectories

    Format specification
    --------------------
    The file handed to this function should be a text file, delimited by tabs,
    and without header line. The columns are interpreted as [x, y, t, id, ...],
    where ... indicates that additional columns in the data will be ignored.
    The time is assumed to be given in frames, i.e. should be an integer. The
    id column should identify trajectories, i.e. all the data belonging to one
    trajectory should be consecutive and have the same id. ids do not have to
    start at 1, or be consecutive integers.
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
    return TaggedSet.generate(data2gen(data, tags))

# def twoLocusCSV(filename, tags):
#     def data2gen(data, tags):
#         for mydata in np.split(data, np.where(np.diff(data[:, 1]) != 0)[0]+1):
#             t = mydata[:, 2gt
