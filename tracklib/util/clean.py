"""
Everything to do with cleaning up experimental data
"""

from copy import deepcopy
import numpy as np

from tracklib import Trajectory, TaggedSet

def split_trajectory_at_big_steps(traj, threshold):
    """
    Removes suspected mislinkages.

    Exception: two consecutive big steps such that the overall displacement is
    smaller than the threshold are single misconnections in otherwise fine
    trajectories. In this case we will simply remove the misconnected point.

    Parameters
    ----------
    traj : Trajectory
        the trajectory to investigate
    threshold : float
        the maximum allowed frame to frame displacement
    
    Returns
    -------
    set of tracklib.Trajectory

    See also
    --------
    split_dataset_at_big_steps

    Notes
    -----
    As of now, this only works on trajectories with ``N=1``.

    This really just checks for frame-to-frame connections exceeding the
    threshold. So if there are missing frames in a trajectory, the step across
    those missing data will not be considered.
    """
    if traj.N != 1: # pragma: no cover
        raise ValueError("Cannot detect mislinkages in trajectories with N > 1")

    old_npwarns = np.seterr(invalid='ignore') # yes, we'll do np.nan <= threshold. Gives False.

    difftraj = traj.diff().abs()
    step_isBig = np.where(difftraj[:][:, 0] <= threshold, 0, 1.)
    step_isBig[np.where(np.isnan(difftraj[:][:, 0]))[0]] = np.nan

    np.seterr(**old_npwarns)
    
    step_isBig = np.pad(step_isBig, 1, constant_values=np.nan) # Now step_isBig[i] describes traj[i] - traj[i-1]
    inds_bigsteps = np.where(step_isBig == 1)[0]

    # Check for single misconnections
    for ind in inds_bigsteps:
        if ((step_isBig[(ind-1):(ind+2)] == 1).tolist() == [False, True, True]
            and step_isBig[ind+2] != 1):
            traj.data[:, ind, :] = np.nan
            step_isBig[ind:(ind+2)] = np.nan # Now the steps don't exist anymore
    
    # If now everything's fine, that's cool
    if not np.any(step_isBig == 1):
        return {traj}
    
    # Split at remaining big steps
    inds_bigsteps = np.where(step_isBig == 1)[0]
    new_trajs = set()
    old_ind = 0
    for ind in inds_bigsteps:
        new_trajs.add(Trajectory.fromArray(traj.data[:, old_ind:ind, :]))
        old_ind = ind
    new_trajs.add(Trajectory.fromArray(traj.data[:, old_ind:, :]))
    del traj
    
    return {traj for traj in new_trajs if len(traj) > 1}

def split_dataset_at_big_steps(data, threshold):
    """
    Apply `split_trajectory_at_big_steps` to a whole data set

    Parameters
    ----------
    data : TaggedSet
        the dataset
    threshold : float
        the maximum allowed step size

    Returns
    -------
    TaggedSet
        a new data set with the split trajectories

    See also
    --------
    split_trajectory_at_big_steps
    """
    def gen():
        for traj, tags in data(giveTags=True):
            for part_traj in split_trajectory_at_big_steps(traj, threshold):
                yield (deepcopy(part_traj), deepcopy(tags))
    
    return TaggedSet(gen())
