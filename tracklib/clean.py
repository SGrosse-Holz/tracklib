# A small module to do some data cleaning, e.g. split trajectories at
# mislinkages

from copy import deepcopy
import numpy as np

from .taggedlist import TaggedList
from .trajectory import Trajectory

def split_trajectory_at_big_steps(traj, threshold):
    """
    Check a trajectory for steps bigger than the given threshold.
    If there are any, split the trajectory at these steps.
    Exception: two consecutive big steps such that the overall
        displacement is smaller than the threshold are single
        misconnections in otherwise fine trajectories. In this
        case we will simply erase the misconnected point.

    Input
    -----
    traj : tracklib.Trajectory, N=1
        the trajectory to investigate
    threshold : float
        the maximum allowed frame to frame displacement
    
    Output
    ------
    A set of new trajectories
    """
    if traj.N != 1:
        raise ValueError("Cannot detect mislinkages in trajectories with N > 2")

    difftraj = traj.diff().abs()
    step_isBig = np.where(difftraj[:][0, :, 0] <= threshold, 0, 1.)
    step_isBig[np.where(np.isnan(difftraj[:][0, :, 0]))] = np.nan
    
    step_isBig = np.pad(step_isBig, 1, constant_values=np.nan) # Now step_isBig[i] describes traj[i] - traj[i-1]
    inds_bigsteps = np.where(step_isBig == 1)[0]
    
    # Check for single misconnections
    for ind in inds_bigsteps:
        if ((step_isBig[(ind-1):(ind+2)] == 1).tolist() == [False, True, True]
            and step_isBig[ind+2] != 1):
            traj[:][:, ind, :] = np.nan
            step_isBig[ind:(ind+2)] = np.nan # Now the steps don't exist anymore
    
    # If now everything's fine, that's cool
    if not np.any(step_isBig == 1):
        return {traj}
    
    # Split at remaining big steps
    inds_bigsteps = np.where(step_isBig == 1)[0]
    new_trajs = set()
    old_ind = 0
    for ind in inds_bigsteps:
        new_trajs.add(Trajectory.fromArray(traj[old_ind:ind]))
        old_ind = ind
    new_trajs.add(Trajectory.fromArray(traj[old_ind:]))
    del traj
    
    return {traj for traj in new_trajs if len(traj) > 1}

def split_dataset_at_big_steps(data, threshold):
    """
    Apply split_trajectory_at_big_steps() to all trajectories in the dataset
    and return the results in a new dataset.

    Input
    -----
    data : TaggedList
        the dataset with possibly too big steps
    threshold : float
        the maximum allowed step size

    Output
    ------
    A new data set with the split trajectories
    """
    def gen():
        for traj, tags in data(giveTags=True):
            for part_traj in split_trajectory_at_big_steps(traj, threshold):
                yield (deepcopy(part_traj), deepcopy(tags))
    
    return TaggedList.generate(gen())
