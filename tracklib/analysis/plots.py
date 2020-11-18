"""
A module to quickly produce some overview plots
"""

import numpy as np
from matplotlib import pyplot as plt

from tracklib import Trajectory, TaggedSet
from .msd import MSD

def length_distribution(dataset, **kwargs):
    """
    Plot a histogram of trajectory lengths for the given dataset

    All keyword arguments will be forwarded to `!plt.hist()`.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the dataset to use

    Returns
    -------
    tuple
        the return value of `!plt.hist()`, i.e. a tuple ``(n, bins, patches)``

    Notes
    -----
    This should be thought of only as a quickshot way to take a look at the
    data. For more elaborate plotting, obtain the trajectory lengths as

    >>> lengths = [len(traj) for traj in dataset]

    and produce a plot to your liking.
    """
    lengths = [len(traj) for traj in dataset]
    
    if 'bins' not in kwargs.keys():
        kwargs['bins'] = 'auto'

    plt.figure()
    h = plt.hist(lengths, **kwargs)
    plt.title("Histogram of trajectory lengths")
    plt.xlabel("Length in frames")
    return h

def msd_overview(dataset, dt=1., **kwargs):
    """
    Plot individual and ensemble MSDs of the given dataset

    All keyword arguments are forwarded to `!plt.loglog()` for plotting of the
    individual trajectory MSDs

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the dataset to use
    dt : float or (float, str) tuple, optional
        the time step between two frames of the trajectory. This will simply
        rescale the horizontal axis of the plot. Optionally, give a string to
        identify the units, e.g. ``dt = (2, 'seconds')`` if your trajectories
        have a lapse time of 2 seconds. Note that you might want to set
        `!plt.xlabel()` yourself if you omit the unit string.

    Returns
    -------
    list of lines
        aggregate of the ``plt.plot()`` outputs

    Notes
    -----
    Only intended as a quick overview plot, for more customization write your
    own plotting routine using `analysis.MSD <tracklib.analysis.msd.MSD>`
    """
    if isinstance(dt, tuple):
        unit_str = dt[1]
        dt = dt[0]
    elif dt == 1:
        unit_str = "frames"
    else:
        unit_str = "{} frames".format(1/dt)

    ensembleLabel = 'ensemble mean'
    if 'label' in kwargs.keys():
        ensembleLabel = kwargs['label']
    kwargs['label'] = None

    plt.figure()
    lines = []
    for traj in dataset:
        msd = MSD(traj)
        tmsd = dt*np.arange(len(msd))
        lines.append(plt.loglog(tmsd, msd, **kwargs))
    msd = MSD(dataset)
    tmsd = dt*np.arange(len(msd))
    lines.append(plt.loglog(tmsd, msd, color='k', linewidth=2, label='ensemble mean'))
    plt.legend()

    plt.title('MSDs')
    plt.xlabel("time in " + unit_str)
    plt.ylabel("MSD")
    
    return lines

def trajectories_spatial(dataset, **kwargs):
    """
    Plot the trajectories in the given dataset.
    
    Additional keyword arguments will be forwarded to
    `Trajectory.plot_spatial`.

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the set of trajectories to plot

    Keyword Arguments
    -----------------
    colordict : dict
        determines which tag is colored in which color. This should be a dict
        whose keys are the tags in the dataset, while the entries are anything
        recognized by the `!'color'` kwarg of `!plt.plot()`. If omitted, will
        cycle through the default color cycle.
    fallback_color : str, or RGB tuple
        the color to use for anything not appearing in `!'colordict'`
        
    Returns
    -------
    list of lines
        the aggregated output of `!plt.plot()`

    Notes
    -----
    Each tag will be associated with one color, and trajectories will be
    colored by one of the tags they're associated with.  There is no way to
    determine which one.
    """
    flags = {'fallback_used' : False}

    try:
        fallback_color = kwargs['fallback_color']
        del kwargs['fallback_color']
    except KeyError:
        fallback_color = '#aaaaaa'

    # Get coloring
    try:
        colordict = kwargs['colordict']
        del kwargs['colordict']
    except KeyError:
        try:
            if isinstance(kwargs['color'], list):
                colors = kwargs['color']
            else:
                colors = [kwargs['color']]
        except KeyError:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        colordict = {tag : colors[i%len(colors)] for i, tag in enumerate(dataset.tagset())}
    
    # Plotting
    plt.figure()
    lines = []
    for traj, trajtags in dataset(giveTags=True):
        try:
            kwargs['color'] = colordict[(trajtags & colordict.keys()).pop()]
        except KeyError: # set().pop() raises a KeyError
            kwargs['color'] = fallback_color
            flags['fallback_used'] = True
        lines += traj.plot_spatial(**kwargs)

    # Delete all non-plotting kwargs
    for key in {'dims'}:
        try:
            del kwargs[key]
        except KeyError:
            pass

    # Need some faking for the legend
    x0 = sum(plt.xlim())/2
    y0 = sum(plt.ylim())/2
    for tag in colordict.keys():
        kwargs['color'] = colordict[tag]
        kwargs['label'] = tag
        if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                kwargs['linestyle'] = kwargs['linestyle'][0]
        plt.plot(x0, y0, **kwargs)

    if flags['fallback_used']:
        kwargs['color'] = fallback_color
        kwargs['label'] = '<other tags>'
        if 'linestyle' in kwargs.keys() and isinstance(kwargs['linestyle'], list):
                kwargs['linestyle'] = kwargs['linestyle'][0]
        plt.plot(x0, y0, **kwargs)
        

    # Control appearance
    if len(colordict) > 1 or flags['fallback_used']:
        plt.legend()
        plt.title("Trajectories in real space")
    else:
        plt.title("Trajectories for tag {}".format(list(colordict.keys())[0]))

    # Done
    return lines

def distance_distribution(dataset, **kwargs):
    """
    Draw a histogram of distances.
    
    For two-locus trajectories, this is the absolute distance between the loci,
    for single locus trajectories it is simply the absolute value of the
    trajectory.

    All keyword arguments will be forwarded to `!plt.hist()`

    Parameters
    ----------
    dataset : `TaggedSet` of `Trajectory`
        the trajectories to use

    Returns
    -------
    tuple
        the output of `!plt.hist()`, i.e. a tuple ``(n, bins, patches)``.

    Notes
    -----
    This is intended for gaining a quick overview. For more elaborate tasks,
    obtain the distances as

    >>> dists = np.concatenate([traj[:].flatten() for traj in dataset.process(<preproc>)])

    where ``<preproc>`` is the preprocessing function appropriate for your
    dataset.  For a two-locus trajectory, this would presumably be

    >>> preproc = lambda traj : traj.relative().abs()
    """
    N = dataset.map_unique(lambda traj : traj.N)
    if N == 2:
        preproc = lambda traj : traj.relative().abs()
    elif N == 1:
        preproc = lambda traj : traj.abs()
    else:
        raise RuntimeError("Dataset has neither homogeneously N = 1 nor N = 2")

    data = np.concatenate([traj[:].flatten() for traj in dataset.process(preproc)])

    if 'bins' not in kwargs.keys():
        kwargs['bins'] = 'auto'

    plt.figure()
    h = plt.hist(data, **kwargs)
    plt.title('Distance histogram')
    return h
