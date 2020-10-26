import numpy as np
from matplotlib import pyplot as plt

from tracklib import Trajectory, TaggedSet
from .msd import MSD

def length_distribution(dataset, **kwargs):
    """
    Plot a histogram of trajectory lengths for the given dataset

    Input
    -----
    dataset : TaggedSet (possibly with some selection set)
        the dataset to use
    All other keyword arguments will be forwarded to plt.hist().

    Output
    ------
    The return value of plt.hist(), i.e. a tuple (n, bins, patches)

    Notes
    -----
    This should be thought of only as a quickshot way to take a look at the
    data. For more elaborate plotting, obtain the trajectory lengths as
        lengths = [len(traj) for traj in dataset]
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

def msd_overview(dataset, **kwargs):
    """
    Plot individual and ensemble MSDs of the given dataset

    Input
    -----
    dataset : TaggedSet (possibly with some selection set)
        the dataset to use
    All further keyword arguments are forwarded to plt.loglog for plotting of
    the individual trajectory MSDs

    Output
    ------
    Aggregate of the plt.plot outputs, i.e. a list of lines

    Notes
    -----
    Only intended as a quick overview plot, for more customization write your
    own plotting routine using analysis.MSD(dataset) and Trajectory.msd().
    """
    ensembleLabel = 'ensemble mean'
    if 'label' in kwargs.keys():
        ensembleLabel = kwargs['label']
    kwargs['label'] = None

    plt.figure()
    lines = []
    for traj in dataset:
        msd = traj.msd()
        tmsd = np.arange(len(msd))
        lines.append(plt.loglog(tmsd, msd, **kwargs))
    msd = MSD(dataset)
    tmsd = np.arange(len(msd))
    lines.append(plt.loglog(tmsd, msd, color='k', linewidth=2, label='ensemble mean'))
    plt.legend()

    plt.title('MSDs')
    plt.xlabel("time in frames")
    plt.ylabel("MSD")
    
    return lines

def trajectories_spatial(dataset, **kwargs):
    """
    Plot the trajectories in the given dataset. The preset selection can be
    overridden manually to ensure proper coloring (see Notes).

    Input
    -----
    dataset : TaggedSet
        the set of trajectories to plot
    colordict : dict
        determines which tag is colored in which color. This should be a dict
        whose keys are the tags in the dataset, while the entries are anything
        recognized by the 'color' kwarg of plt.plot. 
        default: cycle through the default color cycle
    All further keyword arguments will be forwarded to
    Trajectory.plot_spatial()

    Output
    ------
    A list of lines, as returned by plt.plot()

    Notes
    -----
    Each tag will be associated with one color, and trajectories will be
    colored by one of the tags they're associated with.  There is no way to
    determine which one.

    Similarly to the other analysis.plot_* and analysis.hist_* functions, this
    is mostly intended for use in a quick overview. It does provide some more
    functionality though, in the hope that the user will not see a necessity to
    start plotting trajectories themselves.
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
        lines.append(traj.plot_spatial(**kwargs))

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
    Draw a histogram of distances. For two-locus trajectories, this is the
    absolute distance between the loci, for single locus trajectories it is
    simply the absolute value of the trajectory.

    Input
    -----
    dataset : TaggedSet (possibly with some selection set)
        the trajectories to use
    All keyword arguments will be forwarded to plt.hist()

    Output
    ------
    The output of plt.hist(), i.e. a tuple (n, bins, patches).

    Notes
    -----
    This is intended for gaining a quick overview. For more elaborate tasks,
    obtain the distances as
        dists = np.concatenate([traj[:].flatten() for traj in dataset.process(<preproc>)])
    where <preproc> is the preprocessing function appropriate for your dataset.
    For a two-locus trajectory, this would presumably be
        preproc = lambda traj : traj.relative().abs()
    """
    if dataset.getHom('N') == 2:
        preproc = lambda traj : traj.relative().abs()
    elif dataset.getHom('N') == 1:
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
