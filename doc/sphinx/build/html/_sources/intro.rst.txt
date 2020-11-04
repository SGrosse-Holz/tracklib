Quickstart
==========

This library is intended for downstream analysis of single particle tracking
data. Downstream here means that we are not concerned with particle detection
or the linking problem, instead our starting point are the linked particle
trajectories. Consequently, the core of this library are the classes
`Trajectory`, representing a single trajectory, and `TaggedSet`, which provides
a useful way of organizing multiple (sets of) trajectories. Surrounding these
core functionalities, we then have several tools for processing, modelling, and
analysis, as illustrated in the figure below.

.. image:: scheme.png

This document will introduce the basic ideas behind the core functionalities.
For further details, see the :doc:`documentation <tracklib>`.

Trajectory
----------

The `Trajectory` is the "atom" of this library. Besides the actual `data`, it
contains a dict for `meta` data. Users can store all sorts of information about
the trajectory here (e.g. the time step, some auxiliary trajectory like a
reporter lighting up, analysis results, any or all of those). The library will
also make use of this occasionally, for example to store analysis results like
the MSD of that trajectory.

`Trajectory` objects themselves have relatively limited capabilities, mostly
just some functions for simple geometric processing that we call "Modifiers".
The following example showcases how these can be used and combined:

>>> import numpy as np
... import tracklib as tl
... 
... # Generate some random trajectory with two loci in 3d
... traj = tl.Trajectory.fromArray(np.random.normal(size=(2, 10, 3)))
... traj.meta['info'] = 'some meta data'
... 
... # "Rephrase" this a little bit:
... rel = traj.relative()                    # trajectory of the vector between the two loci
... abs = traj.relative().abs()              # trajectory of absolute distance
... steps = traj.diff(dt=1).abs()            # trajectory of stepsizes for both loci individually
... rel_steps = traj.relative().diff(dt=1)   # trajectory of steps in the relative trajectory
... plane = traj.dims([0, 1])                # restrict to only the first two dimensions

TaggedSet
---------

When working with tracking data, we often have a host of different "kinds" of
trajectories (different experimental condition, different tracked objects, or
simply trajectories that fall into different classes based on some analysis).
Depending on the exact analysis we are doing, different aggregation schemes
might make sense (e.g. run some analysis on all trajectories from a certain
experimental condition, or on all trajectories with a frame rate of 10 seconds,
etc). It thus seems useful to have a data structure that allows running
analyses on arbitrary subsets of data.

The centerpiece of the `TaggedSet` is its selection mechanism. Consider the
following minimal example:

>>> import tracklib as tl
... data = tl.TaggedSet()
... data.add(1, tags='a')
... data.add(2, tags='b')
... data.add(3, tags=['a', 'b'])
... 
... print(len(data)) # prints: 3
... for i in data:
...     print(i)     # prints: 1 2 3
... 
... data.makeSelection(tags='a') # select all entries tagged with 'a'
... print(len(data)) # prints: 2
... for i in data:
...     print(i)     # prints: 1 3

Note how, once we make a selection, the whole data set simply behaves as if it
contained only those data. In addition to making selections by single tags, as
shown in the example above, we can also select by combinations of tags, or even
by properties of the data. Continuing from above:

>>> data.makeSelection(tags=['a', 'b'], logic=all)
... for i in data:
...     print(i)     # prints: 3     (everything carrying all the mentioned tags)
... 
... data.makeSelection(tags=['a', 'b'], logic=any)
... for i in data:
...     print(i)     # prints: 1 2 3 (everything carrying any of the mentioned tags)
... 
... data.makeSelection(selector = lambda i, tags : i >= 2)
... for i in data:
...     print(i)     # prints: 2 3

Refer to the documentation on `makeSelection` for more details.

Finally, the `TaggedSet` class provides some means of applying functions to all
of the data (in the current selection). Note that since a `TaggedSet` works as
an iterator, the built-in `!map()` function will work for many cases:

>>> data.makeSelection()
... times2 = lambda x : 2*x
... 
... doubles = list(map(times2, data)) # a list: [2, 4, 6]
... double_set = data.process(times2) # a new TaggedSet with the corresponding entries (and tags!)
... data.filter(times2)               # same as process(), but in-place
... dtype = data.map_unique(type)     # shortcut for functions that should return the same value on all data.

At the beginning of this example, we call `makeSelection` without arguments to
reset the selection to the whole data set.
