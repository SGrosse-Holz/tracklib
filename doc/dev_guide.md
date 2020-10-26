tracklib developer's guide
==========================
--------------------------
This document is intended to explain and document style and implementation
choices made in the library. Before deciding that stupid design choices were
made, consult the corresponding section here. (Though there might of course be
explicitly made decisions that nevertheless turn out to be stupid).

This document is organized like the library, with individual
modules/sub-modules/functions each having their own section. For completeness,
everything in the library should have a section here, even if that section is
empty. I expect this document to be navigated by search anyways, so many empty
sections are not much of a nuisance.

trajectory
==========

Trajectory
----------
The main purpose of this class is to provide a common type for trajectories of
all kinds, with differing spatial dimension or number of loci. The main
functionality of the base class are the modifier functions (abs, diff,
relative, ...) that can be used to perform standard processing tasks.

Any specific trajectory will be of a subclass of Trajectory, such that
functionality that is specific to say two-locus trajectories in three
dimensions can be implemented. It is unlikely that this will find use, but the
specializations depending on either number of loci or spatial dimensions prove
to be useful.

The plotting functions might undergo review at some point.

We chose to base this library on constant framerate trajectories, because some
analysis methods (e.g. MSD) do not work as well with freely spaced
trajectories.

Currently the paradigm concerning the actual data is to try and avoid ever
having to directly access it from outside the class. This leads to
complications (see []-operator), so maybe we should move away from it? It is
mostly a style question... (TODO)

### fromArray()

### N, T, d

### \_\_len\_\_()

### \_\_getitem\_\_()
When providing element access, do we return a three dimensional array, or do we
squeeze it (remove single entry dimensions)? There are arguments for both: on
the one hand, keeping dimensionality definite means that we know exactly what
to expect from the []-operator. On the other hand, it is annoying to have a
bunch of single entry dimensions around; consider accessing one time point of a
single locus trajectory: non-squeezed this would give a (1, 1, d) array, so
we'd have to write something like `traj[t][0, 0, :]`, which is ugly.

After some deliberation, the best solution seems to be to squeeze the N and T
dimensions, but leave d as it is. When processing trajectories we usually will
know (or check) N, and what happens to T is determined by the user-provided
slice. We do for the most part want to write analysis methods that are agnostic
to d, so polymorphism seems useful here. The only problem with this solution is
that for some "naturally 1d" trajectories (such as absolute distances between
loci) it might be annoying to carry that extra dimension. I deem the
polymorphism argument to be stronger though.

### Modifiers: abs, diff, relative, dims
Note that fromArray() already copies the array passed to it, so there's no need
to do that explicity. Otherwise, the key for these functions is that they're
chainable:
```
traj_processd = traj.relative().abs().diff()
```
(or something like that).

### #yield_dims()
More of a sketch of an idea. Would it be useful to have something like this?

### plot_vstime()

### plot_spatial()
What exactly to do here depends on N and d, but independently. This is thus
implemented in Trajectory\_?d, calling the 'raw' plotting function in
Trajectory\_?N. 

N12Error
--------
This special exception might be useful if there were more use cases for it,
which might happen as the library grows. Right now it's a bit pointless.

taggedset
=========

TaggedSet
---------
The idea here is a many-to-many dict: have a bunch of data that can belong to
one or more subsets of the whole data set. Of course one subset will usually
also contain more than one datum, thus many-to-many. It is very natural to then
select some of these subsets for processing. For practically all purposes the
class will then behave as if it contained only those data in the current
selection. The idea for usage is thus: load all data whatsoever into one
TaggedSet object, then work with subsets of this.

Interfaces: this class actually does implement the Sequence interface,
implicitly. We do not implement the Set/MutableSet interface though, because
dealing with copy operations would be tricky: we want `__iter__` to return just
the data, no tags, but the Set functions assume that iterating through the Set
gives full information. Apart from that the functionality added by
Set/MutableSet (comparisons and set operations) is not particularly relevant to
this class, so we resort to implementing just the &= operator by hand, because
it's useful. Note that `mergein()` has slightly more functionality though.

### makeTagsSet()

### add()

### \_\_iter\_\_(), \_\_call\_\_()

### \_\_len\_\_()

### \_\_getitem\_\_()

### makeSelection(), refineSelection()

### saveSelection(), restoreSelection()
Need the copying to prevent accidentally giving away access to
`self._selection`.

### copySelection()
Should be mostly unnecessary (see the paradigm about using just one data set
above), but who knows.

### mergein(), \_\_iand\_\_()

### addTags(), tagset()

### filter(), process()
