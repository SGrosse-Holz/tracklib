Some notes on this (sub-)module
-------------------------------

### Terminology
Trajectory vs. trace:
 - by 'trajectory' we mean a whole trajectory, possibly containing multiple loci,
   dimensions, meta data, etc. These are represented by tracklib.Trajectory
 - by 'trace' we mean a single time trace, such as one spatial component of a
   trajectory. Traces are represented as np.arrays with shape (T,).
 - this submodule works on the trace level.

Loop sequence vs. loop trace:
 - a 'loop trace' is a trace with boolean entries, indicting presence/absence
   of a loop. Loop traces are represented as np.arrays(dtype=bool) with shape (T,)
 - a 'loop sequence' is an abstract representation of a loop trace. It stores
   only some time intervals and the loop status during these intervals. Loop
   sequences are represented by util.LoopSequence

### Config
Some functions take a dict named config as argument. The expected entries in
that dict are compatible throughout this submodule, i.e. we can define one big
config dict and just pass it to everything. The full specification is as
follows:

 - 'MCMC iterations' : int
	the number of MCMC steps to run
	default: 100
 - 'MCMC burn-in' : int
	the number of MCMC steps to discard from the beginning of the
	sampling
	default: 50
 - 'MCMC best only' : bool
	whether the MCMC sampler should return the whole chain or only the best
	fit
	default: False, i.e. return whole chain
 - 'MCMC log every' : int
	print status information every .. frames.
	default: -1, i.e. no logging
 - 'MCMC show progress' : bool
	whether to show a progress bar
	default: False
 - 'MCMC stepsize' : float
	the stepsize to use in the MCMC scheme. Precise meaning depends on the
	sampler used.
	default: 0.1
 - 'unknown params' : list of str
	which parameters of the rouse.Model are unknown (i.e. should be
	sampled)
	default: ['D', 'k']
 - 'numIntervals' : int
	the number of loop intervals to use for the sampling
	default: 10
 - 'pLoop_method' : 'sequence' or 'trace'
	whether to use loop traces or loop sequences for calculating the
	looping probability.
	default: 'sequence'

See also the default values defined in util.py
