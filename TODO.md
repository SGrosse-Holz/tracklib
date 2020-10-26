 + Reformat docstrings to be consistent with numpy format
 + rewrite test suite
 + write intro / overview

 + rewrite analysis.kld.Estimator as something like util.Sweeper. This class is
   really just a wrapper that takes a function to apply to a whole dataset,
   together with lists for the parameters of that function, and sweeps through the
   space of all possible parameter combinations. As such, this is a slightly
   beefed up version of multiprocessing.Pool.map().

Unresolved usage issues
=======================
 + how should element access work for Trajectory? Half of the time one would
   want [:] to return a squeezed version of the data, half the time it should
   be unsqueezed, and for the third half we would want only N to be squeezed. Also
   plays into whether there should be methods to access spatial dimensions
   individually.

   Maybe just expose the []-operator of \_data? This would make Trajectory look
   like an np.ndarray, which might be good because of familiarity. It might be
   clumsy for single-locus trajectories though, since there is no way to hide the
   N-dimension.
   
   Maybe use automatic reduction for N, T, but not d? We usually know/check how
   many loci there are, and how long the interval is will be known to the user,
   but for d polymorphism is useful. Then the only unresolved issue is that for
   1d trajectories, [:] would give (T, 1) sized arrays, which might or might not
   be desirable (see 50% rule above). This might be the best we can do.
