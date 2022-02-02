"""
Bayesian MSD fitting

Any *valid* MSD function MSD(Δt) defines a stationary Gaussian process with the
appropriate correlation structure. Therefore, instead of fitting the graph of
"empirically" calculated MSDs, we can perform Bayesian inference of parametric
MSD curves, using the Gaussian process likelihood function. Specifically, we
discriminate two cases, depending on what exactly is stationary:

 + the trajectories themselves might be sampled from a stationary process (e.g.
   distance of two points on a polymer). In terms of the MSD, the decisive
   criterion is ``MSD(inf) < inf``. In this case, it is straightforward to
   prove the following relation between the MSD μ and autocovariance γ of the
   process:

    .. code-block:: text

        μ(k) = 2*( γ(0) - γ(k) )

   Thus, the full autocovariance function can be obtained from the MSD and the
   steady state covariance ``γ(0)``. For decaying correlations (``γ(k) --> 0 as
   k --> ∞``) we furthermore see that ``2*γ(0) = μ(∞)`` is the asymptotic value
   of the MSD. Finally, this allows us to calculate the covariance matrix of
   the process as

    .. code-block:: text

        C_ij := <x_i*x_j>
              = γ(|i-j|)
              = γ(0) - 1/2*μ(|i-j|)

   We call this case a steady state of order 0.
 + in many cases (e.g. sampling a diffusing particle's position) the
   trajectories itself will not be stationary, but the increment process is. In
   this case the Gaussian process of interest is the one generating the
   increments of the trajectory, whose autocorrelation is the second derivative
   of the MSD:

    .. code-block:: text

        γ(k) = 1/2 * (d/dk)^2 μ(k)

   where derivatives should be understood in a weak ("distributional") sense.
   More straightforwardly, the correlation matrix of the increments is given by

    .. code-block:: text

       C_ij := <(x_{i+1} - x_i)(x_{j+1}-x_j)>
             = 1/2 * ( μ(t_{i+1} - t_j) + μ(t_i - t_{j+1})
                      -μ(t_{i+1} - t_{j+1}) - μ(t_i - t_j) )

   where by definition we let ``μ(-k) = μ(k)``. In this case, we talk about a
   steady state of order 1.

In any case, the covariance matrix ``C`` (potentially together with a mean /
drift term for steady state order 0 / 1 respectively) defines a Gaussian
process, which lets us assign a likelihood to the generating MSD. Via this
construction, we can perform rigorous Bayesian analysis, cast in the familiar
language of MSDs.

This module provides a base class for performing such inferences / fits, namely
`Fit`. We also provide a few example implementations of fitting schemes in the
`lib` submodule. Finally, the `Profiler` allows to explore the posterior once a
point estimate has been found, by tracing out either conditional posterior or
profile posterior curves in each parameter direction. Note that you can also
just sample the posterior by MCMC.

Examples
--------
TODO: once `!lib` is implemented, add some examples

An example implementation of an MSD sampler for sampling the posterior, given
data from a `Profiler` instance.
TODO: test this

>>> import tracklib as tl
... import numpy as np
...
... class PosteriorSampler(tl.util.mcmc.Sampler):
...     def __init__(self, profiler, mci=None, stepsize=0.1):
...         \"\"\"
...         `mci` is the output of a previous run of the profiler (if there was
...         one). This is used to set the stepsize of the MCMC according to the
...         guesses on the posterior shape from those results.
...         \"\"\"
...         self.profiler = profiler
... 
...         fix_values = self.profiler.fit.fix_values
...         self.min_target_from_fit = self.profiler.fit.get_min_target(fix_values=fix_values)
... 
...         # Figure out, which parameters are actually independent
...         ifix = [i for i, _ in fix_values]
...         i_independent = np.array([i for i in range(len(self.profiler.fit.bounds)) if i not in ifix])
... 
...         # Set stepsizes (if not given by mci)
...         self.stepsizes = np.array(len(i_independent)*[stepsize])
...         if mci is not None:
...             for iparam in range(len(mci)):
...                 if not np.any(np.isnan(mci[iparam])) and iparam in i_independent:
...                     m, *ci = mci[iparam]
...                     step = np.min(np.abs(ci - m))
...                     self.stepsizes[i_independent == iparam] = step
... 
...     def get_initial_values(self):
...         p0 = self.profiler.point_estimate['params'].copy()
...         ifix = np.array([i for i, _ in self.profiler.fit.fix_values])
...         p0[ifix] = np.nan
...         return p0[~np.isnan(p0)]
... 
...     # Implement MCMC interface
...     def logL(self, params):
...         return -self.min_target_from_fit(params)
... 
...     def propose_update(self, current_params):
...         i_update = np.random.choice(len(current_params))
...         new_params = current_params.copy()
...         step_dist = stats.norm(scale=self.stepsizes[i_update])
...         step = step_dist.rvs()
...         new_params[i_update] += step
... 
...         return new_params, 1, 1 # don't have to care about fwd/bwd probabilities, bc Gaussian is symmetric
...         # return new_params, step_dist.logpdf(step), step_dist.logpdf(-step) # "more correct"

See also
--------
tracklib.util.mcmc, tracklib.models.rouse
"""
from .core import *
from . import lib
