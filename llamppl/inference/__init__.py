"""Provides inference methods for use with LLaMPPL models.

This module currently provides the following inference methods:

* `smc_standard(model, num_particles, ess_threshold=0.5)`: Standard SMC with multinomial resampling.

* `smc_steer(model, num_beams, num_expansions)`: a without-replacement SMC algorithm that resembles beam search.

* `csmc_standard(model, n_particles, ess_threshold=0.5)`: Conditional SMC (Andrieu, Doucet & Holenstein 2010) — a $\\pi$-invariant transition kernel on trajectories, suitable as the inner loop of an MCMC or Particle Gibbs sampler.
"""

from .csmc_standard import csmc_standard
from .smc_standard import smc_standard
from .smc_steer import smc_steer
