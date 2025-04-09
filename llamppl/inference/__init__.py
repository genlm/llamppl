"""Provides inference methods for use with LLaMPPL models.

This module currently provides the following inference methods:

* `smc_standard(model, num_particles, ess_threshold=0.5)`: Standard SMC with multinomial resampling.

* `smc_steer(model, num_beams, num_expansions)`: a without-replacement SMC algorithm that resembles beam search.
"""

from .smc_standard import smc_standard
from .smc_steer import smc_steer
