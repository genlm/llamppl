"""Conditional Sequential Monte Carlo (C-SMC).

C-SMC (Andrieu, Doucet & Holenstein 2010, §4.3) is a variant of SMC in
which slot 0 of the particle population is a *retained particle*: it is
extended deterministically along a fixed pre-specified trajectory and
exempt from resampling. The remaining ``n_particles - 1`` slots evolve
as in standard SMC.

The retained slot's deterministic extension is the responsibility of
the supplied :class:`Model`'s ``step`` method, which is expected to
branch on a writeable ``is_retained`` attribute that this loop sets and
re-establishes after every resample.
"""

import asyncio
import copy

import numpy as np

from ..util import logsumexp


async def csmc_standard(model, n_particles, ess_threshold=0.5):
    """Conditional sequential Monte Carlo with multinomial resampling.

    Slot 0 is the retained particle: pinned across resampling rounds and
    extended deterministically by the model whenever its ``is_retained``
    attribute is ``True``. Slots 1..``n_particles - 1`` are free.

    Args:
        model (llamppl.Model): The particle model. Must expose a writeable
            ``is_retained`` attribute; its ``step`` method is expected to
            branch on this tag and force the next move when retained.
        n_particles (int): Total number of particles, including the
            retained one. Must be ``>= 1``; ``n_particles >= 2`` is
            required for nontrivial mixing (Andrieu et al. 2010,
            Theorem 5(b)).
        ess_threshold (float): Fraction of ``n_particles`` below which
            resampling is triggered.

    Returns:
        particles (list[llamppl.Model]): The completed particles. Slot 0
            carries the retained-particle trajectory; the remaining
            ``n_particles - 1`` carry trajectories that may have inherited
            from any slot (including slot 0).

    Notes:
        Resampling is hardcoded to **multinomial**. The C-SMC invariance
        proof (Andrieu, Doucet & Holenstein 2010, Assumption 2) relies on
        unbiased multinomial draws for the free slots; stratified,
        systematic, and residual resampling introduce dependence between
        resampled indices that complicates the invariance argument and
        is not supported here. Extending C-SMC to other resampling
        schemes is possible but requires a separate derivation (see,
        e.g., Chopin & Singh 2015, *On Particle Gibbs Sampling*) and is
        out of scope for this implementation.
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    # Retainedness is a slot property, not a particle property: slot 0 is
    # always the retained slot, regardless of which trajectory any slot
    # inherits via resampling.
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    particles[0].is_retained = True
    for p in particles[1:]:
        p.is_retained = False

    # Run the model's start hook before stepping, exactly as smc_standard
    # and smc_steer do. Subclasses use start() to establish the initial
    # weight (e.g. the prefix weight of the empty sequence) and to validate
    # the target; skipping it drops that contribution from every particle.
    await asyncio.gather(*[p.start() for p in particles])

    while any(not p.done_stepping() for p in particles):
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        W = np.array([p.weight for p in particles])
        if np.all(W == -np.inf):
            # All particles dead — skip resampling to avoid NaNs.
            continue
        w_sum = logsumexp(W)
        normalized_log_weights = W - w_sum

        ess_log = -logsumexp(2 * normalized_log_weights)
        if ess_log < np.log(ess_threshold) + np.log(n_particles):
            probs = np.exp(normalized_log_weights)
            avg_weight = w_sum - np.log(n_particles)
            # Slot 0 survives unchanged and stays retained; slots 1..N-1
            # are drawn multinomially from all N weights and marked free.
            particles[0].weight = avg_weight
            particles[0].is_retained = True
            new_particles = [particles[0]]
            for _ in range(n_particles - 1):
                idx = np.random.choice(n_particles, p=probs)
                p = copy.deepcopy(particles[idx])
                p.weight = avg_weight
                p.is_retained = False
                new_particles.append(p)
            particles = new_particles

    return particles
