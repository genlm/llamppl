"""Tests for :func:`llamppl.inference.csmc_standard`.

These tests exercise the C-SMC contract: slot 0 of the particle
population is retained across resampling rounds and the ``is_retained``
tag is a slot property re-established after every resample.
"""

import asyncio

import pytest

from llamppl.inference import csmc_standard


class _MockModel:
    """A duck-typed mock of the subset of ``llamppl.Model`` that
    :func:`csmc_standard` actually touches: ``untwist``, ``step``,
    ``done_stepping``, the writeable ``weight`` attribute, and the
    ``is_retained`` tag the loop manipulates.

    ``step`` records whether the slot is retained at the time of the
    call and scores the weight differently for retained vs. free
    particles, so that weights diverge enough for the ESS check to fire
    resampling on every round.
    """

    def __init__(self, n_steps=3, score_retained=10.0, score_free=0.0):
        self.n_steps = n_steps
        self.step_idx = 0
        self.weight = 0.0
        self.slot_history = []  # list of "retained"/"free" per call to step
        self.is_retained = False
        self._score_retained = score_retained
        self._score_free = score_free

    def untwist(self):
        pass

    def done_stepping(self):
        return self.step_idx >= self.n_steps

    async def step(self):
        self.slot_history.append("retained" if self.is_retained else "free")
        self.weight += (
            self._score_retained if self.is_retained else self._score_free
        )
        self.step_idx += 1


def test_runs_to_termination():
    """Smoke test: with a model that finishes after ``n_steps`` calls,
    the loop terminates and returns ``n_particles`` particles."""
    model = _MockModel(n_steps=3)
    particles = asyncio.run(
        csmc_standard(model, n_particles=4, ess_threshold=0.5)
    )
    assert len(particles) == 4
    for p in particles:
        assert p.done_stepping()


def test_slot_zero_history_all_retained():
    """Slot 0's recorded history is "retained" at every step, regardless
    of how many resamples happened in between."""
    model = _MockModel(n_steps=4)
    particles = asyncio.run(
        csmc_standard(model, n_particles=8, ess_threshold=1.0)
    )
    assert particles[0].slot_history == ["retained"] * 4


def test_other_slots_free_at_termination():
    """Non-zero slots have ``is_retained == False`` at termination.

    Note: we do *not* assert anything about ``slot_history`` of non-zero
    slots, because the loop's post-last-step resample (after the final
    step but before the while-check exits) may overwrite their
    ``slot_history`` with deepcopied content from slot 0. The invariant
    that survives that final inheritance is on the ``is_retained`` tag
    itself, which the resample block re-establishes (slot 0 stays
    retained, slots 1..N-1 are explicitly marked free) immediately after
    every resample.
    """
    model = _MockModel(n_steps=4)
    particles = asyncio.run(
        csmc_standard(model, n_particles=8, ess_threshold=1.0)
    )
    for p in particles[1:]:
        assert p.is_retained is False


def test_n_particles_one():
    """M=1 degenerate case: only the retained slot exists, no
    resampling possible, all steps record "retained"."""
    model = _MockModel(n_steps=2)
    particles = asyncio.run(
        csmc_standard(model, n_particles=1, ess_threshold=0.5)
    )
    assert len(particles) == 1
    assert particles[0].is_retained is True
    assert particles[0].slot_history == ["retained", "retained"]


def test_n_particles_zero_raises():
    with pytest.raises(ValueError, match="n_particles must be >= 1"):
        asyncio.run(csmc_standard(_MockModel(), n_particles=0))


def test_skips_resample_when_all_particles_dead():
    """If every particle's weight is -inf, the resample block must skip
    rather than feed NaN into ``np.random.choice``. Mirrors the same
    guard in ``smc_standard``."""
    model = _MockModel(
        n_steps=2, score_retained=float("-inf"), score_free=float("-inf")
    )
    particles = asyncio.run(
        csmc_standard(model, n_particles=4, ess_threshold=1.0)
    )
    assert len(particles) == 4
    for p in particles:
        assert p.done_stepping()
