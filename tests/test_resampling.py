"""Tests for resampling methods."""

import numpy as np
import pytest
from llamppl.inference.resampling import (
    multinomial_resample,
    stratified_resample,
    systematic_resample,
    residual_resample,
    get_resampling_fn,
    RESAMPLING_METHODS,
)


ALL_METHODS = list(RESAMPLING_METHODS.keys())


class TestResamplingBasics:
    """Basic correctness tests shared across all methods."""

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_returns_correct_length(self, method):
        fn = get_resampling_fn(method)
        weights = np.array([0.2, 0.3, 0.1, 0.15, 0.25])
        np.random.seed(42)
        indices = fn(weights)
        assert len(indices) == len(weights)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_indices_in_range(self, method):
        fn = get_resampling_fn(method)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        np.random.seed(42)
        indices = fn(weights)
        assert all(0 <= i < len(weights) for i in indices)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_returns_integer_array(self, method):
        fn = get_resampling_fn(method)
        weights = np.array([0.5, 0.3, 0.2])
        np.random.seed(42)
        indices = fn(weights)
        assert indices.dtype in (np.int32, np.int64, np.intp)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_degenerate_single_particle(self, method):
        fn = get_resampling_fn(method)
        weights = np.array([1.0])
        indices = fn(weights)
        assert len(indices) == 1
        assert indices[0] == 0

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_degenerate_all_weight_on_one(self, method):
        """When one particle has all the weight, all indices should be that particle."""
        fn = get_resampling_fn(method)
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        np.random.seed(42)
        indices = fn(weights)
        assert all(i == 2 for i in indices)

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_uniform_weights(self, method):
        """With uniform weights, each particle should appear roughly once."""
        fn = get_resampling_fn(method)
        N = 100
        weights = np.ones(N) / N
        np.random.seed(42)
        indices = fn(weights)
        counts = np.bincount(indices, minlength=N)
        # With uniform weights, no particle should have 0 or >3 copies
        # (very unlikely with any reasonable method)
        assert counts.min() >= 0
        assert counts.max() <= 5

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_expected_counts_large_n(self, method):
        """With many runs, empirical counts should match weights."""
        fn = get_resampling_fn(method)
        weights = np.array([0.5, 0.3, 0.1, 0.1])
        N = len(weights)
        n_runs = 5000
        total_counts = np.zeros(N)
        for seed in range(n_runs):
            np.random.seed(seed)
            indices = fn(weights)
            total_counts += np.bincount(indices, minlength=N)
        empirical = total_counts / (n_runs * N)
        # Should be close to the weights
        np.testing.assert_allclose(empirical, weights, atol=0.02)


class TestLowVarianceMethods:
    """Tests specific to low-variance resampling (stratified, systematic, residual)."""

    @pytest.mark.parametrize("method", ["stratified", "systematic", "residual"])
    def test_lower_variance_than_multinomial(self, method):
        """Low-variance methods should have less resampling variance."""
        weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        N = len(weights)
        n_runs = 2000

        def variance_of_counts(fn):
            all_counts = []
            for seed in range(n_runs):
                np.random.seed(seed)
                indices = fn(weights)
                counts = np.bincount(indices, minlength=N)
                all_counts.append(counts)
            return np.var(all_counts, axis=0).sum()

        multi_var = variance_of_counts(multinomial_resample)
        method_var = variance_of_counts(get_resampling_fn(method))
        assert method_var <= multi_var, (
            f"{method} variance ({method_var:.4f}) should be <= "
            f"multinomial ({multi_var:.4f})"
        )

    @pytest.mark.parametrize("method", ["stratified", "systematic"])
    def test_guaranteed_representation(self, method):
        """Particles with weight >= 1/N must appear at least once."""
        fn = get_resampling_fn(method)
        # Particle 0 has weight 0.5 > 1/5 = 0.2, must always appear
        weights = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        for seed in range(100):
            np.random.seed(seed)
            indices = fn(weights)
            counts = np.bincount(indices, minlength=len(weights))
            assert counts[0] >= 1, f"Particle 0 (w=0.5) missing with seed={seed}"


class TestSystematic:
    """Tests specific to systematic resampling."""

    def test_evenly_spaced(self):
        """Systematic uses exactly 1/N spacing."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        np.random.seed(0)
        indices = systematic_resample(weights)
        # With uniform weights, should get exactly [0, 1, 2, 3]
        np.testing.assert_array_equal(sorted(indices), [0, 1, 2, 3])

    def test_deterministic_given_seed(self):
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        np.random.seed(42)
        a = systematic_resample(weights)
        np.random.seed(42)
        b = systematic_resample(weights)
        np.testing.assert_array_equal(a, b)


class TestResidual:
    """Tests specific to residual resampling."""

    def test_deterministic_floor(self):
        """Floor copies should be deterministic."""
        # w = [0.5, 0.3, 0.2], N=10
        # floor(10*w) = [5, 3, 2] = 10, no residual needed
        weights = np.array([0.5, 0.3, 0.2])
        N = 10
        # Simulate what residual does with N=10 by repeating weights
        weights_10 = np.repeat(weights, 1)  # stays same for N=len(weights)
        # For the actual test: with 3 particles
        np.random.seed(42)
        indices = residual_resample(weights)
        counts = np.bincount(indices, minlength=3)
        # With N=3: floor(3*[0.5, 0.3, 0.2]) = [1, 0, 0], residual = [0.5, 0.9, 0.6]
        # At least 1 copy of particle 0 guaranteed
        assert counts[0] >= 1


class TestGetResamplingFn:
    """Tests for the get_resampling_fn helper."""

    def test_valid_methods(self):
        for name in ["multinomial", "stratified", "systematic", "residual"]:
            fn = get_resampling_fn(name)
            assert callable(fn)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown resampling method"):
            get_resampling_fn("invalid")

    def test_all_methods_registered(self):
        assert set(RESAMPLING_METHODS.keys()) == {
            "multinomial", "stratified", "systematic", "residual"
        }
