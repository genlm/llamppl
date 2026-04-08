"""Resampling methods for sequential Monte Carlo.

Four standard resampling algorithms, ordered from lowest to highest variance:

- **systematic**: Single random offset, evenly spaced. Lowest variance,
  standard choice in the SMC literature (Heng 2017, Golowich 2026).
- **stratified**: One random draw per stratum. Low variance, slightly
  more random than systematic.
- **residual**: Deterministic floor copies + multinomial on remainders.
  Low variance, maximizes deterministic component.
- **multinomial**: Independent categorical draws. Highest variance,
  included as baseline.

All functions take a 1-D array of normalized probability weights and
return an integer array of ancestor indices.

Reference implementations from FilterPy (R. Labbe):
https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html
"""

import numpy as np


def systematic_resample(weights):
    """Systematic resampling with a single random offset.

    Separates the sample space into N equal divisions and uses one
    random offset for all divisions. Every sample is exactly 1/N apart.
    Lowest variance among standard resampling methods.

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # avoid round-off errors
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def stratified_resample(weights):
    """Stratified resampling with one random draw per stratum.

    Divides the cumulative sum into N equal strata and draws one
    sample uniformly from each. Guarantees samples are between
    0 and 2/N apart.

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    positions = (np.random.random(N) + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def residual_resample(weights):
    """Residual resampling: deterministic floor copies + multinomial remainder.

    Takes floor(N * w_i) copies of each particle deterministically,
    then resamples the remaining slots from the fractional residuals
    using multinomial resampling.

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.

    Reference:
        J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
        systems. JASA, 93(443):1032-1044, 1998.
    """
    N = len(weights)
    weights = np.asarray(weights, dtype=float)
    indexes = np.zeros(N, "i")

    # Deterministic copies
    num_copies = np.floor(N * weights).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):
            indexes[k] = i
            k += 1

    # Multinomial resample on the residual
    if k < N:
        residual = weights * N - num_copies
        residual /= residual.sum()
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N - k))

    return indexes


def multinomial_resample(weights):
    """Multinomial resampling: independent categorical draws.

    Each of the N ancestor indices is drawn independently from the
    categorical distribution defined by the weights. Matches the
    original llamppl implementation (N independent np.random.choice
    calls) for reproducibility with existing results.

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    return np.array([
        np.random.choice(N, p=weights) for _ in range(N)
    ], dtype=np.intp)


RESAMPLING_METHODS = {
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "residual": residual_resample,
    "multinomial": multinomial_resample,
}


def get_resampling_fn(method):
    """Get a resampling function by name.

    Args:
        method (str): One of 'systematic', 'stratified', 'residual', 'multinomial'.

    Returns:
        callable: Resampling function that takes weights and returns indices.

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method '{method}'. "
            f"Must be one of: {', '.join(RESAMPLING_METHODS.keys())}"
        )
    return RESAMPLING_METHODS[method]
