"""Utility functions"""

import numpy as np


def logsumexp(nums):
    m = np.max(nums)
    return np.log(np.sum(np.exp(nums - m))) + m


def log_softmax(nums):
    """Compute log(softmax(nums)).

    Args:
        nums: a vector or numpy array of unnormalized log probabilities.

    Returns:
        np.array: an array of log (normalized) probabilities.
    """
    return nums - logsumexp(nums)


def softmax(nums):
    return np.exp(log_softmax(nums))
