import numpy as np

from .distribution import Distribution


class Geometric(Distribution):
    """A Geometric distribution."""

    def __init__(self, p):
        """Create a Geometric distribution.

        Args:
            p: the rate of the Geometric distribution.
        """
        self.p = p

    async def sample(self):
        n = np.random.geometric(self.p)
        return n, await self.log_prob(n)

    async def log_prob(self, value):
        return np.log(self.p) + np.log(1 - self.p) * (value - 1)

    async def argmax(self, idx):
        return idx - 1  # Most likely outcome is 0, then 1, etc.
