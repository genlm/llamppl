import numpy as np

from .distribution import Distribution


class Bernoulli(Distribution):
    """A Bernoulli distribution."""

    def __init__(self, p):
        """Create a Bernoulli distribution.

        Args:
            p: the probability-of-True for the Bernoulli distribution.
        """
        self.p = p

    async def sample(self):
        b = np.random.rand() < self.p
        return (b, await self.log_prob(b))

    async def log_prob(self, value):
        return np.log(self.p) if value else np.log1p(-self.p)

    async def argmax(self, idx):
        return (self.p > 0.5) if idx == 0 else (self.p < 0.5)
