class Distribution:
    """Abstract base class for a distribution."""

    async def sample(self):
        """Generate a random sample from the distribution.

        Returns:
            x: a value randomly sampled from the distribution."""
        raise NotImplementedError()

    async def log_prob(self, x):
        """Compute the log probability of a value under this distribution,
        or the log probability density if the distribution is continuous.

        Args:
            x: the point at which to evaluate the log probability.
        Returns:
            logprob (float): the log probability of `x`."""
        raise NotImplementedError()

    async def argmax(self, n):
        """Return the nth most probable outcome under this distribution (assuming this is a discrete distribution).

        Args:
            n (int): which value to return to, indexed from most probable (n=0) to least probable (n=|support|).
        Returns:
            x: the nth most probable outcome from this distribution."""
        raise NotImplementedError()
