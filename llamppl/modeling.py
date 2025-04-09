import copy


class SubModel:
    def __init__(self):
        self.parent = None

    async def run_with_parent(self, parent):
        old_parent = self.parent
        self.parent = parent
        val = await self.forward()
        self.parent = old_parent
        return val

    async def forward(self):
        raise NotImplementedError(
            "SubModel.forward() must be implemented by subclasses"
        )

    async def sample(self, dist, proposal=None):
        return await self.parent.sample(dist, proposal)

    async def observe(self, dist, x):
        return await self.parent.observe(dist, x)

    async def intervene(self, dist, x):
        return await self.parent.intervene(dist, x)

    def condition(self, b):
        return self.parent.condition(b)

    def score(self, score):
        return self.parent.score(score)

    def twist(self, amt):
        return self.parent.twist(amt)

    async def call(self, submodel):
        return await submodel.run_with_parent(self.parent)


# For use as a decorator
import functools


def submodel(f):
    """Decorator to create a SubModel implementation from an async function.

    For example:

    ```python
    @submodel
    async def sample_two_tokens(self, context):
        token1 = await self.sample(context.next_token())
        token2 = await self.sample(context.next_token())
        return token1, token2
    ```

    This SubModel can then be used from another model or submodel, using the syntax `await self.call(sample_two_tokens(context))`.
    """

    @functools.wraps(f, updated=())  # unclear if this is the best way to do it
    class SubModelImpl(SubModel):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

        async def forward(self):
            return await f(self, *self.args, **self.kwargs)

    return SubModelImpl


class Model:
    """Base class for all LLaMPPL models.

    Your models should subclass this class. Minimally, you should provide an `__init__` method
    that calls `super().__init__(self)`, and a `step` method.
    """

    def __init__(self):
        self.weight = 0.0
        self.finished = False
        self.mode = "sample"
        self.beam_idx = 0
        self.force_eos = False
        self.twist_amount = 0.0

    def reset(self):
        self.weight = 0.0
        self.finished = False
        self.mode = "sample"
        self.beam_idx = 0
        self.force_eos = False
        self.twist_amount = 0.0

    def immutable_properties(self):
        """Return a `set[str]` of properties that LLaMPPL may assume do not change during execution of `step`.
        This set is empty by default but can be overridden by subclasses to speed up inference.

        Returns:
            properties (set[str]): a set of immutable property names"""
        return set()

    def __deepcopy__(self, memo):
        cpy = type(self).__new__(type(self))
        immutable = self.immutable_properties()

        for k, v in self.__dict__.items():
            if k in immutable:
                setattr(cpy, k, v)
            else:
                setattr(cpy, k, copy.deepcopy(v, memo))

        return cpy

    def twist(self, amt):
        """Multiply this particle's weight by `exp(amt)`, but divide it back out before the next `step`.

        Use this method to provide heuristic guidance about whether a particle is "on the right track"
        without changing the ultimate target distribution.

        Args:
            amt: the logarithm of the amount by which to (temporarily) multiply this particle's weight.
        """
        self.twist_amount += amt
        self.score(amt)

    def untwist(self):
        self.score(-self.twist_amount)
        self.twist_amount = 0.0

    def finish(self):
        self.untwist()
        self.finished = True

    def done_stepping(self):
        return self.finished

    async def step(self):
        """Defines the computation performed in each step of the model.

        All subclasses should override this method."""

        if not self.done_stepping():
            raise NotImplementedError("Model.step() must be implemented by subclasses")

    def __str__(self):
        return "Particle"

    async def start(self):
        pass

    def score(self, score):
        """Multiply this particle's weight by `exp(score)`.

        The `score` method is a low-level way to change the target distribution.
        For many use cases, it is sufficient to use `sample`, `observe`, `condition`,
        and `twist`, all of which are implemented in terms of `score`.

        Args:
            score: logarithm of the amount by which the particle's weight should be multiplied.
        """
        self.weight += score

    def condition(self, b):
        """Constrain a given Boolean expression to be `True`.

        If the condition is False, the particle's weight is set to zero and `self.finish()`
        is called, so that no further `step` calls are made.

        Args:
            b: the Boolean expression whose value is constrained to be True.
        """
        if not b:
            self.score(float("-inf"))
            self.finish()

    async def intervene(self, dist, x):
        """Force the distribution to take on the value `x`, but do not _condition_ on this result.

        This is useful primarily with distributions that have side effects (e.g., modifying some state).
        For example, a model with the code

        ```python
        token_1 = await self.sample(self.stateful_lm.next_token())
        await self.observe(self.stateful_lm.next_token(), token_2)
        ```

        encodes a posterior inference problem, to find `token_1` values that *likely preceded* `token_2`. By contrast,

        ```python
        token_1 = await self.sample(stateful_lm.next_token())
        await self.intervene(self.stateful_lm.next_token(), token_2)
        ```

        encodes a much easier task: freely generate `token_1` and then force-feed `token_2` as the following token.

        Args:
            dist (llamppl.distributions.distribution.Distribution): the distribution on which to intervene.
            x: the value to intervene with.
        """
        await dist.log_prob(x)
        return x

    async def observe(self, dist, x):
        """Condition the model on the value `x` being sampled from the distribution `dist`.

        For discrete distributions `dist`, `await self.observe(dist, x)` specifies the same constraint as
        ```
        val = await self.sample(dist)
        self.condition(val == x)
        ```
        but can be much more efficient.

        Args:
            dist: a `Distribution` object from which to observe
            x: the value observed from `dist`
        """
        p = await dist.log_prob(x)
        self.score(p)
        return x

    async def sample(self, dist, proposal=None):
        """Extend the model with a sample from a given `Distribution`, with support for autobatching.
        If specified, the Distribution `proposal` is used during inference to generate informed hypotheses.

        Args:
            dist: the `Distribution` object from which to sample
            proposal: if provided, inference algorithms will use this `Distribution` object to generate proposed samples, rather than `dist`.
              However, importance weights will be adjusted so that the target posterior is independent of the proposal.

        Returns:
            value: the value sampled from the distribution.
        """
        # Special logic for beam search
        # if self.mode == "beam":
        #     d = dist if proposal is None else proposal
        #     x, w = d.argmax(self.beam_idx)
        #     if proposal is not None:
        #         self.score(dist.log_prob(x))
        #     else:
        #         self.score(w)
        #     return x

        if proposal is None:
            x, _ = await dist.sample()
            return x
        else:
            x, q = await proposal.sample()
            p = await dist.log_prob(x)
            self.score(p - q)
            return x

    async def call(self, submodel):
        return await submodel.run_with_parent(self)

    def string_for_serialization(self):
        """Return a string representation of the particle for serialization purposes.

        Returns:
            str: a string representation of the particle.
        """
        return str(self)
