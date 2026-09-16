import numpy as np
from genlm.backend.draw import draw_from, logp_at, logsumexp_from
from genlm.backend.llm import temper

from ..llms import Token
from .distribution import Distribution


class LMNextToken(Distribution):
    def __init__(self, ctx):
        self.ctx = ctx

    async def log_prob(self, x):
        if isinstance(x, Token):
            x = x.token_id

        lp = await logp_at(self.ctx.next_token_logprobs, x)
        await self.ctx._advance(x)
        return lp

    async def sample(self):
        token_id, _, logprob = await draw_from(self.ctx.next_token_logprobs)
        await self.ctx._advance(token_id)
        t = Token(
            self.ctx.lm, token_id, self.ctx.lm.tokenizer.convert_ids_to_tokens(token_id)
        )
        return t, logprob


class LMTokenMask(Distribution):
    def __init__(self, ctx, mask):
        self.ctx = ctx
        self.mask = mask

    def _masked(self, drop):
        """The next-token row with the tokens in ``drop`` ruled out."""
        return self.ctx.next_token_logprobs.masked_fill(drop, float("-inf"))

    async def sample(self):
        yes = self._masked(~self.mask)
        logprob_yes_mask = await logsumexp_from(yes)
        if logprob_yes_mask >= 0:
            logprob_no_mask = float("-inf")
        else:
            # `log1p(-1.0)` warns on the way to -inf, which is the value wanted here.
            with np.errstate(divide="ignore"):
                logprob_no_mask = np.log1p(-np.exp(logprob_yes_mask))
        if np.random.rand() < np.exp(logprob_no_mask):
            self.ctx._restrict(self._masked(self.mask), logprob_no_mask)
            return False, logprob_no_mask
        self.ctx._restrict(yes, logprob_yes_mask)
        return True, logprob_yes_mask

    async def log_prob(self, v):
        masked = self._masked(~self.mask if v else self.mask)
        logprob_good = await logsumexp_from(masked)
        if logprob_good == float("-inf"):
            # Conditioning on a mask no token satisfies is a zero-probability event.
            # The context stays as it was; `Model.observe` finishes the particle.
            return logprob_good
        self.ctx._restrict(masked, logprob_good)
        return logprob_good


class LMContext:
    """Represents a generation-in-progress from a language model.

    The state tracks two pieces of information:

    * A sequence of tokens — the ever-growing context for the language model.
    * A *current mask* — carried in `next_token_logprobs`, whose -inf entries are the
      tokens ruled out as the next token.

    Storing a mask enables _sub-token_ generation: models can use `LMContext` to sample
    the next token in _stages_, first deciding, e.g., whether to use an upper-case or lower-case
    first letter, and only later deciding which upper-case or lower-case token to generate.

    The state of a `LMContext` can be advanced in two ways:

    1. Sampling, observing, or intervening the `next_token()` distribution. This causes a token
    to be added to the growing sequence of tokens. Supports auto-batching.
    2. Sampling, observing, or intervening the `mask_dist(mask)` distribution for a given mask (a
    `[V]` boolean tensor over the vocabulary). This changes the current mask.

    Attributes:
        lm (llamppl.llms.CachedCausalLM): the language model for which this is a context
        tokens (list[int]): the underlying sequence of tokens, including prompt, in this context
        next_token_logprobs (torch.Tensor): `[V]` log probabilities for the next token, on
            `lm.device`. Unlike the log probabilities reported by `CachedCausalLM.next_token_logprobs`,
            these are rescaled for this `LMContext`'s temperature parameter, and for any active masks.
            Managed internally; do not mutate.
        temp (float): temeprature for next-token distribution (0 < temp < float('inf'))
        show_prompt (bool): controls whether the string representation of this `LMContext` includes the initial prompt or not. Defaults to `False`.
    """

    def __init__(self, lm, prompt, temp=1.0, show_prompt=False, show_eos=True):
        """Create a new `LMContext` with a given prompt and temperature.

        Args:
            lm (llamppl.llms.CachedCausalLM): the language model for which this is a context.
            prompt (str): a string with which to initialize the context. Will be tokenized using `lm.tokenizer`.
            temp (float): temeprature for next-token distribution (0 < temp < float('inf'))

        Note:
            For async initialization of LMContext, use LMContext.create().
        """
        self._init_common(lm, prompt, temp, show_prompt, show_eos)
        self._set_row(lm.next_token_logprobs_unbatched(self.tokens))

    @classmethod
    async def create(cls, lm, prompt, temp=1.0, show_prompt=False, show_eos=True):
        """Asynchronously create a new `LMContext` with a given prompt and temperature."""
        self = cls.__new__(cls)
        self._init_common(lm, prompt, temp, show_prompt, show_eos)
        self._set_row(await lm.next_token_logprobs(self.tokens))
        return self

    def _init_common(self, lm, prompt, temp, show_prompt, show_eos):
        """Initialize common attributes shared between __init__ and create."""
        self.lm = lm
        self.tokens = lm.tokenizer.encode(prompt)
        self.temp = temp
        self.prompt_string_length = len(lm.tokenizer.decode(self.tokens))
        self.prompt_token_count = len(self.tokens)
        self.show_prompt = show_prompt
        self.show_eos = show_eos

    def _set_row(self, logprobs):
        """Install a fresh next-token row at this context's temperature."""
        self.next_token_logprobs = temper(logprobs, self.temp)

    async def _advance(self, token_id):
        """Append `token_id` and fetch the row for the extended context."""
        self.tokens.append(token_id)
        self._set_row(await self.lm.next_token_logprobs(self.tokens))

    def _restrict(self, masked, log_norm):
        """Condition the next-token row on `masked`, whose log-mass is `log_norm`."""
        self.next_token_logprobs = masked.sub_(log_norm)

    def next_token(self):
        """Distribution over the next token.

        Sampling or observing from this distribution advances the state of this `LMContext` instance.
        """
        return LMNextToken(self)

    def mask_dist(self, mask):
        """Bernoulli distribution, with probability of True equal to the probability that the next token of this `LMContext` belongs
        to the given mask.

        Sampling or observing from this distribution modifies the state of this `LMContext` instance, so that
        the `next_token()` distribution either *will* (if True) or *will not* (if False) generate a token from
        the given mask.

        Args:
            mask (torch.Tensor): a `[V]` boolean tensor over the vocabulary, such as one of
                `lm.masks` or `lm.token_mask(ids)`.
        """
        return LMTokenMask(self, mask)

    @property
    def token_count(self):
        return len(self.tokens) - self.prompt_token_count

    def __str__(self):
        full_string = self.lm.tokenizer.decode(self.tokens)
        if not self.show_prompt:
            full_string = full_string[self.prompt_string_length :]
        if not self.show_eos and full_string.endswith(self.lm.tokenizer.eos_token):
            full_string = full_string[: -len(self.lm.tokenizer.eos_token)]
        return full_string

    def __deepcopy__(self, memo):
        # Everything but the token list is shared: `lm` by design, the scalars because
        # they are immutable, the row because it is replaced, never edited.
        cpy = type(self).__new__(type(self))
        cpy.__dict__.update(self.__dict__)
        cpy.tokens = list(self.tokens)
        return cpy
