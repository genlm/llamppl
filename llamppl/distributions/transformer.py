from genlm.backend.draw import draw_from, logp_at
from genlm.backend.llm import temper

from ..llms import Token
from ..llms import TokenSequence
from .distribution import Distribution


# Transformer(lm, prompt) -- where prompt can either be a string or a list of Tokens.
class Transformer(Distribution):
    def __init__(self, lm, prompt, temp=1.0):
        """Create a Categorical distribution whose values are Tokens, with probabilities given
        by a language model. Supports auto-batching.

        Args:
            lm (llamppl.llms.CachedCausalLM): the language model.
            prompt (str | llamppl.llms.TokenSequence): the sequence of tokens to use as the prompt. If a string, `lm.tokenizer` is used to encode it.
            temp (float): temperature at which to generate (0 < `temp` < `float('inf')`).
        """
        self.lm = lm
        self.temp = temp

        # prompt will be a list of ints
        if isinstance(prompt, str):
            prompt = self.lm.tokenizer.encode(prompt)
        elif isinstance(prompt, TokenSequence):
            prompt = prompt.seq

        self.prompt = prompt

    async def _log_probs(self):
        return temper(await self.lm.next_token_logprobs(self.prompt), self.temp)

    async def log_prob(self, x):
        if isinstance(x, Token):
            x = x.token_id
        return await logp_at(await self._log_probs(), x)

    async def sample(self):
        token_id, _, logprob = await draw_from(await self._log_probs())
        return (
            Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)),
            logprob,
        )


#     def argmax(self, idx):
#         token_id = np.argsort(self.log_probs)[-idx]
#         return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), log_probs[token_id]
