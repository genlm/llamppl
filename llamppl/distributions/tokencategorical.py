import torch
from genlm.backend.draw import draw_from, logp_at

from ..llms import Token
from .distribution import Distribution


class TokenCategorical(Distribution):
    def __init__(self, lm, logits):
        """Create a Categorical distribution whose values are Tokens, not integers.
        Given a language model `lm` and an array of unnormalized log probabilities (of length `len(lm.vocab)`),
        uses softmax to normalize them and samples a Token from the resulting categorical.

        Args:
            lm (llamppl.llms.CachedCausalLM): the language model whose vocabulary is to be generated from.
            logits (torch.Tensor | numpy.ndarray): unnormalized log probabilities.
        """
        self.lm = lm
        self.log_probs = torch.log_softmax(torch.as_tensor(logits), dim=-1)
        if self.lm.tokenizer.vocab_size != len(logits):
            raise RuntimeError(
                f"TokenCategorical: vocab size is {self.lm.tokenizer.vocab_size} but provided {len(logits)} logits."
            )

    async def sample(self):
        n, _, logprob = await draw_from(self.log_probs)
        return (
            Token(self.lm, n, self.lm.tokenizer.convert_ids_to_tokens(n)),
            logprob,
        )

    async def log_prob(self, value):
        return await logp_at(self.log_probs, value.token_id)

    async def argmax(self, idx):
        tok = int(torch.argsort(self.log_probs)[-idx])
        return (
            Token(self.lm, tok, self.lm.tokenizer.convert_ids_to_tokens(tok)),
            self.log_probs[tok].item(),
        )
