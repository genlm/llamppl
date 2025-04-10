"""SMC Steering with Grammar Constraints

Author: Gabriel Grand (grandg@mit.edu)

This example illustrates grammar-constrained inference with SMC Steering.
`GrammarConstrainedSMC` takes as input a grammar in Lark format.
We use the Synchromesh (Poesia et al., 2022) to align the grammar with the
language model vocabulary.

Requires synchromesh (github.com/kanishkg/synchromesh)
"""

import asyncio

from synchromesh.completion_engine import LarkCompletionEngine
from synchromesh.synchromesh import StreamingCSD

from llamppl.distributions import LMContext
from llamppl.inference import smc_standard
from llamppl.llms import CachedCausalLM
from llamppl.modeling import Model


class GrammarConstrainedSMC(Model):
    def __init__(
        self,
        lm: CachedCausalLM,
        grammar: str,
        start_rule: str,
        prompt: str = None,
        allow_ws: bool = False,
        max_tokens: int = 32,
        verbose: bool = False,
    ):
        super().__init__()
        self.lm = lm
        self.grammar = grammar
        self.context = LMContext(lm, prompt)
        self.vocab = self.lm.str_vocab
        self.eos_token_id = self.lm.tokenizer.eos_token_id

        self.comp_engine = LarkCompletionEngine(
            grammar, start_token=start_rule, allow_ws=allow_ws
        )
        self.csd = StreamingCSD(
            completion_engine=self.comp_engine,
            lm_vocabulary=self.vocab,
            enforce_token_maximality=False,
        )

        self.max_tokens = max_tokens
        self.n_tokens = 0

        self.verbose = verbose

    async def step(self):
        # Get valid tokens for next step
        valid_token_ids = self.csd.get_valid_tokens()

        # If generation is a complete derivation, allow the end-of-string token
        if self.csd.is_complete():
            valid_token_ids += [self.eos_token_id]

        # If no valid next tokens, reject and terminate
        if len(valid_token_ids) == 0:
            self.condition(False)
            return

        # Sample a token from the valid tokens
        await self.observe(self.context.mask_dist(set(valid_token_ids)), True)
        token = await self.sample(self.context.next_token())

        # If the token is the end-of-string token, accept and terminate
        if token.token_id == self.eos_token_id:
            self.finish()
            return

        # Feed the token to StreamingCSD
        self.csd.feed_prediction(token.token_id)
        self.n_tokens += 1

        if self.verbose:
            print(str(self.context))

        # Max tokens reached
        if self.n_tokens >= self.max_tokens:
            self.condition(False)
            self.finish()

    def immutable_properties(self):
        return set(
            [
                "grammar",
                "max_tokens",
                "verbose",
            ]
        )


EXAMPLE_PROMPT = """Paraphrase the following sentences
Human:who teaches CSE101?
Bot:instructor of CSE101
Human:how many students can enroll in PSY456?
Bot:capacity of PSY456
Human:at what school is BIO433 taught?
Bot:"""

EXAMPLE_GRAMMAR = r"""
    ?start: " "? function " of " dept code
    function: "instructor" | "students" | "capacity" | "department" | "school" | "college"
    dept: /[A-Z]{3}/
    code: /[0-9]{3}/
"""


async def run_generation(
    model: str,
    grammar: str,
    start_rule: str,
    prompt: str = None,
    allow_ws: bool = False,
    n_particles: int = 5,
    max_tokens: int = 32,
    verbose: bool = False,
):
    LLM = CachedCausalLM.from_pretrained(args.model)
    if LLM.backend == "hf":
        LLM.batch_size = args.batch_size
    model = GrammarConstrainedSMC(
        lm=LLM,
        grammar=grammar,
        start_rule=start_rule,
        prompt=prompt,
        max_tokens=max_tokens,
        allow_ws=allow_ws,
        verbose=verbose,
    )
    particles = await smc_standard(model, n_particles=n_particles)
    particles_sorted = sorted(particles, key=lambda p: p.weight, reverse=True)
    print([(p.weight, str(p.context)) for p in particles_sorted])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="codellama/CodeLlama-7b-hf",
        help="Name of the HuggingFace model to use",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default=None,
        help="Path to the grammar file",
    )
    parser.add_argument(
        "--start-rule",
        type=str,
        default="start",
        help="Name of the start rule in the grammar",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to start generation from",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=5,
        help="Number of particles to use in SMC",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--allow-ws",
        action="store_true",
        help="Allow whitespace",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate generations",
    )
    args = parser.parse_args()

    if args.grammar is not None:
        # Load the grammar
        with open(args.grammar, "r") as f:
            grammar = f.read()
    else:
        grammar = EXAMPLE_GRAMMAR

    prompt = args.prompt or EXAMPLE_PROMPT

    asyncio.run(
        run_generation(
            model=args.model,
            grammar=grammar,
            start_rule=args.start_rule,
            prompt=prompt,
            n_particles=args.n_particles,
            max_tokens=args.max_tokens,
            allow_ws=args.allow_ws,
            verbose=args.verbose,
        )
    )
