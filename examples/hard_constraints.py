import asyncio
import string

from llamppl import CachedCausalLM
from llamppl import LMContext
from llamppl import Model
from llamppl import smc_standard


def make_masks(LLM):
    return {
        i: set(
            j
            for (j, v) in enumerate(LLM.str_vocab)
            if j != LLM.tokenizer.eos_token_id
            and "\n" not in v
            and any(c.isalpha() or c in string.punctuation for c in v)
            and len(v.strip()) <= 5
            and (not v[0].isalpha() or i + len(v) <= 5)
        )
        for i in range(6)
    }


class ConstraintModel(Model):
    def __init__(self, LLM, prompt, max_tokens):
        super().__init__()
        self.context = LMContext(LLM, prompt)
        self.max_tokens = max_tokens
        self.masks = make_masks(LLM)
        self.eos_token_id = LLM.tokenizer.eos_token_id

    async def start(self):
        mask = self.active_constraint_mask()
        await self.observe(self.context.mask_dist(mask), True)

    async def step(self):
        # Generate proposed token.
        token = await self.sample(self.context.next_token())

        # Reduce number of max tokens remaining
        self.max_tokens -= 1

        print(f"{self.context}")

        # Check if done
        if token == self.eos_token_id or self.max_tokens == 0:
            self.finish()
            return

        # Observe that next token follows the constraint.
        mask = self.active_constraint_mask()
        await self.observe(self.context.mask_dist(mask), True)

    def active_constraint_mask(self):
        string_so_far = str(self.context)
        words = string_so_far.split()
        last_word = words[-1] if len(words) > 0 else ""
        return self.masks[min(5, len(last_word))]

    def string_for_serialization(self):
        return f"{self.context}"

    def immutable_properties(self):
        return ["masks"]


# From Politico.com
prompt = """3 things to watch …

1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

3."""


async def run_example(
    LLM,
    max_tokens=50,
    n_particles=20,
    ess_threshold=0.5,
    resampling_method="multinomial",
):
    # Cache the key value vectors for the prompt.
    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    # Initialize the Model.
    constraint_model = ConstraintModel(LLM, prompt, max_tokens)

    # Run inference.
    particles = await smc_standard(
        constraint_model,
        n_particles,
        ess_threshold,
        "html",
        "results/output.json",
        resampling_method,
    )
    for p in particles:
        print(f"{p.context}")

    return particles


def main():
    # Load the language model.
    # Mistral and Vicuna are open models; to use a model with restricted access, like LLaMA 3,
    # authenticate using the Huggingface CLI.
    LLM = CachedCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # LLM = CachedCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
    # LLM = CachedCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set batch size if provided. This operation is only valid for the HuggingFace backend.
    if LLM.backend == "hf":
        LLM.batch_size = 40

    # Run the example.
    asyncio.run(run_example(LLM))


if __name__ == "__main__":
    main()
