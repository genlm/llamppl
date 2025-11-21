import asyncio

import nltk

from llamppl import CachedCausalLM
from llamppl import LMContext
from llamppl import Model
from llamppl import sample_word
from llamppl import smc_standard

# download the CMU pronunciation dictionary (if we haven't already)
nltk.download("cmudict")

# Load the CMU pronunciation dictionary and use it for syllable counting
from nltk.corpus import cmudict

CMUDICT = cmudict.dict()


def count_syllables(word, unknown_word_syllables=100):
    # Use the dictionary to get the list of possible phonetic representations for the word
    phonetic_transcriptions = CMUDICT.get(word.strip().lower(), [])

    # Count the number of syllables based on the number of phonetic transcriptions
    syllable_count = min(
        [
            len([ph for ph in transcription if ph[-1].isdigit()])
            for transcription in phonetic_transcriptions
        ],
        default=unknown_word_syllables,
    )

    return syllable_count


# Example poems for the prompt.
# Authors:
#   - Amy Lowell
#   - Sonia Sanchez
#   - Katsushika Hokusai
#   - Matsuo Basho
# Note that not all of these follow the syllabic constraints of a Haiku; the goal is
# to encode a certain 'poetic style' but to leave the syllabic constraints to be enforced
# by the probabilistic program (enabling generalization to other syllabic constraints).
EXAMPLE_POEMS = """Example poems. Note how they tend to end on a somewhat surprising or otherwise satisfying note, and are not repetitive at the end.

1. "Portrait"
Sweet smell of wet flowers
Over an evening garden.
Your portrait, perhaps?

2. "River of Love"
love between us is
speech and breath. loving you is
a long river running.

3. "Practice"
I write, erase, rewrite
Erase again, and then
A poppy blooms.

4. "Caterpillar"
A caterpillar,
this deep in fall,
still not a butterfly."""


# LLaMPPL model
class Haiku(Model):
    def __init__(self, LLM, prompt, syllable_pattern=[5, 7, 5]):
        super().__init__()
        self.context = LMContext(LLM, prompt)
        self.syllable_pattern = syllable_pattern
        self.previous_string = str(self.context)
        self.newline_token = LLM.str_vocab.index("\n")
        self.eos_token = LLM.tokenizer.eos_token_id

    async def step(self):
        self.previous_string = str(self.context)

        # Get the number of syllables required in the next line
        syllables_remaining = self.syllable_pattern.pop(0)

        # Loop to sample words until this line is over
        while syllables_remaining > 0:
            # Sample a word
            word, punctuation = await self.call(sample_word(self.context))

            # Subtract syllables from the remaining count
            syllables_remaining -= count_syllables(word)

        # Reject if we overshot
        self.condition(syllables_remaining == 0)

        # If there are no more lines, finish
        if not self.syllable_pattern:
            await self.observe(self.context.next_token(), self.eos_token)
            self.finish()
            return

        # Otherwise, observe a line break
        await self.observe(self.context.next_token(), self.newline_token)

        # Print current result
        print(str(self.context))

    def string_for_serialization(self):
        # Replace newlines with slashes in str(self.context)
        s = (
            self.previous_string
            + "<<<"
            + str(self.context)[len(self.previous_string) :]
            + ">>>"
        )
        return s.replace("\n", "/")


async def run_example(
    LLM,
    poem_title,
    syllable_pattern=[5, 7, 5],
    n_particles=20,
    ess_threshold=0.5,
    resampling_method="multinomial",
):
    # Construct prompt
    prompt = f"""{EXAMPLE_POEMS}

5. "{poem_title}"
"""

    # Cache the key value vectors for the prompt
    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    # Initialize the Model
    haiku_model = Haiku(LLM, prompt, syllable_pattern)

    # Run inference
    particles = await smc_standard(
        haiku_model,
        n_particles,
        ess_threshold,
        "html",
        "results/haiku.json",
        resampling_method,
    )

    return particles


def main():
    # Load the language model.
    # Mistral is an open model; to use a model with restricted access, like LLaMA 3,
    # authenticate using the Huggingface CLI.
    LLM = CachedCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # LLM = CachedCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set batch size if using HuggingFace backend
    if LLM.backend == "hf":
        LLM.batch_size = 40

    # Get poem title from user
    poem_title = input("Enter a title for your Haiku: ")

    syllables_per_line = [5, 7, 5]  # [5, 3, 5] for a Lune

    # Run the example
    particles = asyncio.run(
        run_example(LLM, poem_title, syllable_pattern=syllables_per_line)
    )

    print("--------")
    for i, particle in enumerate(particles):
        print(f"\nPoem {i} (weight {particle.weight}):")
        print(f"{particle.context}")


if __name__ == "__main__":
    main()
