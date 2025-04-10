# Getting Started

## Colab

One easy way to try LLaMPPL out is to use a Colab notebook. We have [a demo notebook](https://colab.research.google.com/drive/1uJEC-U8dcwsTWccCDGVexpgXexzZ642n?usp=sharing) that performs constrained generation with GPT-2, a small enough model that the RAM and GPU constraints of Colab's free version should not prevent you from running the demo.

## Installing LLaMPPL

To get started, clone the `llamppl` repository and install the `llamppl` package.

```bash
git clone https://github.com/genlm/llamppl
cd llamppl
poetry install
```

We use [poetry](https://python-poetry.org/) to manage dependencies. If you don't have poetry installed, you can install it with `pip install poetry`.

You can then run an example. The first time you run it, the example may ask to downlaod model weights from the HuggingFace model repository.

```bash
poetry run python examples/hard_constraints.py
```

Depending on your available GPU memory, you may wish to edit the example to change parameters such as the batch size, or which HuggingFace model to use. The `hard_constraints.py` example has been run successfully on an NVIDIA L4 GPU (with 24 GB of VRAM) on Google Cloud.

## Your First Model

Let's write a LLaMPPL model to generate according to the hard constraint that completions do not use the lowercase letter `e`.

To do so, we write subclass the [`Model`](llamppl.modeling.Model) class:

```python
# examples/no_e.py

from llamppl import Model, LMContext, CachedCausalLM

# A LLaMPPL model subclasses the Model class
class MyModel(Model):

    # The __init__ method is used to process arguments
    # and initialize instance variables.
    def __init__(self, lm, prompt, forbidden_letter):
        super().__init__()

        # A stateful context object for the LLM, initialized with the prompt
        self.context = LMContext(lm, prompt)
        self.eos_token = lm.tokenizer.eos_token_id

        # The forbidden letter
        self.forbidden_tokens = set(i for (i, v) in enumerate(lm.vocab)
                                      if forbidden_letter in v)

    # The step method is used to perform a single 'step' of generation.
    # This might be a single token, a single phrase, or any other division.
    # Here, we generate one token at a time.
    async def step(self):
        # Condition on the next token *not* being a forbidden token.
        await self.observe(self.context.mask_dist(self.forbidden_tokens), False)

        # Sample the next token from the LLM -- automatically extends `self.context`.
        token = await self.sample(self.context.next_token())

        # Check for EOS or end of sentence
        if token.token_id == self.eos_token or str(token) in ['.', '!', '?']:
            # Finish generation
            self.finish()

    # To improve performance, a hint that `self.forbidden_tokens` is immutable
    def immutable_properties(self):
        return set(['forbidden_tokens'])
```

To run the model, we use an inference method, like `smc_steer`:

```python
import asyncio
from llamppl import smc_steer

# Initialize the HuggingFace model
lm = CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=<YOUR_HUGGINGFACE_API_TOKEN_HERE>)

# Create a model instance
model = MyModel(lm, "The weather today is expected to be", "e")

# Run inference
particles = asyncio.run(smc_steer(model, 5, 3)) # number of particles N, and beam factor K
```

Each returned particle is an instance of the `MyModel` class that has been `step`-ped to completion.
The generated strings can be printed along with the particle weights:

```python
for particle in particles:
    print(f"{particle.context.s} (weight: {particle.weight})")
```


## Learning more

For more intuition on language model probabilistic programming, see [our paper](https://arxiv.org/abs/2306.03081), or the rest of this documentation.
