# Working with Transformers

## Load your Transformer as a `CachedCausalLM`

The easiest way to load a Transformer model is to use the [`CachedCausalLM.from_pretrained`][llamppl.llms.CachedCausalLM.from_pretrained] static method, which accepts as input a HuggingFace model identifier. This loads the model's weights into memory, and also loads the appropriate tokenizer. The optional `auth_token` parameter can be provided if the model in question requires HuggingFace authorization (e.g., Meta's Llama 2 models).

## Use the LLM within your model via the `Transformer` distribution

Within a model, you can `sample` or `observe` from the [`Transformer`][llamppl.distributions.transformer.Transformer] distribution. It accepts as arguments a [`CachedCausalLM`][llamppl.llms.CachedCausalLM] instance, as well as a list of integer token ids specifying the context. It returns a distribution over next tokens. The [`Transformer`][llamppl.distributions.transformer.Transformer] distirbution is stateless, and so your model will need to manually extend the context with newly sampled tokens.

## Use the LLM within your model via the `LMContext` class

Alternatively, you can initialize an [`LMContext`][llamppl.distributions.lmcontext.LMContext] object with a [`CachedCausalLM`][llamppl.llms.CachedCausalLM] instance instance and a string-valued prompt. It maintains a growing context as state, and exposes a [`next_token`][llamppl.distributions.lmcontext.LMContext.next_token] distribution that, when sampled, observed, or intervened, grows the context. It also supports a form of 'sub-token' generation, via the [`mask_dist`][llamppl.distributions.lmcontext.LMContext.mask_dist] distribution.

## Create custom token distributions with `TokenCategorical`

You may also create a custom distribution over the vocabulary of a language model using the [`TokenCategorical`][llamppl.distributions.tokencategorical.TokenCategorical] distribution. It is parameterized by a [`CachedCausalLM`][llamppl.llms.CachedCausalLM] instance, and an array of logits equal in length to the language model's vocabulary size.
This distribution is particularly useful as a proposal distribution; for example, a model might `sample` with `dist` set
to the LM's next token distribution, but with `proposal` set to a modified distribution that uses a heuristic to upweight
'good' tokens and downweight 'bad' ones.
