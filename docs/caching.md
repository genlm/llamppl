# Caching in LLaMPPL

LLaMPPL performs two kinds of caching to improve performance.

## Log probability caching
Next-token log probabilities are always cached, whenever they are computed.
This way, if different particles make exactly the same log probability queries,
the Transformer is run only once. This is primarily beneficial when:

* particles are cloned during resampling: when each particle is

* cloned particles happen to sample the same next token: if the next-token distribution is concentrated,
  it is likely that multiple copies of a particle will sample the same next token. Log probability caching
  allows them to sample the _following_ token using only a single call to the language model.

The log probability cache can be cleared using the [`lm.clear_cache()`][llamppl.llms.CachedCausalLM.clear_cache] method. Note that this method
will also clear the KV cache.

## Key-value caching
Key-value caching caches the key and value vectors computed by each layer of a Transformer,
for reuse when processing new tokens at the end of a previously evaluated sequence.

In principle, key-value caching is most useful when:

* There is a long common *prompt* from which all particles are generating.
  In this case, the prompt's tokens can be evaluated just once by the language model,
  and each subsequent call only has to pay for the new tokens generated after the prompt.

* Generations from the model are very long. In this case, it may be worth paying the memory
  cost to cache *different* key-value sequences for *each* particle, to speed up future next-token
  queries.

Currently, only the first use case is well-supported by the LLaMPPL library, via the
[`lm.cache_kv(prompt)`][llamppl.llms.CachedCausalLM.cache_kv] method. This method computes and caches key and value vectors
for every token in `prompt`. Future calls to [`lm.next_token_logprobs`][llamppl.llms.CachedCausalLM.next_token_logprobs] and [`lm.next_token_logprobs_unbatched`][llamppl.llms.CachedCausalLM.next_token_logprobs_unbatched]
will automatically recognize when `prompt` is a prefix of the new query, and automatically
exploit incremental computation. Multiple prompts can be cached, and [`lm.clear_kv_cache()`][llamppl.llms.CachedCausalLM.clear_kv_cache] can
be used to clear the KV-cache without clearing the log probability cache.

Because [`lm.cache_kv`][llamppl.llms.CachedCausalLM.cache_kv] is not a batched call,
it is not well-suited to caching
different strings for different particles.
Rather, it is best used in the `__init__` method of a model--or even
outside of a model--on fixed prompt strings that every particle will share.
