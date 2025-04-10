# Auto-Batching

If running in a GPU-accelerated environment, LLaMPPL supports **auto-batching**.

The `step` method of a LLaMPPL model describes how to advance a *single* particle one step of generation.
But inference methods must maintain many particles at once.

With auto-batching, LLaMPPL will execute particles' `step` methods concurrently, and automatically batch calls
to large language models. This batching is handled by the `CachedCausalLM` object.

If you are using the `vllm` backend, all batching decisions are handled internally by `vllm`.

If you are using the huggingface backend, the behavior is controlled by two parameters:

* `lm.batch_size`: the maximum number of requests to batch. The default value is 20.
* `lm.timeout`: if `lm.timeout` seconds pass with no new request, the current batch is processed even if not full. The default value is 0.02.

You may want to set the batch size (`#!python lm.batch_size`) to the number of particles you are using (if the number of particles is not too large).
