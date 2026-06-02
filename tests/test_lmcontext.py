import asyncio

import numpy as np
import pytest
import torch

from llamppl.distributions.lmcontext import LMContext
from llamppl.llms import CachedCausalLM, MLX_AVAILABLE
from llamppl.modeling import Model

if MLX_AVAILABLE:
    backends = ["mock", "mlx"]
else:
    backends = [
        "mock",
        "hf",
        pytest.param(
            "vllm",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="vLLM backend requires CUDA"
            ),
        ),
    ]


@pytest.fixture
def lm(backend):
    kwargs = {"cache_size": 10} if backend == "mlx" else {}
    return CachedCausalLM.from_pretrained("gpt2", backend=backend, **kwargs)


@pytest.mark.parametrize("backend", backends)
def test_init(lm):
    prompt = "Hello, world!"
    lmcontext = LMContext(lm, prompt)
    assert lmcontext.tokens == lm.tokenizer.encode(prompt)
    logprobs = lm.next_token_logprobs_unbatched(lmcontext.tokens)
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs,
        logprobs,
        rtol=5e-4,
        err_msg="Sync context __init__",
    )

    async def async_context():
        return LMContext(lm, prompt)

    lmcontext = asyncio.run(async_context())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs,
        logprobs,
        rtol=5e-4,
        err_msg="Async context __init__",
    )

    async def async_context_create():
        return await LMContext.create(lm, prompt)

    lmcontext = asyncio.run(async_context_create())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs,
        logprobs,
        rtol=5e-4,
        err_msg="Async context create",
    )


def test_observe_impossible_mask_kills_particle():
    # Regression for #47: conditioning on a mask that rules out every token is a
    # zero-probability event. LMTokenMask.log_prob must return -inf (not raise),
    # and Model.observe must finish the particle (weight 0) so it is dropped at the
    # next resample instead of aborting the whole run. Backend-independent, so a
    # fast mock LM with a hand-set, disjoint mask exercises the path deterministically.
    lm = CachedCausalLM.from_pretrained("gpt2", backend="mock")
    ctx = LMContext(lm, "Hello, world!")
    ctx.model_mask = {0, 1, 2}
    impossible = ctx.mask_dist({3, 4})  # disjoint from model_mask: no good tokens

    m = Model()
    result = asyncio.run(m.observe(impossible, True))

    assert result is True
    assert m.weight == float("-inf")  # zero-probability observation -> weight 0
    assert m.finished  # ...and the particle is finished
    assert ctx.model_mask == {0, 1, 2}  # context untouched (returned before mutating)
