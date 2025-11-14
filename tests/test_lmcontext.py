import asyncio

import numpy as np
import pytest
import torch

from llamppl.distributions.lmcontext import LMContext
from llamppl.llms import CachedCausalLM, MLX_AVAILABLE

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
    kwargs = (
        {"engine_opts": {"gpu_memory_utilization": 0.45}}
        if backend == "vllm"
        else {"cache_size": 10}
        if backend == "mlx"
        else {}
    )
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
        rtol=1e-5,
        err_msg="Sync context __init__",
    )

    async def async_context():
        return LMContext(lm, prompt)

    lmcontext = asyncio.run(async_context())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs,
        logprobs,
        rtol=1e-5,
        err_msg="Async context __init__",
    )

    async def async_context_create():
        return await LMContext.create(lm, prompt)

    lmcontext = asyncio.run(async_context_create())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs,
        logprobs,
        rtol=1e-5,
        err_msg="Async context create",
    )
