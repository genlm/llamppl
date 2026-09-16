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
    logprobs = lm.next_token_logprobs_unbatched(lmcontext.tokens).cpu()
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs.cpu(),
        logprobs,
        rtol=5e-4,
        err_msg="Sync context __init__",
    )

    async def async_context():
        return LMContext(lm, prompt)

    lmcontext = asyncio.run(async_context())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs.cpu(),
        logprobs,
        rtol=5e-4,
        err_msg="Async context __init__",
    )

    async def async_context_create():
        return await LMContext.create(lm, prompt)

    lmcontext = asyncio.run(async_context_create())
    np.testing.assert_allclose(
        lmcontext.next_token_logprobs.cpu(),
        logprobs,
        rtol=5e-4,
        err_msg="Async context create",
    )


def test_observe_impossible_mask_kills_particle():
    # Conditioning on a mask that rules out every token is a zero-probability event.
    # LMTokenMask.log_prob must return -inf (not raise), and Model.observe must
    # finish the particle (weight 0) so it is dropped at the next resample instead of
    # aborting the whole run. Backend-independent, so a fast mock LM with a hand-set,
    # disjoint mask exercises the path deterministically.
    lm = CachedCausalLM.from_pretrained("gpt2", backend="mock")
    ctx = LMContext(lm, "Hello, world!")
    asyncio.run(ctx.mask_dist(lm.token_mask({0, 1, 2})).log_prob(True))
    row_before = ctx.next_token_logprobs
    impossible = ctx.mask_dist(lm.token_mask({3, 4}))  # disjoint from the live mask

    m = Model()
    result = asyncio.run(m.observe(impossible, True))

    assert result is True
    assert m.weight == float("-inf")  # zero-probability observation -> weight 0
    assert m.finished  # ...and the particle is finished
    assert ctx.next_token_logprobs is row_before  # untouched (returned before mutating)


def test_mask_ops_match_numpy_reference():
    # observe(mask, True) renormalizes onto the mask and returns its log-mass;
    # a second mask composes with the first; observing a token returns its
    # log-probability under the live (masked) row.
    lm = CachedCausalLM.from_pretrained("gpt2", backend="mock")
    ctx = LMContext(lm, "Hello, world!")
    row = ctx.next_token_logprobs.cpu().numpy().astype(np.float64)

    async def run():
        m1 = lm.token_mask(range(0, 200))
        m2 = lm.token_mask(range(100, 300))
        lp1 = await ctx.mask_dist(m1).log_prob(True)
        lp2 = await ctx.mask_dist(m2).log_prob(True)
        lp_tok = await ctx.next_token().log_prob(150)
        return lp1, lp2, lp_tok

    lp1, lp2, lp_tok = asyncio.run(run())

    def lse(x):
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))

    ref1 = lse(row[list(range(0, 200))])
    masked = row[list(range(100, 200))]
    ref2 = lse(masked) - ref1
    ref_tok = row[150] - ref1 - ref2
    assert lp1 == pytest.approx(ref1, rel=1e-4)
    assert lp2 == pytest.approx(ref2, rel=1e-4)
    assert lp_tok == pytest.approx(ref_tok, rel=1e-4)
    assert len(ctx.tokens) == len(lm.tokenizer.encode("Hello, world!")) + 1


def test_masks_are_boolean_rows_on_the_model_device():
    lm = CachedCausalLM.from_pretrained("gpt2", backend="mock")
    eos = lm.tokenizer.eos_token_id
    for mask in (lm.masks.EOS, lm.masks.PUNCTUATION, lm.masks.token_length_mask(max=2)):
        assert mask.dtype == torch.bool and mask.shape == (len(lm.str_vocab),)
        assert mask.device == lm.device
    assert lm.masks.EOS[eos] and lm.masks.EOS.sum() == 1
    assert not (lm.masks.PUNCTUATION & ~lm.masks.PUNCTUATION).any()
    assert lm.masks.token_length_mask(max=0)[eos]  # special tokens have length 0
