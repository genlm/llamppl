import asyncio

import pytest
import torch

from examples.haiku import run_example as run_haiku
from examples.hard_constraints import run_example as run_hard_constraints
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
def LLM(backend):
    # Set lower gpu_memory_utilization in vllm so that we can fit both models on the GPU
    kwargs = (
        {"engine_opts": {"gpu_memory_utilization": 0.45}}
        if backend == "vllm"
        else {"cache_size": 10}
        if backend == "mlx"
        else {}
    )
    return CachedCausalLM.from_pretrained("gpt2", backend=backend, **kwargs)


@pytest.mark.parametrize("backend", backends)
def test_hard_constraints(LLM, n_particles=5, max_tokens=25):
    particles = asyncio.run(
        run_hard_constraints(LLM, max_tokens=max_tokens, n_particles=n_particles)
    )
    assert len(particles) == n_particles

    particles = asyncio.run(
        run_hard_constraints(
            LLM,
            max_tokens=max_tokens,
            n_particles=n_particles,
            resampling_method="stratified",
        )
    )
    assert len(particles) == n_particles


@pytest.mark.parametrize("backend", backends)
def test_haiku(LLM, n_particles=5):
    particles = asyncio.run(
        run_haiku(LLM, poem_title="The beauty of testing", n_particles=n_particles)
    )
    assert len(particles) == n_particles

    particles = asyncio.run(
        run_haiku(
            LLM,
            poem_title="The beauty of testing",
            n_particles=n_particles,
            resampling_method="stratified",
        )
    )
    assert len(particles) == n_particles
