"""
Requires pytest and pytest-benchmark (pip install pytest pytest-benchmark)

Example usage: pytest benchmark/benchmark_backend.py --benchmark-only --benchmark-group-by=func -v
"""

import asyncio

import pytest
import torch

from examples.haiku import run_example as run_haiku
from examples.hard_constraints import run_example as run_hard_constraints
from llamppl.llms import CachedCausalLM, MLX_AVAILABLE

backends = [
    "hf",
    pytest.param(
        "vllm",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="vLLM backend requires CUDA"
        ),
    ),
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(
            not MLX_AVAILABLE, reason="MLX backend requires MLX-LM"
        ),
    ),
]


@pytest.fixture
def LLM(backend):
    # Set lower gpu_memory_utilization in vllm so that we can fit both models on the GPU
    kwargs = (
        {"engine_opts": {"gpu_memory_utilization": 0.45}, "cache_size": 100}
        if backend == "vllm"
        else {}
    )
    return CachedCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", backend=backend, **kwargs
    )


@pytest.mark.parametrize("backend", backends)
def test_hard_constraints_benchmark(LLM, benchmark, n_particles=20, max_tokens=50):
    def run_with_clear_cache():
        LLM.clear_cache()
        return asyncio.run(
            run_hard_constraints(LLM, max_tokens=max_tokens, n_particles=n_particles)
        )

    # warmup
    run_with_clear_cache()

    benchmark.pedantic(
        run_with_clear_cache,
        iterations=1,
        rounds=3,
    )


@pytest.mark.parametrize("backend", backends)
def test_haiku_benchmark(LLM, benchmark, n_particles=20):
    def run_with_clear_cache():
        LLM.clear_cache()
        return asyncio.run(
            run_haiku(LLM, poem_title="The beauty of testing", n_particles=n_particles)
        )

    # warmup
    run_with_clear_cache()

    benchmark.pedantic(
        run_with_clear_cache,
        iterations=1,
        rounds=3,
    )
