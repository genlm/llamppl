# Developer's Guide

This guide describes how to set up `llamppl` for local development, run the
tests, and work with CI.

## Local Installation

Clone the repository:

```bash
git clone git@github.com:genlm/llamppl.git
cd llamppl
```

Create a virtual environment (for example, with `uv`) and install `llamppl`
with its development dependencies:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,examples]"
```

> You may need to install `uv` first: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

For the MLX backend (Apple Silicon):

```bash
uv pip install -e ".[mlx]"
```

For the vLLM (GPU) backend — this pulls `genlm-backend[vllm]`, which keeps the
vLLM and `triton` versions aligned with `genlm-backend`:

```bash
uv pip install -e ".[vllm]"
```

## Testing

The test suite imports the `examples/` package, so the project root must be on
`PYTHONPATH`:

```bash
PYTHONPATH="$PYTHONPATH:$(pwd)" pytest tests
```

With coverage:

```bash
PYTHONPATH="$PYTHONPATH:$(pwd)" pytest tests --cov=llamppl --cov-report=term-missing
```

## GPU tests & CI

Some tests require a CUDA GPU — the vLLM backend tests, parametrized as
`[vllm]`. They are **not** exercised by a plain `pytest tests` on a CPU-only
machine.

**Running GPU tests locally.** On a machine with a CUDA GPU, install the vLLM
extra and run the suite (this mirrors CI in `.github/workflows/tests.yml`):

```bash
uv pip install -e ".[vllm]"
PYTHONPATH="$PYTHONPATH:$(pwd)" pytest tests
```

**How GPU tests run in CI.** The `build` job in `.github/workflows/tests.yml`
runs on the self-hosted **`gpu-runners`** runner group. On every pull request
and push to `main` it runs **by default**.

- **Opt out with the `skip-gpu-tests` label.** A maintainer (triage/write
  access) can add the `skip-gpu-tests` label to a PR to skip the GPU `build`
  job — e.g. for a docs-only change, or when you've already run the suite
  locally on a cloud GPU. External/fork contributors can't add labels, so GPU
  tests always run for their PRs.
- **`gpu-gate` is the required check.** It's a small aggregator job that passes
  when `build` succeeds *or* is intentionally skipped, and fails only if it
  fails or is cancelled. Branch protection requires `gpu-gate` (not `build`
  directly), so a labeled-skip or a queue-stuck run never blocks a merge with a
  dangling `cancelled` check.
- **Queue watchdog.** A scheduled `gpu-queue-watchdog` workflow cancels GPU runs
  stuck in `queued` for more than 20 minutes (e.g. if no runner is available),
  so a PR fails fast instead of hanging at GitHub's 24-hour queue limit. If that
  happens: re-run the job once a runner is free, or apply `skip-gpu-tests`.

**Recommended maintainer flow:** make your change → if it touches GPU paths and
you have a GPU, run the GPU tests locally → open the PR → add `skip-gpu-tests`
if you've validated locally or the change is GPU-irrelevant, otherwise let CI
run it on `gpu-runners`.

## Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage git pre-commit hooks
(formatting, linting, etc.), defined in `.pre-commit-config.yaml`. Install
`pre-commit` (e.g. `uv tool install pre-commit` or `pipx install pre-commit`),
then install the hooks:

```bash
pre-commit install
```

They'll then run on every commit. To run them manually:

```bash
pre-commit run --all-files
```

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/). To build it:

```bash
mkdocs build
```

To serve it locally:

```bash
mkdocs serve
```
