"""Utilities for working with language models."""

import string
import warnings

import torch
from genlm.backend.llm import AsyncTransformer
from genlm.backend.llm import AsyncVirtualLM
from genlm.backend.llm import MockAsyncLM
from genlm.backend.llm import AsyncMlxLM

VLLM_AVAILABLE = True
try:
    import vllm
except ImportError:
    VLLM_AVAILABLE = False

MLX_AVAILABLE = True
try:
    import mlx_lm
except ImportError:
    MLX_AVAILABLE = False

warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("once", category=RuntimeWarning)


MID_PUNCTUATION_CHARS = (",", ":", ";", "-", '"')
END_PUNCTUATION_CHARS = (".", "!", "?")


class Masks:
    """``[V]`` boolean masks over ``lm.str_vocab``, on ``lm.device``. Combine them with
    ``|``, ``&`` and ``~``; ``lm.token_mask(ids)`` builds one from token ids."""

    def __init__(self, lm):
        vocab = lm.str_vocab
        self.STARTS_NEW_WORD = lm.token_mask(
            i
            for (i, v) in enumerate(vocab)
            if v[0] == " "
            and len(v) > 1
            and v[1] not in string.whitespace
            and v[1] not in string.punctuation
        )
        self.CONTINUES_CURRENT_WORD = lm.token_mask(
            i for (i, v) in enumerate(vocab) if all(c in "'" or c.isalpha() for c in v)
        )
        self.MID_PUNCTUATION = lm.token_mask(
            i for (i, v) in enumerate(vocab) if v in MID_PUNCTUATION_CHARS
        )
        self.END_PUNCTUATION = lm.token_mask(
            i for (i, v) in enumerate(vocab) if v in END_PUNCTUATION_CHARS
        )
        self.PUNCTUATION = self.MID_PUNCTUATION | self.END_PUNCTUATION
        self.CONTAINS_WHITESPACE = lm.token_mask(
            i for (i, v) in enumerate(vocab) if any(c in string.whitespace for c in v)
        )
        self.EOS = lm.token_mask([lm.tokenizer.eos_token_id])
        special = set(lm.tokenizer.all_special_ids)  # special tokens have length 0
        self._token_lengths = torch.tensor(
            [0 if i in special else len(v) for (i, v) in enumerate(vocab)],
            device=lm.device,
        )

    def token_length_mask(self, min: int = None, max: int = None):
        """Tokens whose string length lies in ``[min, max]``."""
        lengths = self._token_lengths
        if min is None and max is None:
            return torch.ones_like(lengths, dtype=torch.bool)
        if min is None:
            return lengths <= max
        if max is None:
            return lengths >= min
        return (lengths >= min) & (lengths <= max)


class TokenSequence:
    """A sequence of tokens.

    Supports addition (via `+` or mutating `+=`) with:

    * other `TokenSequence` instances (concatenation)
    * individual tokens, represented as integers or `Token` instances
    * strings, which are tokenized by `lm.tokenizer`

    Attributes:
        lm (llamppl.llms.CachedCausalLM): the language model whose vocabulary the tokens come from.
        seq (list[llamppl.llms.Token]): the sequence of tokens."""

    def __init__(self, lm, seq=None):
        """Create a `TokenSequence` from a language model and a sequence.

        Args:
            lm (llamppl.llms.CachedCausalLM): the language model whose vocabulary the tokens come from.
            seq (str | list[int]): the sequence of token ids, or a string which will be automatically tokenized. Defaults to the singleton sequence containing a bos token.
        """
        self.lm = lm
        if seq is None:
            self.seq = [lm.tokenizer.bos_token_id]
        elif isinstance(seq, str):
            self.seq = self.lm.tokenizer.encode(seq)
        else:
            self.seq = seq

    def __str__(self):
        return self.lm.tokenizer.decode(self.seq)

    def __iadd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            self.seq.append(other.token_id)
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            self.seq.extend(other.seq)
        elif isinstance(other, str):
            self.seq.extend(self.lm.tokenizer.encode(other, add_special_tokens=False))
        elif isinstance(other, int):
            self.seq.append(other)
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")
        return self

    def __radd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            return TokenSequence(self.lm, [other.token_id, *self.seq])
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            return TokenSequence(self.lm, other.seq + self.seq)
        elif isinstance(other, str):
            return TokenSequence(
                self.lm,
                self.lm.tokenizer.encode(other, add_special_tokens=False) + self.seq,
            )
        elif isinstance(other, int):
            return TokenSequence(self.lm, [other, *self.seq])
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")

    def __add__(self, other):
        s = TokenSequence(self.lm, self.seq)
        s += other
        return s


class Token:
    """Class representing a token.

    Attributes:
        lm (llamppl.llms.CachedCausalLM): the language model for which this is a Token.
        token_id (int): the integer token id (an index into the vocabulary).
        token_str (str): a string, which the token represents—equal to `lm.str_vocab[token_id]`.
    """

    def __init__(self, lm, token_id, token_str):
        self.lm = lm
        self.token_id = token_id
        self.token_str = token_str

    # Adding tokens
    def __add__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        s += other
        return s

    def __radd__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        return other + s

    # Support checking for EOS
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.lm is other.lm and self.token_id == other.token_id
        elif isinstance(other, int):
            return self.token_id == other
        else:
            return self.token_str == other

    def __int__(self):
        return self.token_id

    def __str__(self):
        return self.token_str

    def __repr__(self):
        return f"<{self.token_str}|{self.token_id}>"


class CachedCausalLM:
    """Wrapper around a [`genlm.backend.llm.AsyncLM`](https://genlm.org/genlm-backend/reference/genlm/backend/llm/__init__/).

    Attributes:
        model (genlm_backend.llm.AsyncLM): The underlying language model (either `AsyncVirtualLM` or `AsyncTransformer`).
        str_vocab (list[str]): List mapping token IDs to their string representations.
        byte_vocab (list[bytes]): List mapping token IDs to their byte representations.
        device (torch.device): Where next-token rows and masks live.
        masks (Masks): Token masks for filtering logits during generation.
    """

    @classmethod
    def from_pretrained(cls, model_id, backend=None, **kwargs):
        """Create a CachedCausalLM from a HuggingFace model name.

        This is a convenience method that instantiates the underlying `AsyncLM` from a HuggingFace model name.

        Args:
            model_id (str): Name or path of the HuggingFace pretrained model to load.
            backend (str, optional): `AsyncLM` backend to use:
                - 'vllm' to instantiate an `AsyncVirtualLM`; ideal for GPU usage
                - 'hf' for an `AsyncTransformer`; ideal for CPU usage
                - 'mock' for a `MockAsyncLM`; ideal for testing.
                - 'mlx' for an `AsyncMlxLM`; ideal for usage on devices with Apple silicon.
                Defaults to 'vllm' if CUDA is available, otherwise 'hf'.
            **kwargs: Additional keyword arguments passed to the `AsyncLM` constructor.
                See [`AsyncLM` documentation](https://probcomp.github.io/genlm-backend/reference/genlm_backend/llm/__init__/).

        Returns:
            CachedCausalLM: The llamppl-compatible interface to the `AsyncLM` model.
        """
        backend = backend or (
            "vllm" if (torch.cuda.is_available() and VLLM_AVAILABLE) else "hf"
        )

        if backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ValueError(
                    "vLLM backend requested but vLLM is not installed. "
                    "Please install vLLM with `pip install vllm`."
                )
            model_cls = AsyncVirtualLM
        elif backend == "hf":
            model_cls = AsyncTransformer
        elif backend == "mock":
            model_cls = MockAsyncLM
        elif backend == "mlx":
            model_cls = AsyncMlxLM
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Must be one of ['vllm', 'hf', 'mock', 'mlx']"
            )

        # Handle legacy auth_token parameter. The ability to pass in the auth_token should
        # be removed in a future version since it is not supported by the vllm backend.
        # Users should authenticate with the HuggingFace CLI.
        auth_token = kwargs.pop("auth_token", None)
        if auth_token:
            if backend == "vllm":
                raise ValueError(
                    "Explicitly passing auth_token is not compatible with the vLLM AsyncLM backend. "
                    "Authenticate using `huggingface-cli login` instead."
                )

            if "hf_opts" not in kwargs:
                kwargs["hf_opts"] = {}
            kwargs["hf_opts"]["token"] = auth_token

            warnings.warn(
                "Passing auth_token directly is deprecated and will be removed in a future version. "
                "Please authenticate using `huggingface-cli login` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        load_in_8bit = kwargs.pop("load_in_8bit", False)
        if load_in_8bit:
            if "bitsandbytes_opts" not in kwargs:
                kwargs["bitsandbytes_opts"] = {}
            kwargs["bitsandbytes_opts"]["load_in_8bit"] = True

            warnings.warn(
                "load_in_8bit is deprecated and will be removed in a future version. "
                "Please pass `bitsandbytes_opts` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        model = model_cls.from_name(model_id, **kwargs)

        return cls(model)

    def __init__(self, model):
        """
        Create a `CachedCausalLM` from an `AsyncLM`.

        Args:
            model (genlm_backend.llm.AsyncLM): an `AsyncLM` instance.
        """
        if isinstance(model, AsyncVirtualLM):
            self.backend = "vllm"
        elif isinstance(model, AsyncTransformer):
            self.backend = "hf"
        elif isinstance(model, MockAsyncLM):
            self.backend = "mock"
        elif isinstance(model, AsyncMlxLM):
            self.backend = "mlx"
        else:
            raise ValueError(
                f"Unknown model type: {type(model)}. Must be one of [AsyncVirtualLM, AsyncTransformer, MockAsyncLM, AsyncMlxLM]"
            )

        self.model = model
        self.tokenizer = model.tokenizer
        self.str_vocab = model.str_vocab
        self.byte_vocab = model.byte_vocab
        self.device = model.device
        self.masks = Masks(self)

    @property
    def vocab(self):
        """Legacy accessor for string vocabulary. Prefer using `.str_vocab` directly for access to the model's string vocabulary."""
        warnings.warn(
            "Accessing .vocab directly is deprecated and will be removed in a future version. Use .str_vocab or .byte_vocab instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model.str_vocab

    def __deepcopy__(self, memo):
        return self

    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token. This version is asynchronous and support auto batching of concurrent requests; use with `await`.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.

        Returns:
            logprobs (torch.Tensor): a `[len(str_vocab)]` tensor on `self.device` with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        return await self.model.next_token_logprobs(token_ids)

    def next_token_logprobs_unbatched(self, token_ids):
        """Request log probabilities of next token. Not asynchronous, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids, representing a prompt to the language model.

        Returns:
            logprobs (torch.Tensor): a `[len(str_vocab)]` tensor on `self.device` with the language model's log (normalized) probabilities for the next token following the prompt.
        """
        return self.model.next_token_logprobs_sync(token_ids)

    def token_mask(self, token_ids):
        """A `[len(str_vocab)]` boolean mask on `self.device`, `True` on `token_ids`.

        Args:
            token_ids (iterable[int]): the token ids inside the mask.

        Returns:
            (torch.Tensor): the mask. Combine masks with `|`, `&` and `~`.
        """
        mask = torch.zeros(len(self.str_vocab), dtype=torch.bool)
        mask[list(token_ids)] = True
        return mask.to(self.device)

    def clear_cache(self):
        """Clear the cache of log probabilities and key/value pairs.

        For HuggingFace backend: Clears both logprob cache and KV cache.

        For vLLM backend: Only clears logprob cache (KV cache is managed internally by vLLM).
        """
        self.model.clear_cache()

    def clear_kv_cache(self):
        """Clear any key and value vectors from the cache."""
        if self.backend == "hf":
            self.model.clear_kv_cache()
        elif self.backend == "mlx":
            self.model.clear_cache()
        elif self.backend == "vllm":
            warnings.warn(
                "clear_kv_cache() is only supported for the HuggingFace backend. The KV cache for the vLLM backend is handled internally by vLLM. No operation performed.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.backend == "mock":
            pass
        else:
            raise RuntimeError(
                f"clear_kv_cache() is not implemented for backend type {type(self.model)}"
            )

    def reset_async_queries(self):
        """Clear any pending language model queries from the queue."""
        if self.backend == "hf":
            self.model.reset_async_queries()
        elif self.backend in ["vllm", "mlx"]:
            warnings.warn(
                "reset_async_queries() is only supported for the HuggingFace backend. No operation performed.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.backend == "mock":
            pass
        else:
            raise RuntimeError(
                f"reset_async_queries() is not implemented for backend type {type(self.model)}"
            )

    def cache_kv(self, prompt_tokens):
        """Cache the key and value vectors for a prompt.

        Args:
            prompt_tokens (list[int]): token ids for the prompt to cache.
        """
        if self.backend == "hf":
            self.model.cache_kv(prompt_tokens)
        elif self.backend in ["vllm", "mlx"]:
            warnings.warn(
                "cache_kv() is only supported for the HuggingFace backend. The vLLM and MLX backends manage their own KV cache. No operation performed.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif self.backend == "mock":
            pass
        else:
            raise RuntimeError(
                f"cache_kv() is not implemented for backend type {type(self.model)}"
            )
