"""Probabilistic programming with HuggingFace Transformer models."""

from .chunks import *
from .distributions import *
from .inference import *
from .llms import *
from .modeling import *
from .util import *

import warnings

warnings.warn(
    "The 'hfppl' library is deprecated and will be renamed to 'llamppl'. "
    "Please migrate to 'llamppl'. This is the final release of 'hfppl'.",
    DeprecationWarning,
)
