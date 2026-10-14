"""Ordered-prefix compression of cached frozen backbone features.

Modules: ``compression.cache`` (feature cache format), ``compression.extract``
(backbone feature extraction), ``compression.model`` (compressor and reload
API), ``compression.train`` and ``compression.evaluate`` (CLIs).

Model exports are loaded on first access so pure-data modules such as
``compression.results`` do not require Torch or backbone dependencies.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compression.model import ChannelStats, CompressorBundle, OrderedCompressor, load_compressor

__all__ = ["ChannelStats", "CompressorBundle", "OrderedCompressor", "load_compressor"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = import_module(".model", __name__)
    value = getattr(model, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
