"""Ordered-prefix compression of cached frozen backbone features.

Modules: ``compression.cache`` (feature cache format), ``compression.extract``
(backbone feature extraction), ``compression.model`` (compressor and reload
API), ``compression.train`` and ``compression.evaluate`` (CLIs).
"""

from compression.model import ChannelStats, CompressorBundle, OrderedCompressor, load_compressor

__all__ = ["ChannelStats", "CompressorBundle", "OrderedCompressor", "load_compressor"]
