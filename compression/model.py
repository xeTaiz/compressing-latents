"""Ordered-prefix latent compressor for frozen backbone token features.

The encoder maps a variable-length set of source tokens to an ordered sequence
of code vectors.  Code ``i`` only attends to the source and to codes ``<= i``,
so encoding with ``num_codes=k`` yields exactly the first ``k`` codes of the
full encoding.  The decoder reconstructs every requested token from its
position/type query by cross-attending to the physically supplied code prefix;
queries never attend to each other and no source information reaches the
decoder except through the codes.

Only ``torch`` and the standard library are required here, so trained
compressors can be reloaded without any backbone package (see
:func:`load_compressor`).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ATTENTION_MODES = ("separate", "joint")
NUM_TOKEN_TYPES = 4  # 0 patch, 1 CLS, 2 register, 3 camera
CHECKPOINT_FORMAT = "ordered-compressor-checkpoint/v1"
MODEL_FORMAT = "ordered-compressor-model/v1"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FourierPositionEncoding(nn.Module):
    """Deterministic sinusoidal encoding of normalized ``[..., 3]`` coordinates."""

    def __init__(self, num_frequencies: int = 16, max_frequency: float = 64.0) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if max_frequency < 1.0:
            raise ValueError("max_frequency must be >= 1")
        bands = torch.logspace(0.0, math.log2(max_frequency), num_frequencies, base=2.0, dtype=torch.float64)
        self.register_buffer("bands", (bands * math.pi).float(), persistent=False)
        self.output_dim = 3 * 2 * num_frequencies

    def forward(self, positions: Tensor) -> Tensor:
        angles = positions.float()[..., None] * self.bands  # [..., 3, F]
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return encoded.flatten(-2)


class Attention(nn.Module):
    """Multi-head attention with one key/value projection per memory source."""

    def __init__(self, width: int, heads: int, dropout: float, num_memories: int = 1) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by heads {heads}")
        self.heads = heads
        self.dropout = dropout
        self.query = nn.Linear(width, width)
        self.key_value = nn.ModuleList(nn.Linear(width, 2 * width) for _ in range(num_memories))
        self.output = nn.Linear(width, width)

    def forward(self, query: Tensor, memories: list[Tensor], mask: Tensor | None) -> Tensor:
        """``mask`` is boolean, broadcastable to ``[B, Lq, Lk]``; True means attend."""

        batch, length, width = query.shape
        head_dim = width // self.heads
        q = self.query(query).view(batch, length, self.heads, head_dim).transpose(1, 2)
        keys, values = [], []
        for projection, memory in zip(self.key_value, memories, strict=True):
            k, v = projection(memory).chunk(2, dim=-1)
            keys.append(k)
            values.append(v)
        k = torch.cat(keys, dim=1) if len(keys) > 1 else keys[0]
        v = torch.cat(values, dim=1) if len(values) > 1 else values[0]
        k = k.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.heads, head_dim).transpose(1, 2)
        attn_mask = None if mask is None else mask.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.output(out.transpose(1, 2).reshape(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, width: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, multiplier * width),
            nn.GELU(),
            nn.Linear(multiplier * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SeparateEncoderBlock(nn.Module):
    """Causal latent self-attention, then source cross-attention, then FFN."""

    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = Attention(width, heads, dropout)
        self.cross_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.cross_attention = Attention(width, heads, dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = FeedForward(width, ff_multiplier, dropout)

    def forward(self, latents: Tensor, source: Tensor, source_mask: Tensor, causal_mask: Tensor) -> Tensor:
        h = self.self_norm(latents)
        latents = latents + self.self_attention(h, [h], causal_mask)
        latents = latents + self.cross_attention(
            self.cross_norm(latents), [self.source_norm(source)], source_mask
        )
        return latents + self.ff(self.ff_norm(latents))


class JointEncoderBlock(nn.Module):
    """Latents attend to ``concat(source, causally allowed latents)`` in one softmax."""

    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.latent_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads, dropout, num_memories=2)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = FeedForward(width, ff_multiplier, dropout)

    def forward(self, latents: Tensor, source: Tensor, source_mask: Tensor, causal_mask: Tensor) -> Tensor:
        h = self.latent_norm(latents)
        batch = latents.shape[0]
        mask = torch.cat(
            (source_mask.expand(batch, h.shape[1], -1), causal_mask.expand(batch, -1, -1)), dim=-1
        )
        latents = latents + self.attention(h, [self.source_norm(source), h], mask)
        return latents + self.ff(self.ff_norm(latents))


class DecoderBlock(nn.Module):
    """Query cross-attention to the code prefix, then FFN; no query self-attention."""

    def __init__(self, width: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.code_norm = nn.LayerNorm(width)
        self.cross_attention = Attention(width, heads, dropout)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = FeedForward(width, ff_multiplier, dropout)

    def forward(self, queries: Tensor, codes: Tensor) -> Tensor:
        queries = queries + self.cross_attention(self.query_norm(queries), [self.code_norm(codes)], None)
        return queries + self.ff(self.ff_norm(queries))


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class OrderedCompressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int = 512,
        max_codes: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        heads: int = 8,
        ff_multiplier: int = 4,
        attention_mode: str = "separate",
        dropout: float = 0.0,
        position_frequencies: int = 16,
        max_position_frequency: float = 64.0,
    ) -> None:
        super().__init__()
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}, got {attention_mode!r}")
        for name, value in (
            ("input_dim", input_dim),
            ("width", width),
            ("max_codes", max_codes),
            ("encoder_layers", encoder_layers),
            ("decoder_layers", decoder_layers),
            ("heads", heads),
            ("ff_multiplier", ff_multiplier),
        ):
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = {
            "input_dim": int(input_dim),
            "width": int(width),
            "max_codes": int(max_codes),
            "encoder_layers": int(encoder_layers),
            "decoder_layers": int(decoder_layers),
            "heads": int(heads),
            "ff_multiplier": int(ff_multiplier),
            "attention_mode": attention_mode,
            "dropout": float(dropout),
            "position_frequencies": int(position_frequencies),
            "max_position_frequency": float(max_position_frequency),
        }
        self.input_dim = input_dim
        self.width = width
        self.max_codes = max_codes
        self.attention_mode = attention_mode

        self.position_encoding = FourierPositionEncoding(position_frequencies, max_position_frequency)
        # Encoder-side source embedding: feature adapter + position + token type.
        self.source_adapter = nn.Linear(input_dim, width)
        self.source_position = nn.Linear(self.position_encoding.output_dim, width)
        self.source_type = nn.Embedding(NUM_TOKEN_TYPES, width)
        self.code_queries = nn.Parameter(torch.randn(max_codes, width) * 0.02)
        block = SeparateEncoderBlock if attention_mode == "separate" else JointEncoderBlock
        self.encoder = nn.ModuleList(
            block(width, heads, ff_multiplier, dropout) for _ in range(encoder_layers)
        )
        self.code_norm = nn.LayerNorm(width)
        # Decoder-side queries carry only position/type metadata.
        self.query_position = nn.Linear(self.position_encoding.output_dim, width)
        self.query_type = nn.Embedding(NUM_TOKEN_TYPES, width)
        self.decoder = nn.ModuleList(
            DecoderBlock(width, heads, ff_multiplier, dropout) for _ in range(decoder_layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, input_dim)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _check_tokens(positions: Tensor, token_types: Tensor, valid_mask: Tensor | None) -> Tensor:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"positions must be [B,N,3], got {tuple(positions.shape)}")
        if token_types.shape != positions.shape[:2]:
            raise ValueError("token_types must be [B,N] matching positions")
        if valid_mask is None:
            return torch.ones(positions.shape[:2], dtype=torch.bool, device=positions.device)
        if valid_mask.shape != positions.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a bool [B,N] tensor")
        return valid_mask

    def _embed(self, position_proj: nn.Linear, type_embedding: nn.Embedding, positions: Tensor,
               token_types: Tensor, valid: Tensor) -> Tensor:
        # Padding may hold arbitrary values; neutralise it so nothing non-finite
        # can leak through masked (zero-weight) attention.
        positions = torch.where(valid[..., None], positions, torch.zeros_like(positions))
        token_types = torch.where(valid, token_types, torch.zeros_like(token_types))
        return position_proj(self.position_encoding(positions)) + type_embedding(token_types)

    def _resolve_num_codes(self, num_codes: int | None) -> int:
        if num_codes is None:
            return self.max_codes
        if not 1 <= int(num_codes) <= self.max_codes:
            raise ValueError(f"num_codes must be in [1, {self.max_codes}], got {num_codes}")
        return int(num_codes)

    # -- public API --------------------------------------------------------

    def encode(
        self,
        features: Tensor,
        positions: Tensor,
        token_types: Tensor,
        valid_mask: Tensor,
        num_codes: int | None = None,
    ) -> Tensor:
        """Return the first ``num_codes`` ordered codes, ``[B, num_codes, width]``."""

        valid = self._check_tokens(positions, token_types, valid_mask)
        if features.shape[:2] != valid.shape or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must be [B,N,{self.input_dim}], got {tuple(features.shape)}"
            )
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample needs at least one valid source token")
        k = self._resolve_num_codes(num_codes)
        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        source = self.source_adapter(features) + self._embed(
            self.source_position, self.source_type, positions, token_types, valid
        )
        batch = features.shape[0]
        latents = self.code_queries[:k].unsqueeze(0).expand(batch, -1, -1)
        source_mask = valid[:, None, :]  # [B, 1, N]
        causal_mask = torch.ones(k, k, dtype=torch.bool, device=valid.device).tril()[None]
        for block in self.encoder:
            latents = block(latents, source, source_mask, causal_mask)
        return self.code_norm(latents)

    def decode(
        self,
        codes: Tensor,
        positions: Tensor,
        token_types: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Reconstruct ``[B, N, input_dim]`` from a code prefix ``[B, k, width]``.

        Rows where ``valid_mask`` is False are returned as zeros.
        """

        valid = self._check_tokens(positions, token_types, valid_mask)
        if codes.ndim != 3 or codes.shape[-1] != self.width or codes.shape[0] != positions.shape[0]:
            raise ValueError(f"codes must be [B,k,{self.width}], got {tuple(codes.shape)}")
        if not 1 <= codes.shape[1] <= self.max_codes:
            raise ValueError(f"code prefix length must be in [1, {self.max_codes}]")
        queries = self._embed(self.query_position, self.query_type, positions, token_types, valid)
        for block in self.decoder:
            queries = block(queries, codes)
        output = self.output_head(self.output_norm(queries))
        return torch.where(valid[..., None], output, torch.zeros_like(output))

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        token_types: Tensor,
        valid_mask: Tensor,
        num_codes: int | None = None,
    ) -> Tensor:
        codes = self.encode(features, positions, token_types, valid_mask, num_codes)
        return self.decode(codes, positions, token_types, valid_mask)


# ---------------------------------------------------------------------------
# Fixed per-channel statistics
# ---------------------------------------------------------------------------


@dataclass
class ChannelStats:
    """Train-split per-channel mean/std in native feature space (float64)."""

    mean: Tensor
    std: Tensor
    token_count: int
    sample_count: int
    split: str
    cache_identity: str
    layer: int | None
    std_floor: float
    max_samples: int | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.mean = self.mean.detach().to("cpu", torch.float64)
        self.std = self.std.detach().to("cpu", torch.float64)
        if self.mean.ndim != 1 or self.mean.shape != self.std.shape:
            raise ValueError("mean/std must be matching 1-D tensors")
        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()):
            raise ValueError("channel statistics must be finite")
        if not bool((self.std > 0).all()):
            raise ValueError("channel std must be positive")

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def _on(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        key = (reference.device, torch.float32)
        if key not in self._cache:
            self._cache[key] = (
                self.mean.to(reference.device, torch.float32),
                self.std.to(reference.device, torch.float32),
            )
        return self._cache[key]

    def normalize(self, features: Tensor) -> Tensor:
        mean, std = self._on(features)
        return (features.float() - mean) / std

    def denormalize(self, normalized: Tensor) -> Tensor:
        mean, std = self._on(normalized)
        return normalized.float() * std + mean

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.mean.numpy().tobytes())
        h.update(self.std.numpy().tobytes())
        return h.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.clone(),
            "std": self.std.clone(),
            "token_count": int(self.token_count),
            "sample_count": int(self.sample_count),
            "split": self.split,
            "cache_identity": self.cache_identity,
            "layer": self.layer,
            "std_floor": float(self.std_floor),
            "max_samples": self.max_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelStats":
        return cls(
            mean=data["mean"],
            std=data["std"],
            token_count=int(data["token_count"]),
            sample_count=int(data["sample_count"]),
            split=str(data["split"]),
            cache_identity=str(data["cache_identity"]),
            layer=None if data["layer"] is None else int(data["layer"]),
            std_floor=float(data["std_floor"]),
            max_samples=None if data.get("max_samples") is None else int(data["max_samples"]),
        )


# ---------------------------------------------------------------------------
# Persistence / reload
# ---------------------------------------------------------------------------


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_model(config: dict[str, Any]) -> OrderedCompressor:
    return OrderedCompressor(**config)


@dataclass
class CompressorBundle:
    """A reloaded compressor with its fixed statistics and provenance."""

    model: OrderedCompressor
    stats: ChannelStats
    metadata: dict[str, Any]

    @property
    def cache_identity(self) -> str:
        return self.metadata["cache"]["identity"]

    def to(self, device: torch.device | str) -> "CompressorBundle":
        self.model.to(device)
        return self

    def encode_native(self, features: Tensor, positions: Tensor, token_types: Tensor,
                      valid_mask: Tensor, num_codes: int | None = None) -> Tensor:
        return self.model.encode(self.stats.normalize(features), positions, token_types, valid_mask, num_codes)

    def decode_native(self, codes: Tensor, positions: Tensor, token_types: Tensor,
                      valid_mask: Tensor | None = None) -> Tensor:
        normalized = self.model.decode(codes, positions, token_types, valid_mask)
        native = self.stats.denormalize(normalized)
        if valid_mask is not None:
            native = torch.where(valid_mask[..., None], native, torch.zeros_like(native))
        return native


def export_compressor(path: Path, model: OrderedCompressor, stats: ChannelStats,
                      metadata: dict[str, Any]) -> None:
    """Write a model-only file loadable with :func:`load_compressor`."""

    payload = {
        "format": MODEL_FORMAT,
        "model_config": dict(model.config),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "stats": stats.to_dict(),
        **metadata,
    }
    atomic_torch_save(payload, path)


def load_compressor(path: Path | str, map_location: str | torch.device = "cpu") -> CompressorBundle:
    """Load an exported model or a full training checkpoint for inference."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") not in (MODEL_FORMAT, CHECKPOINT_FORMAT):
        raise ValueError(f"{path} is not an ordered-compressor model or checkpoint")
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(map_location).eval()
    stats = ChannelStats.from_dict(payload["stats"])
    if stats.feature_dim != model.input_dim:
        raise ValueError("stored statistics do not match the model input dimension")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("model_state", "optimizer_state", "stats", "rng")
    }
    return CompressorBundle(model=model, stats=stats, metadata=metadata)
