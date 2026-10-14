"""Train an ordered compressor on cached frozen backbone features.

Two explicit stages share one run directory::

    python -m compression.train --config configs/compression/dinov3.json --stage fixed \
        --cache-dir CACHE --run-dir RUN
    python -m compression.train --config configs/compression/dinov3.json --stage prefix \
        --cache-dir CACHE --run-dir RUN          # initialises from RUN/fixed/model.pt

``fixed`` is the fixed-budget control (distortion at one budget only).  ``prefix``
optimises ``full_weight * D_K + (1 - full_weight) * D_k`` with ``k`` drawn from the
short-budget distribution each step; with the defaults this is an unbiased
estimate of ``0.5 * mean_{k<K} D_k + 0.5 * D_K``.  The model is encoded once at
``K`` and exact code prefixes are sliced for the short decode.

Each stage writes ``RUN/<stage>/{config.json,train_log.jsonl,checkpoint_latest.pt,model.pt}``;
train-split channel statistics are computed once into ``RUN/channel_stats.pt``.
A stage never overwrites an existing checkpoint unless ``--resume`` is given and
the stored run is compatible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from compression.cache import FeatureCacheDataset, collate_features
from compression.model import (
    CHECKPOINT_FORMAT,
    ChannelStats,
    OrderedCompressor,
    atomic_torch_save,
    build_model,
    export_compressor,
    load_compressor,
)

STAGES = ("fixed", "prefix")
PRECISIONS = ("bf16", "fp32")
STATS_FORMAT = "ordered-compressor-channel-stats/v1"

DEFAULT_CONFIG: dict[str, Any] = {
    "name": None,
    "feature_dim": None,
    "data": {
        "model": None,
        "identity": None,
        "layer": None,
        "train_split": "train",
        "eval_split": "validation",
        "token_budget": 32768,
        "num_workers": 4,
    },
    "model": {
        "width": 512,
        "max_codes": 256,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "heads": 8,
        "ff_multiplier": 4,
        "attention_mode": "separate",
        "dropout": 0.0,
        "position_frequencies": 16,
        "max_position_frequency": 64.0,
    },
    "stats": {"max_samples": None, "std_floor": 1e-6},
    "optim": {
        "lr": 1e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "warmup_steps": 500,
    },
    "precision": "bf16",
    "seed": 0,
    "stages": {
        "fixed": {"max_steps": 10000, "budget": None},
        "prefix": {
            "max_steps": 20000,
            "full_weight": 0.5,
            "short_budgets": None,
            "short_weights": None,
            "init_from": "fixed",
        },
    },
    "checkpoint_every": 1000,
    "log_every": 50,
    "eval": {"budgets": None},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _merge(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        path = f"{prefix}{key}"
        if key not in base:
            raise KeyError(f"unknown config key {path!r}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"config key {path!r} must be an object")
            merged[key] = _merge(base[key], value, f"{path}.")
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_override(config: dict[str, Any], assignment: str) -> None:
    """Apply ``dotted.key=JSON`` (bare strings allowed) to an existing key."""

    if "=" not in assignment:
        raise ValueError(f"override {assignment!r} must look like key=value")
    dotted, raw = assignment.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"unknown config section {dotted!r}")
        node = node[key]
    if keys[-1] not in node or isinstance(node[keys[-1]], dict):
        raise KeyError(f"unknown config value {dotted!r}")
    node[keys[-1]] = value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer, got {value!r}")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("config 'name' is required")
    _positive_int(config["feature_dim"], "feature_dim")
    data = config["data"]
    _positive_int(data["token_budget"], "data.token_budget")
    _positive_int(data["num_workers"], "data.num_workers", allow_zero=True)
    if data["layer"] is not None:
        _positive_int(data["layer"], "data.layer", allow_zero=True)
    model = config["model"]
    for key in ("width", "max_codes", "encoder_layers", "decoder_layers", "heads", "ff_multiplier",
                "position_frequencies"):
        _positive_int(model[key], f"model.{key}")
    if model["width"] % model["heads"]:
        raise ValueError("model.width must be divisible by model.heads")
    if model["attention_mode"] not in ("separate", "joint"):
        raise ValueError("model.attention_mode must be 'separate' or 'joint'")
    if config["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    _positive_int(config["seed"], "seed", allow_zero=True)
    _positive_int(config["checkpoint_every"], "checkpoint_every")
    _positive_int(config["log_every"], "log_every")
    optim = config["optim"]
    if not optim["lr"] > 0 or optim["grad_clip"] is not None and not optim["grad_clip"] > 0:
        raise ValueError("optim.lr and optim.grad_clip must be positive")
    _positive_int(optim["warmup_steps"], "optim.warmup_steps", allow_zero=True)
    if config["stats"]["max_samples"] is not None:
        _positive_int(config["stats"]["max_samples"], "stats.max_samples")
    if not config["stats"]["std_floor"] > 0:
        raise ValueError("stats.std_floor must be positive")
    for stage in STAGES:
        _positive_int(config["stages"][stage]["max_steps"], f"stages.{stage}.max_steps", allow_zero=True)
    if config["stages"]["prefix"]["init_from"] not in ("fixed", None):
        raise ValueError("stages.prefix.init_from must be 'fixed' or null")
    for stage in STAGES:  # resolves and validates the objectives
        build_objective(config, stage)
    if config["eval"]["budgets"] is not None:
        resolve_eval_budgets(config["eval"]["budgets"], model["max_codes"])
    return config


def load_config(path: Path | str | None, overrides: list[str] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    config = _merge(DEFAULT_CONFIG, raw)
    for assignment in overrides or ():
        apply_override(config, assignment)
    return validate_config(config)


def resolve_eval_budgets(budgets: list[int] | None, max_codes: int) -> list[int]:
    if budgets is None:
        resolved = [1 << i for i in range(max_codes.bit_length()) if 1 << i <= max_codes]
        if resolved[-1] != max_codes:
            resolved.append(max_codes)
        return resolved
    resolved = sorted(set(int(b) for b in budgets))
    if not resolved or resolved[0] < 1 or resolved[-1] > max_codes:
        raise ValueError(f"budgets must lie in [1, {max_codes}], got {budgets}")
    return resolved


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Objective:
    """Training objective with explicitly recorded prefix weights."""

    kind: str  # "fixed" | "sampled_prefix"
    encode_budget: int
    full_weight: float
    short_budgets: tuple[int, ...]
    short_probabilities: tuple[float, ...]

    def expected_weights(self) -> dict[int, float]:
        """Weight of each ``D_k`` in the expected per-step loss (sums to 1)."""

        weights = {self.encode_budget: self.full_weight}
        for budget, probability in zip(self.short_budgets, self.short_probabilities):
            weights[budget] = weights.get(budget, 0.0) + (1.0 - self.full_weight) * probability
        return dict(sorted(weights.items()))

    def sample_short_budget(self, seed: int, step: int) -> int | None:
        """Deterministic in ``(seed, step)`` so resumed runs draw the same budgets."""

        if not self.short_budgets:
            return None
        rng = np.random.default_rng([seed, 0x50524546, step])
        index = rng.choice(len(self.short_budgets), p=np.asarray(self.short_probabilities))
        return self.short_budgets[int(index)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "encode_budget": self.encode_budget,
            "full_weight": self.full_weight,
            "short_budgets": list(self.short_budgets),
            "short_probabilities": list(self.short_probabilities),
            "expected_prefix_weights": {str(k): v for k, v in self.expected_weights().items()},
        }


def build_objective(config: dict[str, Any], stage: str) -> Objective:
    max_codes = config["model"]["max_codes"]
    if stage == "fixed":
        budget = config["stages"]["fixed"]["budget"]
        budget = max_codes if budget is None else budget
        _positive_int(budget, "stages.fixed.budget")
        if budget > max_codes:
            raise ValueError("stages.fixed.budget cannot exceed model.max_codes")
        return Objective("fixed", budget, 1.0, (), ())
    if stage != "prefix":
        raise ValueError(f"stage must be one of {STAGES}")
    spec = config["stages"]["prefix"]
    full_weight = float(spec["full_weight"])
    if max_codes == 1:
        # Only one prefix exists: the objective degenerates to D_1.
        return Objective("sampled_prefix", 1, 1.0, (), ())
    if not 0.0 < full_weight <= 1.0:
        raise ValueError("stages.prefix.full_weight must be in (0, 1]")
    if full_weight == 1.0:
        return Objective("sampled_prefix", max_codes, 1.0, (), ())
    budgets = spec["short_budgets"]
    budgets = list(range(1, max_codes)) if budgets is None else list(budgets)
    for budget in budgets:
        _positive_int(budget, "stages.prefix.short_budgets[]")
    if len(set(budgets)) != len(budgets) or not budgets or max(budgets) >= max_codes:
        raise ValueError("short_budgets must be unique, non-empty and below model.max_codes")
    weights = spec["short_weights"]
    weights = [1.0] * len(budgets) if weights is None else [float(w) for w in weights]
    if len(weights) != len(budgets) or not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError("short_weights must be positive and match short_budgets")
    order = sorted(range(len(budgets)), key=budgets.__getitem__)
    total = math.fsum(weights)
    return Objective(
        "sampled_prefix",
        max_codes,
        full_weight,
        tuple(budgets[i] for i in order),
        tuple(weights[i] / total for i in order),
    )


def masked_sample_mse(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Per-sample mean squared error over real tokens and channels, ``[B]``."""

    error = (prediction.float() - target.float()).square().mean(dim=-1)
    error = torch.where(valid_mask, error, torch.zeros_like(error))
    return error.sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)


def objective_loss(
    model: OrderedCompressor,
    objective: Objective,
    normalized: Tensor,
    positions: Tensor,
    token_types: Tensor,
    valid_mask: Tensor,
    short_budget: int | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Encode once at the objective budget and decode the full and short prefix."""

    codes = model.encode(normalized, positions, token_types, valid_mask, num_codes=objective.encode_budget)
    full = masked_sample_mse(model.decode(codes, positions, token_types, valid_mask), normalized, valid_mask).mean()
    if short_budget is None:
        return full, {"d_full": full}
    if not 1 <= short_budget < objective.encode_budget:
        raise ValueError(f"short budget {short_budget} outside [1, {objective.encode_budget})")
    short = masked_sample_mse(
        model.decode(codes[:, :short_budget], positions, token_types, valid_mask), normalized, valid_mask
    ).mean()
    loss = objective.full_weight * full + (1.0 - objective.full_weight) * short
    return loss, {"d_full": full, "d_short": short}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def plan_token_batches(
    token_counts: list[int], token_budget: int, *, seed: int = 0, epoch: int = 0, shuffle: bool = True
) -> list[list[int]]:
    """Deterministic batches whose padded size ``len(batch) * max(N)`` fits the budget.

    A single sample larger than the budget forms its own batch.
    """

    counts = np.asarray(token_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or (counts < 1).any():
        raise ValueError("token_counts must be a non-empty list of positive integers")
    order = np.random.default_rng([seed, epoch]).permutation(counts.size) if shuffle else np.arange(counts.size)
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in order.tolist():
        tokens = int(counts[index])
        if current and max(longest, tokens) * (len(current) + 1) > token_budget:
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, tokens)
    batches.append(current)
    return batches


class BatchStream:
    """Endless epoch-ordered token-budget batches with a resumable cursor.

    ``epoch``/``batch`` always point at the next unconsumed batch.
    """

    def __init__(self, dataset: Any, token_budget: int, *, seed: int, epoch: int = 0, batch: int = 0,
                 num_workers: int = 0, pin_memory: bool = False) -> None:
        self.dataset = dataset
        self.token_counts = list(dataset.token_counts)
        self.token_budget = token_budget
        self.seed = seed
        self.epoch = epoch
        self.batch = batch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def batches_per_epoch(self, epoch: int) -> int:
        return len(plan_token_batches(self.token_counts, self.token_budget, seed=self.seed, epoch=epoch))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            plan = plan_token_batches(self.token_counts, self.token_budget, seed=self.seed, epoch=self.epoch)
            if self.batch < len(plan):
                # Iterator creation draws a worker base seed even with zero
                # workers. Isolate that draw from the model's saved dropout RNG
                # so restarting an iterator mid-epoch cannot change training.
                loader_rng = torch.Generator().manual_seed(self.seed + self.epoch)
                loader = DataLoader(
                    self.dataset,
                    batch_sampler=plan[self.batch:],
                    collate_fn=collate_features,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    generator=loader_rng,
                )
                for item in loader:
                    self.batch += 1
                    yield item
            self.epoch += 1
            self.batch = 0


def select_stat_indices(num_samples: int, max_samples: int | None) -> list[int]:
    if max_samples is None or max_samples >= num_samples:
        return list(range(num_samples))
    return sorted(set(np.linspace(0, num_samples - 1, max_samples).round().astype(np.int64).tolist()))


def compute_channel_stats(
    dataset: Any,
    *,
    split: str,
    layer: int | None,
    std_floor: float,
    max_samples: int | None = None,
    num_workers: int = 0,
    log: Callable[[str], None] | None = None,
) -> ChannelStats:
    """Streaming float64 per-channel mean/std over every real token of ``dataset``.

    Samples are merged with Chan's parallel update, so memory is ``O(D)`` plus
    one sample regardless of dataset size.
    """

    dim = int(dataset.feature_dim)
    indices = select_stat_indices(len(dataset), max_samples)
    loader = DataLoader(Subset(dataset, indices), batch_size=None, shuffle=False, num_workers=num_workers)
    count = 0
    mean = torch.zeros(dim, dtype=torch.float64)
    m2 = torch.zeros(dim, dtype=torch.float64)
    for seen, sample in enumerate(loader, 1):
        x = sample["features"].to(torch.float64)
        if x.ndim != 2 or x.shape[1] != dim or x.shape[0] == 0:
            raise ValueError(f"sample {sample['sample_id']} has features of shape {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f"sample {sample['sample_id']} contains non-finite features")
        n_b = x.shape[0]
        mean_b = x.mean(dim=0)
        m2_b = (x - mean_b).square().sum(dim=0)
        total = count + n_b
        delta = mean_b - mean
        mean = mean + delta * (n_b / total)
        m2 = m2 + m2_b + delta.square() * (count * n_b / total)
        count = total
        if log is not None and seen % 1000 == 0:
            log(f"channel stats: {seen}/{len(indices)} samples, {count} tokens")
    std = (m2 / count).sqrt().clamp_min(std_floor)
    return ChannelStats(
        mean=mean,
        std=std,
        token_count=count,
        sample_count=len(indices),
        split=split,
        cache_identity=str(dataset.identity),
        layer=layer,
        std_floor=std_floor,
        max_samples=max_samples,
    )


def load_or_compute_stats(
    path: Path,
    dataset: Any,
    *,
    split: str,
    layer: int | None,
    stats_config: dict[str, Any],
    num_workers: int,
    log: Callable[[str], None],
) -> ChannelStats:
    expected = {
        "split": split,
        "cache_identity": str(dataset.identity),
        "layer": layer,
        "std_floor": float(stats_config["std_floor"]),
        "max_samples": stats_config["max_samples"],
    }
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != STATS_FORMAT:
            raise ValueError(f"{path} is not a channel statistics file")
        stats = ChannelStats.from_dict(payload["stats"])
        stored = {key: getattr(stats, key) for key in expected}
        if stored != expected or stats.feature_dim != dataset.feature_dim:
            raise ValueError(f"existing {path} was computed for {stored}, not {expected}; use a new run dir")
        log(f"loaded channel statistics from {path}")
        return stats
    log(f"computing train-split channel statistics over {len(dataset)} samples")
    stats = compute_channel_stats(
        dataset,
        split=split,
        layer=layer,
        std_floor=expected["std_floor"],
        max_samples=expected["max_samples"],
        num_workers=num_workers,
        log=log,
    )
    atomic_torch_save({"format": STATS_FORMAT, "stats": stats.to_dict()}, path)
    log(f"wrote channel statistics ({stats.token_count} tokens, {stats.sample_count} samples) to {path}")
    return stats


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, optim: dict[str, Any]) -> torch.optim.AdamW:
    no_decay = {id(model.code_queries)}
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            no_decay.update(id(p) for p in module.parameters())
    decay, other = [], []
    for parameter in model.parameters():
        (other if id(parameter) in no_decay or parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(optim["weight_decay"])},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=float(optim["lr"]),
        betas=tuple(optim["betas"]),
        eps=float(optim["eps"]),
    )


def learning_rate(optim: dict[str, Any], step: int) -> float:
    warmup = optim["warmup_steps"]
    scale = 1.0 if warmup == 0 else min(1.0, (step + 1) / warmup)
    return float(optim["lr"]) * scale


def effective_precision(requested: str, device: torch.device) -> str:
    if requested == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def autocast_context(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b"")
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _truncate_log(path: Path, last_step: int) -> None:
    if not path.exists():
        return
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["step"] <= last_step]
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def train_stage(
    config: dict[str, Any],
    *,
    stage: str,
    cache_dir: Path | str,
    run_dir: Path | str,
    resume: bool = False,
    init_from: Path | str | None = None,
    max_steps: int | None = None,
    device: str | torch.device | None = None,
    num_workers: int | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run (or resume) one training stage; returns a JSON-serialisable summary."""

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    config = validate_config(copy.deepcopy(config))
    cache_dir, run_dir = Path(cache_dir), Path(run_dir)
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = config["data"]
    workers = data_cfg["num_workers"] if num_workers is None else num_workers
    seed = config["seed"]
    total_steps = config["stages"][stage]["max_steps"] if max_steps is None else max_steps
    _positive_int(total_steps, "max_steps", allow_zero=True)

    dataset = FeatureCacheDataset(cache_dir, data_cfg["train_split"], layer=data_cfg["layer"])
    layer = getattr(dataset, "layer", data_cfg["layer"])
    if dataset.feature_dim != config["feature_dim"]:
        raise ValueError(f"cache feature_dim {dataset.feature_dim} != config feature_dim {config['feature_dim']}")
    if len(dataset) == 0:
        raise ValueError(f"cache split {data_cfg['train_split']!r} is empty")
    identity = str(dataset.identity)
    objective = build_objective(config, stage)
    precision = effective_precision(config["precision"], device)
    model_config = {"input_dim": int(dataset.feature_dim), **config["model"]}
    stage_dir = run_dir / stage
    checkpoint_path = stage_dir / "checkpoint_latest.pt"
    model_path = stage_dir / "model.pt"
    stats_path = run_dir / "channel_stats.pt"
    log_path = stage_dir / "train_log.jsonl"

    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"{checkpoint_path} exists; pass --resume to continue it or choose a new run dir")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"--resume given but {checkpoint_path} does not exist")
    if resume and init_from is not None:
        raise ValueError("--init-from only applies to a fresh stage, not --resume")
    if stage == "fixed" and init_from is not None:
        raise ValueError("the fixed control stage trains from scratch; --init-from applies to the prefix stage")

    cache_info = {
        "identity": identity,
        "layer": layer,
        "feature_dim": int(dataset.feature_dim),
        "train_split": data_cfg["train_split"],
        "train_samples": len(dataset),
    }
    if dry_run:
        stream = BatchStream(dataset, data_cfg["token_budget"], seed=seed)
        return {
            "dry_run": True,
            "stage": stage,
            "cache_dir": str(cache_dir),
            "run_dir": str(run_dir),
            "cache": cache_info,
            "batches_per_epoch": stream.batches_per_epoch(0),
            "max_steps": total_steps,
            "objective": objective.to_dict(),
            "precision": precision,
            "device": str(device),
            "existing_checkpoint": checkpoint_path.exists(),
            "model": model_config,
        }

    stats = load_or_compute_stats(
        stats_path,
        dataset,
        split=data_cfg["train_split"],
        layer=layer,
        stats_config=config["stats"],
        num_workers=workers,
        log=log,
    )
    compatibility = {
        "stage": stage,
        "model": model_config,
        "objective": objective.to_dict(),
        "optim": config["optim"],
        "precision": precision,
        "seed": seed,
        "data": {"token_budget": data_cfg["token_budget"], "train_split": data_cfg["train_split"], "layer": layer},
        "cache_identity": identity,
        "stats_digest": stats.digest(),
    }

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(seed)
    model = build_model(model_config).to(device)
    optimizer = build_optimizer(model, config["optim"])

    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{checkpoint_path} is not an ordered-compressor checkpoint")
        if _canonical(checkpoint["compatibility"]) != _canonical(compatibility):
            stored = checkpoint["compatibility"]
            differing = sorted(k for k in set(stored) | set(compatibility)
                               if _canonical(stored.get(k)) != _canonical(compatibility.get(k)))
            raise ValueError(f"cannot resume {checkpoint_path}: incompatible {differing}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = int(checkpoint["step"])
        cursor = dict(checkpoint["cursor"])
        init_info = checkpoint["init_from"]
        if total_steps < step:
            raise ValueError(f"checkpoint is already at step {step} > requested max_steps {total_steps}")
        restore_rng(checkpoint["rng"])
        _truncate_log(log_path, step)
        log(f"resumed {stage} stage at step {step} (epoch {cursor['epoch']}, batch {cursor['batch']})")
    else:
        step, cursor, init_info = 0, {"epoch": 0, "batch": 0}, None
        source = init_from
        if stage == "prefix" and source is None and config["stages"]["prefix"]["init_from"] == "fixed":
            source = run_dir / "fixed" / "model.pt"
        if source is not None:
            bundle = load_compressor(source)
            if bundle.model.config != model_config:
                raise ValueError(f"{source} model config {bundle.model.config} != {model_config}")
            if bundle.cache_identity != identity:
                raise ValueError(f"{source} was trained on cache {bundle.cache_identity}, not {identity}")
            if bundle.stats.digest() != stats.digest():
                raise ValueError(f"{source} uses different channel statistics than {stats_path}")
            init_state = bundle.model.state_dict()
            model.load_state_dict(init_state, strict=True)
            init_info = {
                "path": str(source),
                "stage": bundle.metadata.get("stage"),
                "step": bundle.metadata.get("step"),
                "state_digest": state_digest(init_state),
            }
            log(f"initialised {stage} stage from {source} (stage {init_info['stage']}, step {init_info['step']})")
        stage_dir.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()
        torch.manual_seed(seed + 1)  # training-time RNG independent of the init path

    write_json(
        stage_dir / "config.json",
        {
            "config": config,
            "stage": stage,
            "cache_dir": str(cache_dir),
            "run_dir": str(run_dir),
            "cache": cache_info,
            "max_steps": total_steps,
            "objective": objective.to_dict(),
            "precision": precision,
            "init_from": init_info,
        },
    )

    def metadata() -> dict[str, Any]:
        return {
            "stage": stage,
            "step": step,
            "cache": cache_info,
            "objective": objective.to_dict(),
            "config": config,
            "init_from": init_info,
            "versions": {"torch": str(torch.__version__)},
        }

    def save_checkpoint() -> None:
        atomic_torch_save(
            {
                "format": CHECKPOINT_FORMAT,
                **metadata(),
                "max_steps": total_steps,
                "cursor": {"epoch": stream.epoch, "batch": stream.batch},
                "model_config": dict(model.config),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "stats": stats.to_dict(),
                "compatibility": compatibility,
                "rng": capture_rng(),
            },
            checkpoint_path,
        )

    stream = BatchStream(
        dataset,
        data_cfg["token_budget"],
        seed=seed,
        epoch=int(cursor["epoch"]),
        batch=int(cursor["batch"]),
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    log(f"{stage}: steps {step}->{total_steps}, objective weights {objective.expected_weights()}, "
        f"precision {precision}, device {device}, {stream.batches_per_epoch(stream.epoch)} batches/epoch")

    model.train()
    window: dict[str, float] = {}
    window_start = time.perf_counter()
    batches = iter(stream)
    try:
        while step < total_steps:
            batch = next(batches)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            positions = batch["positions"].to(device, non_blocking=True)
            token_types = batch["token_types"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            normalized = stats.normalize(features)
            normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
            short_budget = objective.sample_short_budget(seed, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config["optim"], step)
            with autocast_context(precision, device):
                loss, parts = objective_loss(model, objective, normalized, positions, token_types, valid, short_budget)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss {loss_value} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = config["optim"]["grad_clip"]
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip is not None else float("inf"),
                                               error_if_nonfinite=True)
            )
            optimizer.step()
            step += 1

            window["steps"] = window.get("steps", 0) + 1
            window["loss"] = window.get("loss", 0.0) + loss_value
            window["d_full"] = window.get("d_full", 0.0) + parts["d_full"].item()
            if "d_short" in parts:
                window["short_steps"] = window.get("short_steps", 0) + 1
                window["d_short"] = window.get("d_short", 0.0) + parts["d_short"].item()
                window["short_budget"] = window.get("short_budget", 0.0) + short_budget
            window["grad_norm"] = window.get("grad_norm", 0.0) + grad_norm
            window["samples"] = window.get("samples", 0) + len(batch["sample_ids"])
            window["tokens"] = window.get("tokens", 0) + int(batch["valid_mask"].sum())

            if step % config["log_every"] == 0 or step == total_steps:
                elapsed = time.perf_counter() - window_start
                n = window["steps"]
                record = {
                    "step": step,
                    "epoch": stream.epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": window["loss"] / n,
                    "d_full": window["d_full"] / n,
                    "grad_norm": window["grad_norm"] / n,
                    "samples": window["samples"],
                    "tokens": window["tokens"],
                    "elapsed_s": elapsed,
                    "tokens_per_s": window["tokens"] / max(elapsed, 1e-9),
                }
                if window.get("short_steps"):
                    record["d_short"] = window["d_short"] / window["short_steps"]
                    record["mean_short_budget"] = window["short_budget"] / window["short_steps"]
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                log(json.dumps(record))
                window, window_start = {}, time.perf_counter()
            if step % config["checkpoint_every"] == 0 or step == total_steps:
                save_checkpoint()
    finally:
        batches.close()

    if not checkpoint_path.exists():  # max_steps == 0 on a fresh stage
        save_checkpoint()
    export_compressor(model_path, model, stats, metadata())
    summary = {
        "stage": stage,
        "step": step,
        "checkpoint": str(checkpoint_path),
        "model": str(model_path),
        "stats": str(stats_path),
        "cache": cache_info,
        "objective": objective.to_dict(),
        "precision": precision,
        "init_from": init_info,
    }
    write_json(stage_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_cache_dir(config: dict[str, Any], cache_dir: str | None, cache_root: str | None,
                      cache_identity: str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    if cache_root is None:
        raise SystemExit("pass --cache-dir or --cache-root")
    if not config["data"]["model"]:
        raise SystemExit("config data.model is required with --cache-root")
    model_dir = Path(cache_root) / config["data"]["model"]
    identity = cache_identity or config["data"]["identity"]
    if identity:
        return model_dir / identity
    candidates = sorted(p for p in model_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(candidates) != 1:
        raise SystemExit(
            f"{model_dir} holds {len(candidates)} caches {[p.name for p in candidates]}; pass --cache-identity"
        )
    return candidates[0]


def resolve_run_dir(config: dict[str, Any], run_dir: str | None, run_root: str | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if run_root is None:
        raise SystemExit("pass --run-dir or --run-root")
    return Path(run_root) / config["name"]


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", help="exact cache directory (<cache_root>/<model>/<identity>)")
    parser.add_argument("--cache-root", help="cache root; resolved as <root>/<data.model>/<identity>")
    parser.add_argument("--cache-identity", help="cache identity under --cache-root (default: config or the only one)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON config, e.g. configs/compression/dinov3.json")
    parser.add_argument("--stage", required=True, choices=STAGES)
    add_path_arguments(parser)
    parser.add_argument("--run-dir", help="exact run directory shared by both stages")
    parser.add_argument("--run-root", help="run root; run dir is <root>/<config name>")
    parser.add_argument("--resume", action="store_true", help="continue RUN/<stage>/checkpoint_latest.pt")
    parser.add_argument("--init-from", help="prefix stage initialisation (default RUN/fixed/model.pt)")
    parser.add_argument("--max-steps", type=int, help="override stages.<stage>.max_steps")
    parser.add_argument("--device", help="torch device (default cuda if available)")
    parser.add_argument("--num-workers", type=int, help="override data.num_workers")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help="override a config value, e.g. --set model.width=64 (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="resolve config/cache/batches and exit")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.set)
    cache_dir = resolve_cache_dir(config, args.cache_dir, args.cache_root, args.cache_identity)
    run_dir = resolve_run_dir(config, args.run_dir, args.run_root)
    summary = train_stage(
        config,
        stage=args.stage,
        cache_dir=cache_dir,
        run_dir=run_dir,
        resume=args.resume,
        init_from=args.init_from,
        max_steps=args.max_steps,
        device=args.device,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
