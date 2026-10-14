"""Evaluate a trained ordered compressor on a cached split at several code budgets.

Each sample is encoded once at the largest requested budget; every budget is then
decoded from the exact code prefix.  Metrics are sample-equal means of
normalized-space MSE (train-split channel statistics), native-space cosine and
native-space MSE, plus per-token-type MSE and actual code sizes.  Quality is not
assumed monotone in the budget; violations are reported.

    python -m compression.evaluate --checkpoint RUN/prefix/model.pt --cache-dir CACHE \
        --split validation --output RUN/prefix/eval_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from compression.cache import FeatureCacheDataset, collate_features
from compression.model import CompressorBundle, load_compressor
from compression.train import (
    add_path_arguments,
    autocast_context,
    effective_precision,
    plan_token_batches,
    resolve_cache_dir,
    resolve_eval_budgets,
    select_stat_indices,
    write_json,
)

EVAL_FORMAT = "ordered-compressor-eval/v1"
TOKEN_TYPE_NAMES = {0: "patch", 1: "cls", 2: "register", 3: "camera"}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample mean over ``mask``; NaN for samples with no selected token."""

    total = torch.where(mask, values, torch.zeros_like(values)).sum(dim=1)
    count = mask.sum(dim=1)
    return torch.where(count > 0, total / count.clamp_min(1), torch.full_like(total, float("nan")))


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "std": None, "count": 0}
    return {"mean": float(finite.mean()), "std": float(finite.std()), "count": int(finite.size)}


def check_cache_compatibility(bundle: CompressorBundle, dataset: Any) -> None:
    cache = bundle.metadata["cache"]
    if str(dataset.identity) != cache["identity"]:
        raise ValueError(f"cache identity {dataset.identity} does not match the compressor's {cache['identity']}")
    if int(dataset.feature_dim) != bundle.model.input_dim:
        raise ValueError(f"cache feature_dim {dataset.feature_dim} != compressor input_dim {bundle.model.input_dim}")
    layer = getattr(dataset, "layer", cache["layer"])
    if layer != cache["layer"]:
        raise ValueError(f"cache layer {layer} does not match the compressor's layer {cache['layer']}")


@torch.no_grad()
def evaluate_bundle(
    bundle: CompressorBundle,
    dataset: Any,
    budgets: list[int],
    *,
    token_budget: int,
    device: torch.device,
    precision: str = "fp32",
    max_samples: int | None = None,
    num_workers: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return per-budget metrics plus per-sample arrays for ``dataset``."""

    check_cache_compatibility(bundle, dataset)
    model = bundle.model.to(device).eval()
    budgets = resolve_eval_budgets(budgets, model.max_codes)
    indices = select_stat_indices(len(dataset), max_samples)
    counts = list(dataset.token_counts)
    plan = plan_token_batches([counts[i] for i in indices], token_budget, shuffle=False)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_sampler=plan,
        collate_fn=collate_features,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    stats = bundle.stats
    sample_ids: list[str] = []
    token_counts: list[int] = []
    baseline: list[torch.Tensor] = []
    nmse: list[torch.Tensor] = []
    cosine: list[torch.Tensor] = []
    native_mse: list[torch.Tensor] = []
    by_type: dict[int, list[torch.Tensor]] = {t: [] for t in TOKEN_TYPE_NAMES}
    present_types: set[int] = set()

    for batch_index, batch in enumerate(loader):
        valid = batch["valid_mask"].to(device)
        positions = batch["positions"].to(device)
        token_types = batch["token_types"].to(device)
        native = batch["features"].to(device).float()
        native = torch.where(valid[..., None], native, torch.zeros_like(native))
        normalized = stats.normalize(native)
        normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
        with autocast_context(precision, device):
            codes = model.encode(normalized, positions, token_types, valid, num_codes=budgets[-1])
        batch_nmse, batch_cos, batch_native, batch_types = [], [], [], {t: [] for t in TOKEN_TYPE_NAMES}
        for budget in budgets:
            with autocast_context(precision, device):
                prediction = model.decode(codes[:, :budget], positions, token_types, valid)
            prediction = prediction.float()
            token_nmse = (prediction - normalized).square().mean(dim=-1)
            prediction_native = stats.denormalize(prediction)
            token_cos = F.cosine_similarity(prediction_native, native, dim=-1, eps=1e-8)
            token_native = (prediction_native - native).square().mean(dim=-1)
            batch_nmse.append(_masked_mean(token_nmse, valid))
            batch_cos.append(_masked_mean(token_cos, valid))
            batch_native.append(_masked_mean(token_native, valid))
            for token_type in TOKEN_TYPE_NAMES:
                batch_types[token_type].append(_masked_mean(token_nmse, valid & (token_types == token_type)))
        nmse.append(torch.stack(batch_nmse, dim=1).cpu())
        cosine.append(torch.stack(batch_cos, dim=1).cpu())
        native_mse.append(torch.stack(batch_native, dim=1).cpu())
        for token_type in TOKEN_TYPE_NAMES:
            by_type[token_type].append(torch.stack(batch_types[token_type], dim=1).cpu())
        present_types.update(int(t) for t in torch.unique(token_types[valid]).tolist())
        baseline.append(_masked_mean(normalized.square().mean(dim=-1), valid).cpu())
        sample_ids.extend(batch["sample_ids"])
        token_counts.extend(batch["valid_mask"].sum(dim=1).tolist())
        if log is not None and (batch_index + 1) % 50 == 0:
            log(f"evaluated {len(sample_ids)}/{len(indices)} samples")

    return {
        "budgets": budgets,
        "sample_ids": sample_ids,
        "token_counts": np.asarray(token_counts, dtype=np.int64),
        "normalized_mse": torch.cat(nmse).double().numpy(),
        "native_cosine": torch.cat(cosine).double().numpy(),
        "native_mse": torch.cat(native_mse).double().numpy(),
        "normalized_mse_by_type": {
            t: torch.cat(by_type[t]).double().numpy() for t in sorted(present_types)
        },
        "train_mean_baseline_nmse": torch.cat(baseline).double().numpy(),
    }


def build_report(raw: dict[str, Any], bundle: CompressorBundle, *, per_sample: bool) -> dict[str, Any]:
    width = bundle.model.width
    feature_dim = bundle.model.input_dim
    budgets = raw["budgets"]
    mean_tokens = float(raw["token_counts"].mean())
    metrics = []
    for column, budget in enumerate(budgets):
        scalars = budget * width
        metrics.append(
            {
                "budget": budget,
                "code_scalars": scalars,
                "code_bytes_float32": 4 * scalars,
                "code_bytes_bfloat16": 2 * scalars,
                "scalar_compression_ratio": mean_tokens * feature_dim / scalars,
                "normalized_mse": _summary(raw["normalized_mse"][:, column]),
                "native_cosine": _summary(raw["native_cosine"][:, column]),
                "native_mse": _summary(raw["native_mse"][:, column]),
                "normalized_mse_by_token_type": {
                    TOKEN_TYPE_NAMES[t]: _summary(values[:, column])
                    for t, values in raw["normalized_mse_by_type"].items()
                },
            }
        )
    means = [m["normalized_mse"]["mean"] for m in metrics]
    increases = [
        {"from_budget": budgets[i], "to_budget": budgets[i + 1], "from": means[i], "to": means[i + 1]}
        for i in range(len(budgets) - 1)
        if means[i + 1] > means[i]
    ]
    per_sample_increase = (np.diff(raw["normalized_mse"], axis=1) > 0).any(axis=1) if len(budgets) > 1 else None
    report = {
        "num_samples": len(raw["sample_ids"]),
        "num_tokens": int(raw["token_counts"].sum()),
        "mean_tokens_per_sample": mean_tokens,
        "source_scalars_per_sample": mean_tokens * feature_dim,
        "source_bytes_float16_per_sample": 2 * mean_tokens * feature_dim,
        "budgets": budgets,
        "train_mean_baseline": {"normalized_mse": _summary(raw["train_mean_baseline_nmse"])},
        "metrics": metrics,
        "monotonicity": {
            "mean_normalized_mse_increases": increases,
            "fraction_samples_with_any_increase": None
            if per_sample_increase is None
            else float(per_sample_increase.mean()),
        },
    }
    if per_sample:
        report["per_sample"] = {
            "sample_ids": raw["sample_ids"],
            "token_counts": raw["token_counts"].tolist(),
            "normalized_mse": raw["normalized_mse"].tolist(),
            "native_cosine": raw["native_cosine"].tolist(),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="RUN/<stage>/model.pt or checkpoint_latest.pt")
    add_path_arguments(parser)
    parser.add_argument("--split", help="cache split to evaluate (default: config data.eval_split)")
    parser.add_argument("--budgets", help="comma-separated code budgets (default: config eval.budgets or powers of two)")
    parser.add_argument("--max-samples", type=int, help="deterministic evenly spaced subset of the split")
    parser.add_argument("--token-budget", type=int, help="padded tokens per batch (default: config data.token_budget)")
    parser.add_argument("--device", help="torch device (default cuda if available)")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True, help="metrics JSON path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--per-sample", action="store_true", help="include per-sample metric arrays")
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_compressor(args.checkpoint)
    config = bundle.metadata["config"]
    cache_info = bundle.metadata["cache"]
    cache_dir = resolve_cache_dir(config, args.cache_dir, args.cache_root, args.cache_identity or cache_info["identity"])
    split = args.split or config["data"]["eval_split"]
    dataset = FeatureCacheDataset(cache_dir, split, layer=cache_info["layer"])
    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    else:
        budgets = config["eval"]["budgets"]
    budgets = resolve_eval_budgets(budgets, bundle.model.max_codes)
    precision = effective_precision(args.precision, device)
    raw = evaluate_bundle(
        bundle,
        dataset,
        budgets,
        token_budget=args.token_budget or config["data"]["token_budget"],
        device=device,
        precision=precision,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        log=lambda message: print(message, flush=True),
    )
    report = {
        "format": EVAL_FORMAT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": bundle.metadata.get("stage"),
        "checkpoint_step": bundle.metadata.get("step"),
        "objective": bundle.metadata.get("objective"),
        "model_config": bundle.model.config,
        "cache_dir": str(cache_dir),
        "cache_identity": str(dataset.identity),
        "layer": cache_info["layer"],
        "split": getattr(dataset, "split", split),
        "heldout": getattr(dataset, "split", split) != cache_info["train_split"],
        "stats": {"split": bundle.stats.split, "token_count": bundle.stats.token_count,
                  "sample_count": bundle.stats.sample_count},
        "precision": precision,
        "device": str(device),
        "max_samples": args.max_samples,
        **build_report(raw, bundle, per_sample=args.per_sample),
    }
    write_json(output, report)
    summary = [
        (m["budget"], m["normalized_mse"]["mean"], m["native_cosine"]["mean"]) for m in report["metrics"]
    ]
    print(json.dumps({"output": str(output), "budget_nmse_cosine": summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
