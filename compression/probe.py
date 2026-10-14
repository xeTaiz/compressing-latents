"""Frozen native/reconstruction/direct-code linear probes, without backbone imports.

Vectors are float32 disk mmaps. Ordered codes are encoded once at the largest
requested budget, then physically sliced; they are never pooled or decoded for
the compressed classifier. All affine preprocessing is fitted on train only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cache import FeatureCacheDataset, collate_features
from .model import atomic_torch_save, load_compressor

FORMAT = "compression-linear-probe/v1"
DEFAULT_BUDGETS = (4, 32, 64, 128, 256)

# Bound padded native activations as well as sample count, without changing the
# selected row order or the persisted feature identity.
_FEATURE_BATCH_SAMPLES = 64
_FEATURE_TOKEN_BUDGET = 32768
_FEATURE_CHECKPOINT_ROWS = 256
_FEATURE_CHECKPOINT_SECONDS = 60.0


class ProbeDeadline(Exception):
    """A resumable, non-failure stop at the explicit wall-clock boundary."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_deadline(value: str | None) -> float | None:
    if value is None:
        return None
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("deadline must include an explicit timezone")
    return date.timestamp()


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.time() >= deadline:
        raise ProbeDeadline()


def pool_native(features: torch.Tensor, types: torch.Tensor,
                valid: torch.Tensor, pooling: str) -> torch.Tensor:
    mask = valid & (types == (1 if pooling == "cls" else 0))
    counts = mask.sum(dim=1)
    if pooling == "cls" and not torch.all(counts == 1):
        raise ValueError("DINO CLS pooling requires exactly one type-1 token per sample")
    if pooling not in ("cls", "patch_mean"):
        raise ValueError(f"unsupported native pooling {pooling!r}")
    if torch.any(counts == 0):
        raise ValueError("patch-mean pooling requires type-0 patch tokens")
    return torch.where(mask[..., None], features, 0.0).sum(dim=1) / counts[:, None]


def selected_indices(dataset: FeatureCacheDataset, limit: int | None, seed: int) -> list[int]:
    if limit is not None and limit <= 0:
        raise ValueError("sample limits must be positive")
    if limit is None or limit >= len(dataset):
        return list(range(len(dataset)))
    # Hash ordering is independent of filesystem/shard ordering. Return cache order.
    ordered = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(
        f"{seed}:{dataset.split}:{dataset.sample_ids[i]}".encode()).digest())
    return sorted(ordered[:limit])


def read_labels(dataset: FeatureCacheDataset, indices: list[int]) -> list[int]:
    labels = []
    for index in indices:
        try:
            label = dataset.metadata[index]["manifest_record"]["label"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"missing manifest label for {dataset.sample_ids[index]}") from error
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"label must be an integer for {dataset.sample_ids[index]}")
        labels.append(label)
    return labels


def class_coverage(train: list[int], validation: list[int]) -> dict[str, Any]:
    tr, va = Counter(train), Counter(validation)
    return {"train_labels": sorted(tr), "validation_labels": sorted(va),
            "unseen_validation_labels": sorted(va.keys() - tr.keys()),
            "train_classes": len(tr), "validation_classes": len(va),
            "per_class": [{"label": label, "train_samples": tr[label],
                           "validation_samples": va[label]} for label in sorted(tr.keys() | va.keys())],
            "known_label_samples": sum(count for label, count in va.items() if label in tr),
            "unseen_label_samples": sum(count for label, count in va.items() if label not in tr)}


def conditions(representations: list[str], budgets: list[int]) -> list[tuple[str, int | None]]:
    return [(representation, None if representation == "original" else budget)
            for representation in representations
            for budget in ([None] if representation == "original" else budgets)]


def condition_name(condition: str, budget: int | None) -> str:
    return condition if budget is None else f"{condition}-{budget}"


def prepare_features(root: Path, datasets: dict, indices: dict, representations: list[str],
                     budgets: list[int], pooling: str, bundle: Any, device: str,
                     feature_identity: dict, deadline: float | None) -> dict:
    """Batch native extraction; publish only flushed, all-condition row prefixes."""
    manifest_path = root / "features_manifest.json"
    specs = {}
    max_k = max(budgets) if bundle is not None else None
    for split, dataset in datasets.items():
        count = len(indices[split])
        if "original" in representations:
            specs[f"{split}-original"] = [count, dataset.feature_dim]
        if "reconstructed" in representations:
            for k in budgets:
                specs[f"{split}-reconstructed-{k}"] = [count, dataset.feature_dim]
        if "compressed" in representations:
            specs[f"{split}-codes"] = [count, max_k, bundle.model.width]
    manifest = {"format": FORMAT, "identity": feature_identity, "dtype": "float32",
                "arrays": specs, "rows_completed": {split: 0 for split in datasets}, "completed": False}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != feature_identity or previous["arrays"] != specs:
            raise ValueError("feature preparation identity mismatch; refusing foreign artifacts")
        manifest = previous
        if set(manifest["rows_completed"]) != set(datasets):
            raise ValueError("feature preparation split mismatch")
        for split in datasets:
            rows = manifest["rows_completed"][split]
            if not isinstance(rows, int) or not 0 <= rows <= len(indices[split]):
                raise ValueError("invalid committed feature row count")
            if manifest["completed"] and rows != len(indices[split]):
                raise ValueError("completed feature manifest has incomplete rows")
    else:
        if (root / "features").exists() and any((root / "features").iterdir()):
            raise ValueError("uncommitted foreign feature files without a manifest")
        atomic_json(manifest_path, manifest)
    feature_dir = root / "features"
    feature_dir.mkdir(exist_ok=True)
    arrays = {}
    try:
        for name, shape in specs.items():
            path = feature_dir / f"{name}.npy"
            if path.exists():
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
                if list(array.shape) != shape or array.dtype != np.float32:
                    raise ValueError(f"feature shape/precision mismatch: {path}")
            else:
                split = name.split("-", 1)[0]
                if manifest["rows_completed"][split] or manifest["completed"]:
                    raise ValueError(f"missing committed feature array {path}")
                # Never expose a partially written .npy header as a reusable array.
                tmp = path.with_name(f".{path.name}.tmp")
                pending = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
                pending.flush()
                pending._mmap.close()
                os.replace(tmp, path)
                array = np.load(path, mmap_mode="r+", allow_pickle=False)
            arrays[name] = array
        def commit(split: str, stop: int) -> None:
            if stop == manifest["rows_completed"][split]:
                return
            for name, array in arrays.items():
                if name.startswith(split + "-"):
                    array.flush()
                    # msync the mapping before fsyncing its file, including a
                    # newly created header, before publishing the durable cursor.
                    with (feature_dir / f"{name}.npy").open("rb") as handle:
                        os.fsync(handle.fileno())
            manifest["rows_completed"][split] = stop
            atomic_json(manifest_path, manifest)

        with torch.inference_mode():
            for split, dataset in datasets.items():
                row = manifest["rows_completed"][split]
                last_commit = time.monotonic()
                try:
                    while row < len(indices[split]):
                        check_deadline(deadline)
                        stop, longest = row, 0
                        while stop < min(row + _FEATURE_BATCH_SAMPLES, len(indices[split])):
                            tokens = dataset.token_counts[indices[split][stop]]
                            if stop > row and max(longest, tokens) * (stop - row + 1) > _FEATURE_TOKEN_BUDGET:
                                break
                            longest = max(longest, tokens)
                            stop += 1
                        samples = []
                        for selected in indices[split][row:stop]:
                            check_deadline(deadline)
                            samples.append(dataset[selected])
                        batch = collate_features(samples)
                        del samples
                        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                                 for key, value in batch.items()}
                        native, positions, types, valid = (batch[key] for key in
                                                          ("features", "positions", "token_types", "valid_mask"))
                        values = {}
                        check_deadline(deadline)
                        if "original" in representations:
                            values[f"{split}-original"] = pool_native(native, types, valid, pooling).cpu().numpy()
                        if bundle is not None:
                            check_deadline(deadline)
                            codes = bundle.encode_native(native, positions, types, valid, num_codes=max_k)
                            check_deadline(deadline)
                            if "compressed" in representations:
                                values[f"{split}-codes"] = codes.cpu().numpy()
                            if "reconstructed" in representations:
                                for k in budgets:
                                    check_deadline(deadline)
                                    reconstructed = bundle.decode_native(codes[:, :k], positions, types, valid)
                                    values[f"{split}-reconstructed-{k}"] = pool_native(
                                        reconstructed, types, valid, pooling).cpu().numpy()
                                    del reconstructed
                                    check_deadline(deadline)
                            del codes
                        # A deadline or failure above leaves this entire batch
                        # uncommitted. Resume always overwrites the durable tail.
                        check_deadline(deadline)
                        for name in values:
                            arrays[name][row:stop] = values[name]
                        del values, batch, native, positions, types, valid
                        row = stop
                        check_deadline(deadline)
                        if (row - manifest["rows_completed"][split] >= _FEATURE_CHECKPOINT_ROWS
                                or time.monotonic() - last_commit >= _FEATURE_CHECKPOINT_SECONDS):
                            commit(split, row)
                            last_commit = time.monotonic()
                    commit(split, row)
                except ProbeDeadline:
                    commit(split, row)
                    raise
        manifest["completed"] = True
        atomic_json(manifest_path, manifest)
        return manifest
    finally:
        for array in arrays.values():
            array._mmap.close()


def feature_batch(array: np.ndarray, rows: np.ndarray, budget: int | None) -> np.ndarray:
    # Slice the code axis BEFORE gathering rows; never duplicate max-K vectors.
    source = array[:, :budget, :] if array.ndim == 3 else array
    return np.asarray(source[rows], dtype=np.float32).reshape(len(rows), -1)


def fit_normalization(array: np.ndarray, budget: int | None, batch_size: int,
                      deadline: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable bounded-memory parallel variance reduction using train vectors only."""
    count, mean, m2 = 0, None, None
    for start in range(0, len(array), batch_size):
        check_deadline(deadline)
        values = feature_batch(array, np.arange(start, min(start + batch_size, len(array))), budget).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("non-finite training probe features")
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        n = len(values)
        if mean is None:
            mean, m2, count = batch_mean, batch_m2, n
        else:
            delta = batch_mean - mean
            total = count + n
            m2 += batch_m2 + delta * delta * (count * n / total)
            mean += delta * (n / total)
            count = total
    if count == 0:
        raise ValueError("empty training split")
    std = np.maximum(np.sqrt(m2 / count), 1e-6)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))


def validation_metrics(head: nn.Linear, array: np.ndarray, labels: torch.Tensor,
                       mean: torch.Tensor, std: torch.Tensor, budget: int | None,
                       batch_size: int, device: str, deadline: float | None = None) -> dict:
    top1 = top5 = known = 0
    ce = 0.0
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(array), batch_size):
            check_deadline(deadline)
            stop = min(start + batch_size, len(array))
            x = torch.from_numpy(feature_batch(array, np.arange(start, stop), budget)).to(device)
            logits = head((x - mean) / std)
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite validation logits")
            y = labels[start:stop].to(device)
            seen = y >= 0
            predictions = logits.topk(min(5, logits.shape[1]), dim=1).indices
            top1 += int(((predictions[:, 0] == y) & seen).sum())
            top5 += int(((predictions == y[:, None]).any(dim=1) & seen).sum())
            known += int(seen.sum())
            if seen.any():
                ce += float(F.cross_entropy(logits[seen], y[seen], reduction="sum"))
    total = len(array)
    if not total:
        raise ValueError("empty validation split")
    return {"top1": top1 / total, "top5": top5 / total,
            "cross_entropy": ce / known if known else None,
            "cross_entropy_samples": known, "validation_samples": total,
            "known_label_samples": known, "unseen_label_samples": total - known}


def read_head(path: Path, run_identity: str) -> dict | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["identity"] != run_identity or state["format"] != FORMAT
            or path.stem != condition_name(state["condition"], state["budget"])):
        raise ValueError(f"classifier identity mismatch: {path}")
    return state


def train_head(root: Path, condition: str, budget: int | None, run_identity: str,
               train_labels: torch.Tensor, validation_labels: torch.Tensor, classes: int,
               args: argparse.Namespace, deadline: float | None, publish) -> None:
    name = condition_name(condition, budget)
    suffix = "codes" if condition == "compressed" else name
    train_array = np.load(root / "features" / f"train-{suffix}.npy", mmap_mode="r", allow_pickle=False)
    validation_array = np.load(root / "features" / f"validation-{suffix}.npy", mmap_mode="r", allow_pickle=False)
    head_path = root / "heads" / f"{name}.pt"
    norm_path = root / "heads" / f"{name}-normalization.pt"
    try:
        dim = int(np.prod(train_array.shape[1:])) if condition != "compressed" else budget * train_array.shape[2]
        state = read_head(head_path, run_identity)
        if state is not None and state["epoch"] >= args.epochs:
            return
        check_deadline(deadline)
        if norm_path.exists():
            norm = torch.load(norm_path, map_location="cpu", weights_only=True)
            if (norm["identity"] != run_identity or norm["condition"] != condition
                    or norm["budget"] != budget):
                raise ValueError("normalization identity mismatch")
            mean, std = norm["mean"], norm["std"]
        else:
            mean, std = fit_normalization(train_array, budget, args.batch_size, deadline)
            atomic_torch_save({"identity": run_identity, "condition": condition, "budget": budget,
                               "mean": mean, "std": std}, norm_path)
        torch.manual_seed(args.seed)
        head = nn.Linear(dim, classes).to(args.device)
        optimizer = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        epoch, cursor, loss_sum, history, last_metric = 0, 0, 0.0, [], None
        if state is not None:
            head.load_state_dict(state["head"])
            optimizer.load_state_dict(state["optimizer"])
            epoch, cursor, loss_sum = state["epoch"], state["cursor"], state["loss_sum"]
            history, last_metric = state["history"], state["metric"]
        mean, std = mean.to(args.device), std.to(args.device)
        def save():
            atomic_torch_save({"format": FORMAT, "identity": run_identity, "condition": condition,
                               "budget": budget, "feature_dim": dim, "class_labels": publish.class_labels,
                               "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                               "epoch": epoch, "cursor": cursor, "loss_sum": loss_sum,
                               "history": history, "metric": last_metric,
                               "normalization": str(norm_path.relative_to(root))}, head_path)
        try:
            while epoch < args.epochs:
                check_deadline(deadline)
                lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(len(train_array))
                head.train()
                while cursor < len(order):
                    check_deadline(deadline)
                    rows = order[cursor:cursor + args.batch_size]
                    x = torch.from_numpy(feature_batch(train_array, rows, budget)).to(args.device)
                    y = train_labels[torch.from_numpy(rows)].to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head((x - mean) / std), y)
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite linear probe training loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(rows)
                    cursor += len(rows)
                metrics = validation_metrics(head, validation_array, validation_labels, mean, std,
                                             budget, args.batch_size, args.device, deadline)
                epoch += 1
                last_metric = {"condition": condition, "budget": budget, "feature_dim": dim,
                               "classifier_parameters": (dim + 1) * classes,
                               "epochs_completed": epoch, "classifier_checkpoint": str(head_path.relative_to(root)),
                               **metrics}
                record = {"condition": condition, "budget": budget, "epoch": epoch,
                          "train_loss": loss_sum / len(train_array), "val_top1": metrics["top1"],
                          "val_top5": metrics["top5"], "cross_entropy": metrics["cross_entropy"],
                          "lr": lr, "elapsed_s": publish.elapsed()}
                history.append(record)
                cursor, loss_sum = 0, 0.0
                save()  # Checkpoint commits before readers receive this epoch.
                publish(last_metric, record)
        except ProbeDeadline:
            save()  # Preserve partial-epoch updates and exact deterministic cursor.
            raise
    finally:
        train_array._mmap.close()
        validation_array._mmap.close()


class Publisher:
    def __init__(self, root: Path, result: dict, run_identity: str, pairs: list):
        self.root, self.result, self.started = root, result, time.monotonic()
        self.class_labels = result["class_coverage"]["train_labels"]
        self.offset = 0.0
        records = []
        for condition, budget in pairs:
            state = read_head(root / "heads" / f"{condition_name(condition, budget)}.pt", run_identity)
            if state:
                records.extend(state["history"])
                if state["metric"]:
                    result["metrics"].append(state["metric"])
        records.sort(key=lambda row: row["elapsed_s"])
        if records:
            self.offset = max(row["elapsed_s"] for row in records)
        # Reconcile a crash between checkpoint commit and JSONL/snapshot publication.
        path = root / "probe_log.jsonl"
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with tmp.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self.snapshot("running")

    def elapsed(self):
        return self.offset + time.monotonic() - self.started

    def snapshot(self, status: str, error: str | None = None):
        self.result.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), elapsed_s=self.elapsed())
        if error is not None:
            self.result["error"] = error
        atomic_json(self.root / "probe_results.json", self.result)

    def __call__(self, metric: dict, record: dict):
        self.result["metrics"] = [row for row in self.result["metrics"]
                                  if (row["condition"], row["budget"]) != (metric["condition"], metric["budget"])]
        self.result["metrics"].append(metric)
        with (self.root / "probe_log.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.snapshot("running")


def run_probe(args: argparse.Namespace) -> dict:
    representations = args.representations.split(",")
    if len(set(representations)) != len(representations) or not set(representations) <= {"original", "reconstructed", "compressed"}:
        raise ValueError("representations must be unique original,reconstructed,compressed names")
    budgets = sorted(set(int(k) for k in args.budgets.split(",")))
    if not budgets or min(budgets) <= 0:
        raise ValueError("budgets must be positive integers")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.weight_decay < 0 or args.seed < 0:
        raise ValueError("epochs/batch-size/lr must be positive; weight-decay/seed nonnegative")
    deadline = parse_deadline(args.deadline)
    root, cache = Path(args.run_dir).resolve(), Path(args.cache_dir).resolve()
    needs_checkpoint = any(rep != "original" for rep in representations)
    if needs_checkpoint and not args.checkpoint:
        raise ValueError("reconstructed/compressed probes require --checkpoint")
    checkpoint = Path(args.checkpoint).resolve() if needs_checkpoint else None
    digest = sha256_file(checkpoint) if checkpoint else None
    bundle = load_compressor(checkpoint, map_location="cpu") if checkpoint else None
    if checkpoint and sha256_file(checkpoint) != digest:
        raise ValueError("checkpoint changed while loading; use an immutable model export")
    # Set before moving the compressor to CUDA or doing any matrix operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    datasets = {}
    try:
        for split in ("train", "validation"):
            datasets[split] = FeatureCacheDataset(cache, split, layer=args.layer)
        train, validation = datasets["train"], datasets["validation"]
        if train.identity != validation.identity or train.layer != validation.layer or train.feature_dim != validation.feature_dim:
            raise ValueError("train/validation cache mismatch")
        model = str(train.provenance.get("model", train.provenance.get("model_id", "unknown")))
        pooling = args.pooling
        if pooling is None:
            if "dino" in model.lower():
                pooling = "cls"
            elif "qwen" in model.lower():
                pooling = "patch_mean"
            else:
                raise ValueError(f"unsupported model {model!r}; choose --pooling explicitly")
        if bundle:
            if bundle.cache_identity != train.identity or bundle.stats.cache_identity != train.identity:
                raise ValueError("checkpoint/cache identity mismatch")
            if bundle.stats.split != "train" or bundle.stats.layer != train.layer or bundle.model.input_dim != train.feature_dim:
                raise ValueError("checkpoint training statistics/layer/dimension mismatch")
            if max(budgets) > bundle.model.max_codes:
                raise ValueError(f"requested budget {max(budgets)} exceeds checkpoint max_codes={bundle.model.max_codes}")
            bundle.to(args.device)
            bundle.model.requires_grad_(False)
        indices = {"train": selected_indices(train, args.max_train_samples, args.seed),
                   "validation": selected_indices(validation, args.max_validation_samples, args.seed)}
        labels = {split: read_labels(dataset, indices[split]) for split, dataset in datasets.items()}
        if not labels["train"] or not labels["validation"]:
            raise ValueError("both official splits must contain selected samples")
        selection = {split: {"sample_ids": [dataset.sample_ids[i] for i in indices[split]],
                             "labels": labels[split], "source_samples": len(dataset),
                             "selected_samples": len(indices[split])} for split, dataset in datasets.items()}
        for value in selection.values():
            value["sample_ids_sha256"] = identity(value["sample_ids"])
        protocol = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "weight_decay": args.weight_decay, "seed": args.seed, "momentum": 0.9,
                    "schedule": "cosine", "optimizer": "SGD", "normalization": "train_channel_standardization",
                    "std_floor": 1e-6, "precision": "float32", "class_mapping": "sorted_train_labels",
                    "validation_accuracy_denominator": "all_validation", "cross_entropy_denominator": "known_validation_labels",
                    "bounded_sample_smoke": args.max_train_samples is not None or args.max_validation_samples is not None,
                    "selection": "seeded_sha256_subset_in_cache_order", "deterministic_algorithms": True}
        feature_identity = {"cache_identity": train.identity, "layer": train.layer,
                            "checkpoint_sha256": digest, "pooling": pooling, "compressed_pooling": "ordered_prefix_flatten",
                            "representations": representations, "budgets": budgets if bundle else [],
                            "sample_selection": selection, "dtype": "float32", "format": FORMAT}
        config = {"format": FORMAT, "feature_identity": feature_identity, "protocol": protocol,
                  "device": args.device, "torch_version": str(torch.__version__)}
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "probe_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("probe configuration identity mismatch; refusing to overwrite existing work")
        else:
            if any(root.iterdir()):
                raise ValueError("nonempty probe directory has no config; refusing foreign artifacts")
            atomic_json(config_path, config)
        result = {"format": FORMAT, "status": "running", "model": model, "cache_dir": str(cache),
                  "cache_identity": train.identity, "layer": train.layer, "checkpoint": str(checkpoint) if checkpoint else None,
                  "checkpoint_sha256": digest, "checkpoint_stage": bundle.metadata.get("stage") if bundle else None,
                  "checkpoint_step": bundle.metadata.get("step") if bundle else None,
                  "pooling": pooling, "compressed_pooling": "ordered_prefix_flatten", "seed": args.seed,
                  "train_samples": len(labels["train"]), "validation_samples": len(labels["validation"]),
                  "class_coverage": class_coverage(labels["train"], labels["validation"]), "protocol": protocol,
                  "sample_selection": selection, "deadline": args.deadline, "metrics": [],
                  "warnings": ["Direct-code feature_dim=budget*width; classifier head capacity varies with budget and differs from native features.",
                               "Validation cross entropy uses known-label subset; accuracy includes every validation sample."]}
        run_identity = identity(config)
        publisher = Publisher(root, result, run_identity, conditions(representations, budgets))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            manifest = prepare_features(root, datasets, indices, representations, budgets, pooling,
                                        bundle, args.device, feature_identity, deadline)
            publisher.result["feature_artifacts"] = {"manifest": "features_manifest.json", "precision": "float32",
                                                      "completed": manifest["completed"]}
            mapping = {label: i for i, label in enumerate(result["class_coverage"]["train_labels"])}
            train_labels = torch.tensor([mapping[label] for label in labels["train"]], dtype=torch.long)
            validation_labels = torch.tensor([mapping.get(label, -1) for label in labels["validation"]], dtype=torch.long)
            for condition, budget in conditions(representations, budgets):
                train_head(root, condition, budget, run_identity, train_labels, validation_labels,
                           len(mapping), args, deadline, publisher)
            publisher.snapshot("completed")
        except ProbeDeadline:
            publisher.snapshot("deadline")
        except Exception as error:
            publisher.snapshot("failed", f"{type(error).__name__}: {error}")
            raise
        return publisher.result
    finally:
        for dataset in datasets.values():
            dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--representations", default="original,reconstructed,compressed")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline")
    parser.add_argument("--pooling", choices=("cls", "patch_mean"))
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run_probe(build_parser().parse_args(argv))
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
