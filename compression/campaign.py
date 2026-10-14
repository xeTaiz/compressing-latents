"""Run explicitly assigned cached-feature experiments, serially, before a hard deadline.

All manifest paths resolve relative to the manifest's parent. The runner never
edits campaign.json, launches fleet jobs, imports backbones, or discovers runs.
Concurrent disjoint assignments merge state under a short filesystem lock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

MANIFEST_FORMAT = "compression-campaign/v1"
STATE_FORMAT = "compression-campaign-state/v1"
PROBE_DEFAULTS = {"epochs": 50, "batch_size": 256, "lr": 0.1, "weight_decay": 0.0, "seed": 0}
PATH_FIELDS = ("cache_dir", "config_path", "run_dir", "probe_dir", "original_probe_dir", "eval_path")


class DeadlineExceeded(RuntimeError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("deadline must contain an explicit timezone")
    return parsed.astimezone(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"manifest format must be {MANIFEST_FORMAT}")
    for key in ("campaign_id", "created_at", "deadline", "scope", "experiments"):
        if key not in manifest:
            raise ValueError(f"manifest is missing {key}")
    if not isinstance(manifest["campaign_id"], str) or not manifest["campaign_id"]:
        raise ValueError("campaign_id must be nonempty")
    aware_datetime(manifest["created_at"])
    aware_datetime(manifest["deadline"])
    if manifest["scope"] not in ("pilot", "training"):
        raise ValueError("campaign scope must be explicit: pilot or training")
    if not isinstance(manifest["experiments"], list) or not manifest["experiments"]:
        raise ValueError("experiments must be a nonempty explicit list")
    experiments = []
    ids: set[str] = set()
    paths: dict[str, set[str]] = {key: set() for key in ("run_dir", "probe_dir", "eval_path")}
    baseline_paths: dict[tuple[str, str], str] = {}
    baseline_owners: dict[str, tuple[str, str]] = {}
    for raw in manifest["experiments"]:
        entry = dict(raw)
        for key in ("id", "model", "max_codes", "attention_mode", "scope", "status", *PATH_FIELDS):
            if key not in entry:
                raise ValueError(f"experiment is missing {key}")
        if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in ids:
            raise ValueError("experiment IDs must be nonempty and unique")
        ids.add(entry["id"])
        if entry["scope"] not in ("pilot", "training"):
            raise ValueError(f"{entry['id']}: scope must be pilot or training")
        if isinstance(entry["max_codes"], bool) or not isinstance(entry["max_codes"], int) or entry["max_codes"] < 256:
            raise ValueError(f"{entry['id']}: max_codes must support every requested budget through 256")
        if entry["attention_mode"] not in ("separate", "joint"):
            raise ValueError(f"{entry['id']}: invalid attention_mode")
        for key in PATH_FIELDS:
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"{entry['id']}: {key} must be nonempty")
            entry[key] = str((path.parent / entry[key]).resolve())
            if key in paths:
                if entry[key] in paths[key]:
                    raise ValueError(f"experiments cannot share {key}")
                paths[key].add(entry[key])
        pair = (entry["model"], entry["cache_dir"])
        baseline = entry["original_probe_dir"]
        if pair in baseline_paths and baseline_paths[pair] != baseline:
            raise ValueError("one model/cache must share a single original_probe_dir")
        if baseline in baseline_owners and baseline_owners[baseline] != pair:
            raise ValueError("different model/cache pairs cannot share original_probe_dir")
        baseline_paths[pair], baseline_owners[baseline] = baseline, pair
        overrides = entry.get("config_overrides", [])
        if not isinstance(overrides, list) or not all(isinstance(v, str) for v in overrides):
            raise ValueError("config_overrides must be an explicit list of train --set assignments")
        experiments.append(entry)
    if set(baseline_owners) & paths["probe_dir"]:
        raise ValueError("original and reconstructed/compressed probes need distinct directories")
    protocol = {**PROBE_DEFAULTS, **manifest.get("probe_protocol", {})}
    if set(protocol) != set(PROBE_DEFAULTS):
        raise ValueError("unknown probe_protocol option")
    for key in ("epochs", "batch_size", "seed"):
        if isinstance(protocol[key], bool) or not isinstance(protocol[key], int) or protocol[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid probe_protocol.{key}")
    if not math.isfinite(protocol["lr"]) or protocol["lr"] <= 0 or not math.isfinite(protocol["weight_decay"]) or protocol["weight_decay"] < 0:
        raise ValueError("invalid probe learning rate or weight decay")
    manifest["probe_protocol"] = protocol
    for key, default in (("cleanup_seconds", 30.0), ("termination_seconds", 2.0)):
        value = manifest.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite duration")
        manifest[key] = float(value)
    return manifest, experiments


def select_experiments(experiments: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("--experiment-ids must be a nonempty unique assigned subset")
    by_id = {entry["id"]: entry for entry in experiments}
    unknown = set(ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown assigned experiments: {sorted(unknown)}")
    # Stable model order within each K; all K256 experiments precede larger K.
    return sorted((by_id[key] for key in ids), key=lambda entry: (entry["max_codes"], experiments.index(entry)))


def slot_boundaries(now: datetime, deadline: datetime, weights: list[float]) -> tuple[datetime, datetime, datetime]:
    """Allocate a weighted remaining slot; absolute boundaries carry unused time."""
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("remaining experiment weights must be positive")
    seconds = max(0.0, (deadline - now).total_seconds()) * weights[0] / sum(weights)
    return tuple(now + timedelta(seconds=seconds * fraction) for fraction in (0.22, 0.70, 1.0))


@contextmanager
def file_lock(path: Path, deadline: datetime) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = (deadline - utcnow()).total_seconds()
                if remaining <= 0:
                    raise DeadlineExceeded(f"deadline waiting for lock {path}")
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def update_state(root: Path, manifest: dict[str, Any], record: dict[str, Any]) -> None:
    # A tiny post-deadline grace applies only to local atomic state, never children.
    with file_lock(root / ".state.lock", utcnow() + timedelta(seconds=5)):
        path = root / "state.json"
        state = read_json(path) if path.exists() else {
            "format": STATE_FORMAT, "campaign_id": manifest["campaign_id"],
            "deadline": manifest["deadline"], "experiments": {},
        }
        if state.get("format") != STATE_FORMAT or state.get("campaign_id") != manifest["campaign_id"]:
            raise ValueError("existing state belongs to an incompatible campaign")
        stamp = utcnow().isoformat()
        record["updated_at"] = stamp
        state["updated_at"] = stamp
        state["experiments"][record["id"]] = record
        atomic_json(path, state)


def run_child(command: list[str], *, hard_deadline: datetime, log_path: Path,
              termination_seconds: float = 2.0) -> dict[str, Any]:
    """Bound a child and only its new process group, including descendant cleanup."""
    started = utcnow()
    remaining = (hard_deadline - started).total_seconds()
    if remaining <= termination_seconds:
        return {"status": "deadline", "returncode": None, "reason": "no time to start child", "command": command}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        stop_at = hard_deadline - timedelta(seconds=termination_seconds)
        try:
            while process.poll() is None:
                remaining = (stop_at - utcnow()).total_seconds()
                if remaining <= 0:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=min(termination_seconds / 2,
                                                 max(0.001, (hard_deadline - utcnow()).total_seconds())))
                    except subprocess.TimeoutExpired:
                        pass
                    break
                try:
                    process.wait(timeout=min(1.0, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # A normally exiting leader can also leave descendants behind.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=max(0.001, (hard_deadline - utcnow()).total_seconds()))
    return {"status": "deadline" if timed_out else ("completed" if process.returncode == 0 else "failed"),
            "returncode": process.returncode, "command": command, "log": str(log_path),
            "started_at": started.isoformat(), "finished_at": utcnow().isoformat(),
            "reason": "child exceeded wall-clock allocation" if timed_out else None}


def snapshot_model(source: Path, run_dir: Path) -> tuple[Path, str]:
    """Preserve an atomic export using a hardlink or an independent atomic copy."""
    digest = sha256_file(source)
    destination = run_dir / "campaign-checkpoints" / digest / "model.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError(f"{destination}: incompatible immutable checkpoint snapshot")
        return destination, digest
    try:
        os.link(source, destination)
    except FileExistsError:
        if sha256_file(destination) != digest:
            raise ValueError(f"{destination}: incompatible immutable checkpoint snapshot")
    except OSError as error:
        if error.errno not in {errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS}:
            raise
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".model-copy-", suffix=".tmp",
                                             dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(temporary) != digest:
                raise ValueError(f"{source}: checkpoint changed or was corrupted while copying")
            if destination.exists():
                if sha256_file(destination) != digest:
                    raise ValueError(f"{destination}: incompatible immutable checkpoint snapshot")
            else:
                os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return destination, digest


def inspect_stage(entry: dict[str, Any], config: dict[str, Any], stage: str,
                  cache_identity: str, device: str, *, cache_layer: int | None = None) -> dict[str, Any]:
    """Validate actual checkpoint provenance, not just a summary or model filename."""
    import torch
    from compression.model import CHECKPOINT_FORMAT, MODEL_FORMAT, ChannelStats
    from compression.train import STATS_FORMAT, build_objective, effective_precision

    directory = Path(entry["run_dir"]) / stage
    checkpoint_path, model_path = directory / "checkpoint_latest.pt", directory / "model.pt"
    result = {"status": "pending", "step": 0, "target_steps": config["stages"][stage]["max_steps"],
              "checkpoint": str(checkpoint_path), "model": str(model_path), "summary": str(directory / "summary.json")}
    if not checkpoint_path.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"{directory}: existing stage artifacts without a resumable checkpoint")
        return result
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("stage") != stage:
        raise ValueError(f"{checkpoint_path}: incompatible checkpoint format/stage")
    expected = {
        "stage": stage, "model": {"input_dim": config["feature_dim"], **config["model"]},
        "objective": build_objective(config, stage).to_dict(), "optim": config["optim"],
        "precision": effective_precision(config["precision"], torch.device(device)), "seed": config["seed"],
        "data": {"token_budget": config["data"]["token_budget"], "train_split": config["data"]["train_split"],
                 "layer": cache_layer if cache_layer is not None else config["data"]["layer"]},
        "cache_identity": cache_identity,
        "stats_digest": ChannelStats.from_dict(checkpoint["stats"]).digest(),
    }
    if checkpoint.get("compatibility") != expected:
        raise ValueError(f"{checkpoint_path}: incompatible configuration/cache; refusing reuse or overwrite")
    stats_path = Path(entry["run_dir"]) / "channel_stats.pt"
    if not stats_path.exists():
        raise ValueError(f"{checkpoint_path}: missing training channel statistics")
    stored_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    if stored_stats.get("format") != STATS_FORMAT:
        raise ValueError(f"{stats_path}: incompatible channel statistics format")
    stats = ChannelStats.from_dict(stored_stats["stats"])
    if stats.digest() != expected["stats_digest"]:
        raise ValueError(f"{checkpoint_path}: run channel statistics differ")
    if (stats.cache_identity != cache_identity or stats.split != config["data"]["train_split"]
            or stats.layer != expected["data"]["layer"] or stats.feature_dim != config["feature_dim"]
            or stats.std_floor != config["stats"]["std_floor"] or stats.max_samples != config["stats"]["max_samples"]):
        raise ValueError(f"{checkpoint_path}: incompatible training statistics selection")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > result["target_steps"]:
        raise ValueError(f"{checkpoint_path}: invalid step or target behind existing checkpoint")
    if not checkpoint.get("model_state") or (step > 0 and not checkpoint.get("optimizer_state", {}).get("state")):
        raise ValueError(f"{checkpoint_path}: missing actual model/optimizer training state")
    result.update(step=step, status="partial")
    expected_model = {key: checkpoint[key] for key in ("stage", "step", "cache", "objective", "model_config")}
    del checkpoint
    if model_path.exists():
        model = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
        if model.get("format") != MODEL_FORMAT or any(model.get(key) != value for key, value in expected_model.items() if key != "step"):
            raise ValueError(f"{model_path}: exported model has incompatible provenance")
        model_step = model.get("step")
        if isinstance(model_step, bool) or not isinstance(model_step, int) or model_step < 0 or model_step > step:
            raise ValueError(f"{model_path}: exported model step exceeds latest checkpoint")
        if not model.get("model_state") or ChannelStats.from_dict(model["stats"]).digest() != expected["stats_digest"]:
            raise ValueError(f"{model_path}: invalid export/statistics")
        result["inference_ready"] = step > 0 and model_step == step
    summary_path = directory / "summary.json"
    if step == result["target_steps"] and result.get("inference_ready") and summary_path.exists():
        summary = read_json(summary_path)
        if (summary.get("stage") == stage and summary.get("step") == step
                and summary.get("cache", {}).get("identity") == cache_identity
                and summary.get("status", "completed") == "completed"):
            result["status"] = "completed"
    return result


def probe_command(entry: dict[str, Any], manifest: dict[str, Any], *, original: bool,
                  checkpoint: Path | None, device: str, deadline: datetime) -> list[str]:
    command = [sys.executable, "-m", "compression.probe", "--cache-dir", entry["cache_dir"],
               "--run-dir", entry["original_probe_dir"] if original else entry["probe_dir"],
               "--representations", "original" if original else "reconstructed,compressed",
               "--budgets", ",".join(map(str, requested_budgets(entry["max_codes"]))),
               "--device", device, "--deadline", deadline.isoformat()]
    if entry.get("cache_layer") is not None:
        command += ["--layer", str(entry["cache_layer"])]
    if not original:
        if checkpoint is None:
            raise ValueError("reconstructed/compressed probes require a trained checkpoint")
        command += ["--checkpoint", str(checkpoint)]
    for key, value in manifest["probe_protocol"].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def requested_budgets(max_codes: int) -> list[int]:
    if max_codes < 256:
        raise ValueError("requested budget 256 exceeds max_codes; never clamp")
    return [4, 32, 64, 128, 256] + ([512] if max_codes >= 512 else []) + ([max_codes] if max_codes > 512 else [])


def evaluation_path(base: Path, digest: str, stage: str, step: int) -> Path:
    """Reuse only fingerprinted reports; retain unverifiable previous outputs."""
    def matches(path: Path) -> bool:
        report = read_json(path)
        return (report.get("checkpoint_sha256") == digest
                and report.get("checkpoint_stage") == stage and report.get("checkpoint_step") == step)

    if base.exists() and matches(base):
        return base
    versioned = base.parent / f"checkpoint-{digest}" / base.name
    candidate = base if not base.exists() and not versioned.exists() else versioned
    generation = 0
    while candidate.exists():
        if matches(candidate):
            return candidate
        generation += 1
        candidate = versioned.with_name(f"{versioned.stem}-verified-{generation}{versioned.suffix}")
    return candidate


def validate_evaluation(report: dict[str, Any], *, checkpoint: Path, digest: str, identity: str,
                        stage: str, step: int, config: dict[str, Any], budgets: list[int],
                        source_model: Path | None = None) -> None:
    expected = {"format": "ordered-compressor-eval/v1",
                "cache_identity": identity, "checkpoint_stage": stage, "checkpoint_step": step,
                "model_config": {"input_dim": config["feature_dim"], **config["model"]},
                "split": "validation", "heldout": True, "max_samples": None, "budgets": budgets}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("evaluation provenance/budgets do not match the selected checkpoint")
    sources = {str(checkpoint)} | ({str(source_model)} if source_model is not None else set())
    if report.get("checkpoint") not in sources:
        raise ValueError("evaluation references a different checkpoint path")
    if report.get("checkpoint_sha256") != digest:
        raise ValueError("evaluation checkpoint fingerprint is missing or differs; report is unverifiable")
    count = report.get("num_samples")
    metrics = report.get("metrics", [])
    if not isinstance(count, int) or count <= 0 or [metric.get("budget") for metric in metrics] != budgets:
        raise ValueError("evaluation has incomplete sample counts or budget metrics")
    for metric in metrics:
        for key in ("normalized_mse", "native_cosine", "native_mse"):
            summary = metric.get(key, {})
            mean = summary.get("mean")
            if summary.get("count") != count or not isinstance(mean, (int, float)) or not math.isfinite(mean):
                raise ValueError("evaluation contains incomplete/non-finite reconstruction metrics")


def completed_probe(entry: dict[str, Any], manifest: dict[str, Any], config: dict[str, Any], *,
                    original: bool, digest: str, device: str) -> bool:
    """Validate completed artifacts offline, including actual committed heads.

    This is also usable after the hard deadline: it never starts a process or
    extracts features. Partial runs go through the probe CLI's exact resumer.
    """
    import numpy as np
    import torch
    from compression.cache import FeatureCacheDataset
    from compression.probe import FORMAT, class_coverage, condition_name, conditions, identity, read_labels
    root = Path(entry["original_probe_dir"] if original else entry["probe_dir"])
    result_path = root / "probe_results.json"
    if not result_path.exists():
        return False
    result = read_json(result_path)
    if result.get("status") != "completed":
        return False
    stored = read_json(root / "probe_config.json")
    representations = ["original"] if original else ["reconstructed", "compressed"]
    budgets = [] if original else requested_budgets(entry["max_codes"])
    selection = {}
    feature_dim = config["feature_dim"]
    cache_identity = None
    for split in ("train", "validation"):
        dataset = FeatureCacheDataset(entry["cache_dir"], split, layer=entry.get("cache_layer"))
        try:
            cache_identity = str(dataset.identity)
            ids, labels = list(dataset.sample_ids), read_labels(dataset, list(range(len(dataset))))
            selection[split] = {"sample_ids": ids, "labels": labels, "source_samples": len(dataset),
                                "selected_samples": len(dataset), "sample_ids_sha256": identity(ids)}
        finally:
            dataset.close()
    pooling = "cls" if "dino" in entry["model"].lower() else "patch_mean"
    feature_identity = {"cache_identity": cache_identity, "layer": entry.get("cache_layer"),
                        "checkpoint_sha256": None if original else digest, "pooling": pooling,
                        "compressed_pooling": "ordered_prefix_flatten", "representations": representations,
                        "budgets": budgets, "sample_selection": selection, "dtype": "float32", "format": FORMAT}
    protocol = {**manifest["probe_protocol"], "momentum": 0.9, "schedule": "cosine", "optimizer": "SGD",
                "normalization": "train_channel_standardization", "std_floor": 1e-6, "precision": "float32",
                "class_mapping": "sorted_train_labels", "validation_accuracy_denominator": "all_validation",
                "cross_entropy_denominator": "known_validation_labels", "bounded_sample_smoke": False,
                "selection": "seeded_sha256_subset_in_cache_order", "deterministic_algorithms": True}
    expected = {"format": FORMAT, "feature_identity": feature_identity, "protocol": protocol,
                "device": device, "torch_version": str(torch.__version__)}
    if stored != expected or result.get("format") != FORMAT or result.get("protocol") != protocol:
        raise ValueError(f"{root}: completed probe identity differs; refusing foreign work")
    coverage = class_coverage(selection["train"]["labels"], selection["validation"]["labels"])
    if (result.get("cache_identity") != cache_identity or result.get("checkpoint_sha256") != feature_identity["checkpoint_sha256"]
            or result.get("sample_selection") != selection or result.get("class_coverage") != coverage):
        raise ValueError(f"{root}: completed probe result provenance differs")
    features = read_json(root / "features_manifest.json")
    specs = {}
    for split, samples in selection.items():
        count = samples["selected_samples"]
        if original:
            specs[f"{split}-original"] = [count, feature_dim]
        else:
            for budget in budgets:
                specs[f"{split}-reconstructed-{budget}"] = [count, feature_dim]
            specs[f"{split}-codes"] = [count, max(budgets), config["model"]["width"]]
    if (features.get("identity") != feature_identity or features.get("completed") is not True
            or features.get("arrays") != specs
            or features.get("rows_completed") != {split: data["selected_samples"] for split, data in selection.items()}):
        raise ValueError(f"{root}: incomplete committed probe features")
    for name, shape in specs.items():
        array = np.load(root / "features" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        try:
            if list(array.shape) != shape or array.dtype != np.float32:
                raise ValueError(f"{root}: incompatible committed probe feature array")
        finally:
            array._mmap.close()
    pairs = conditions(representations, budgets)
    metrics = {(metric["condition"], metric["budget"]): metric for metric in result.get("metrics", [])}
    if len(result.get("metrics", [])) != len(pairs) or set(metrics) != set(pairs):
        raise ValueError(f"{root}: completed probe is missing requested conditions")
    for condition, budget in pairs:
        name = condition_name(condition, budget)
        head_path = root / "heads" / f"{name}.pt"
        if not head_path.exists():
            raise ValueError(f"{root}: completed probe is missing a committed classifier")
        state = torch.load(head_path, map_location="cpu", weights_only=True, mmap=True)
        metric = metrics[(condition, budget)]
        dim = budget * config["model"]["width"] if condition == "compressed" else feature_dim
        classes = len(coverage["train_labels"])
        if (state.get("format") != FORMAT or state.get("identity") != identity(stored)
                or state.get("condition") != condition or state.get("budget") != budget
                or state.get("epoch") != protocol["epochs"] or state.get("cursor") != 0
                or state.get("feature_dim") != dim or state.get("class_labels") != coverage["train_labels"]
                or state.get("metric") != metric or metric.get("epochs_completed") != protocol["epochs"]
                or metric.get("feature_dim") != dim or metric.get("classifier_parameters") != (dim + 1) * classes
                or tuple(state["head"]["weight"].shape) != (classes, dim)
                or tuple(state["head"]["bias"].shape) != (classes,)):
            raise ValueError(f"{root}: completed probe is missing a committed classifier")
        normalization = torch.load(root / "heads" / f"{name}-normalization.pt",
                                   map_location="cpu", weights_only=True, mmap=True)
        if (normalization.get("identity") != identity(stored) or normalization.get("condition") != condition
                or normalization.get("budget") != budget or tuple(normalization["mean"].shape) != (dim,)
                or tuple(normalization["std"].shape) != (dim,)):
            raise ValueError(f"{root}: classifier training normalization is incompatible")
    return True


def run_experiment(entry: dict[str, Any], manifest: dict[str, Any], root: Path, *, device: str,
                   boundaries: tuple[datetime, datetime, datetime], original_done: set[str]) -> dict[str, Any]:
    from compression.cache import FeatureCacheDataset
    from compression.train import load_config

    hard_deadline = aware_datetime(manifest["deadline"])
    fixed_end, prefix_end, slot_end = boundaries
    cleanup = manifest["cleanup_seconds"]
    record: dict[str, Any] = {"id": entry["id"], "stage": "preflight", "status": "running",
        "actual_completed_steps": {"fixed": None, "prefix": None}, "stages": {}, "errors": [],
        "artifacts": {key: entry[key] for key in ("run_dir", "probe_dir", "original_probe_dir", "eval_path")}}

    def publish() -> None:
        update_state(root, manifest, record)

    def child(command: list[str], stage: str, end: datetime) -> dict[str, Any]:
        record.update(stage=stage, status="running")
        publish()
        result = run_child(command, hard_deadline=min(end, hard_deadline),
                           log_path=Path(entry["run_dir"]) / f"campaign-{stage}.log",
                           termination_seconds=manifest["termination_seconds"])
        record.setdefault("children", {})[stage] = result
        if result["status"] == "failed":
            record["errors"].append({"stage": stage, "message": f"child exited {result['returncode']}; see {result['log']}",
                                     "at": utcnow().isoformat()})
        return result

    def original_baseline(config: dict[str, Any]) -> None:
        original_path = entry["original_probe_dir"]
        result_path = Path(original_path) / "probe_results.json"
        if original_path in original_done:
            record["original_probe"] = {"status": "completed", "reused": True, "path": str(result_path)}
            return
        baseline = Path(original_path)
        with file_lock(baseline.with_name(f".{baseline.name}.campaign.lock"), hard_deadline):
            reused = completed_probe(entry, manifest, config, original=True, digest="", device=device)
            if reused:
                record["original_probe"] = {"status": "completed", "reused": True, "path": str(result_path)}
            else:
                remaining = max(0.0, (slot_end - utcnow()).total_seconds())
                end = utcnow() + timedelta(seconds=remaining * 0.4)
                command = probe_command(entry, manifest, original=True, checkpoint=None, device=device,
                                        deadline=end - timedelta(seconds=cleanup))
                outcome = child(command, "original_probe", end)
                result = read_json(result_path) if result_path.exists() else {}
                complete = completed_probe(entry, manifest, config, original=True, digest="", device=device)
                record["original_probe"] = {"status": "completed" if complete else (
                    outcome["status"] if outcome["status"] != "completed" else result.get("status", "failed")),
                    "path": str(result_path), "reason": result.get("error") or outcome.get("reason")}
            if record["original_probe"]["status"] == "completed":
                original_done.add(original_path)

    try:
        with file_lock(root / f".experiment-{hashlib.sha256(entry['id'].encode()).hexdigest()[:16]}.lock", hard_deadline):
            config = load_config(entry["config_path"], entry.get("config_overrides", []))
            if config["model"]["max_codes"] != entry["max_codes"] or config["model"]["attention_mode"] != entry["attention_mode"]:
                raise ValueError("manifest max_codes/attention_mode does not match resolved training config")
            if config["data"]["model"] != entry["model"]:
                raise ValueError("manifest model does not match config data.model")
            if config["data"]["train_split"] != "train" or config["data"]["eval_split"] != "validation":
                raise ValueError("campaign requires the official train/validation splits without resplitting")
            dataset = FeatureCacheDataset(entry["cache_dir"], config["data"]["train_split"], layer=config["data"]["layer"])
            try:
                identity = str(dataset.identity)
                cache_layer = dataset.layer
                if dataset.feature_dim != config["feature_dim"] or not len(dataset):
                    raise ValueError("cache feature dimension mismatch or empty training split")
                record.update(cache_identity=identity, train_samples=len(dataset), scope=entry["scope"])
            finally:
                dataset.close()
            entry = {**entry, "cache_layer": cache_layer}
            for stage, end in (("fixed", fixed_end), ("prefix", prefix_end)):
                record["stage"] = stage
                stage_info = inspect_stage(entry, config, stage, identity, device, cache_layer=cache_layer)
                record["stages"][stage] = stage_info
                record["actual_completed_steps"][stage] = stage_info["step"]
                publish()
                if stage_info["status"] == "completed":
                    stage_info["reused"] = True
                    continue
                if utcnow() + timedelta(seconds=cleanup) >= end:
                    stage_info.update(status="deadline", reason="no remaining stage allocation")
                    continue
                if stage == "prefix" and not record["stages"]["fixed"].get("inference_ready"):
                    stage_info.update(status="deadline", reason="no trained fixed checkpoint to initialize prefix")
                    continue
                command = [sys.executable, "-m", "compression.train", "--config", entry["config_path"],
                           "--stage", stage, "--cache-dir", entry["cache_dir"], "--run-dir", entry["run_dir"],
                           "--device", device, "--deadline", (end - timedelta(seconds=cleanup)).isoformat()]
                for override in entry.get("config_overrides", []):
                    command += ["--set", override]
                if Path(stage_info["checkpoint"]).exists():
                    command.append("--resume")
                outcome = child(command, stage, end)
                stage_info = inspect_stage(entry, config, stage, identity, device, cache_layer=cache_layer)
                if stage_info["status"] != "completed":
                    stage_info.update(status="deadline", reason=outcome.get("reason") or "training wall-clock allocation exhausted")
                record["stages"][stage] = stage_info
                record["actual_completed_steps"][stage] = stage_info["step"]
                publish()
                if outcome["status"] == "failed":
                    stage_info.update(status="failed", reason=f"training child exited {outcome['returncode']}")
                    publish()

            # The native baseline is independent of compressor success.
            original_baseline(config)
            publish()
            available = [stage for stage in ("prefix", "fixed") if record["stages"][stage].get("inference_ready")]
            if not available:
                failed = any(info["status"] == "failed" for info in record["stages"].values())
                record["evaluation"] = record["probes"] = {
                    "status": "failed" if failed else "deadline", "reason": "no positive-step trained checkpoint"}
                if failed:
                    raise RuntimeError("training failed; original baseline retained, reconstruction/probes unavailable")
                raise DeadlineExceeded("no positive-step trained checkpoint is available for evaluation/probes")
            chosen = available[0]
            source = Path(record["stages"][chosen]["model"])
            checkpoint, checkpoint_sha = snapshot_model(source, Path(entry["run_dir"]))
            # A resumed checkpoint gets new outputs; exact old probe identities survive.
            version = f"checkpoint-{checkpoint_sha}"
            eval_base, probe_base = Path(entry["eval_path"]), Path(entry["probe_dir"])
            eval_path = evaluation_path(eval_base, checkpoint_sha, chosen, record["actual_completed_steps"][chosen])
            versioned_probe = probe_base / version
            probe_dir = versioned_probe if versioned_probe.exists() else probe_base
            if (probe_base / "probe_config.json").exists():
                previous_probe = read_json(probe_base / "probe_config.json")
                if previous_probe.get("feature_identity", {}).get("checkpoint_sha256") != checkpoint_sha:
                    probe_dir = versioned_probe
            entry = {**entry, "probe_dir": str(probe_dir)}
            record["artifacts"].update(checkpoint=record["stages"][chosen]["checkpoint"], model=str(checkpoint),
                                       training_model=str(source), probe_dir=str(probe_dir), eval_path=str(eval_path))
            record["checkpoint_sha256"] = checkpoint_sha
            budgets = requested_budgets(entry["max_codes"])
            if eval_path.exists():
                report = read_json(eval_path)
                validate_evaluation(report, checkpoint=checkpoint, digest=checkpoint_sha, identity=identity,
                                    stage=chosen, step=record["actual_completed_steps"][chosen], config=config, budgets=budgets,
                                    source_model=source)
                record["evaluation"] = {"status": "completed", "reused": True, "path": str(eval_path)}
            else:
                remaining = max(0.0, (slot_end - utcnow()).total_seconds())
                eval_end = utcnow() + timedelta(seconds=remaining * 0.15)
                command = [sys.executable, "-m", "compression.evaluate", "--checkpoint", str(checkpoint),
                           "--cache-dir", entry["cache_dir"], "--budgets", ",".join(map(str, budgets)),
                           "--device", device, "--output", str(eval_path)]
                outcome = child(command, "evaluation", eval_end)
                record["evaluation"] = {"status": outcome["status"], "path": str(eval_path)}
                if eval_path.exists():
                    report = read_json(eval_path)
                    # This fresh report came from the immutable snapshot command
                    # above; never infer this fingerprint for an existing report.
                    if "checkpoint_sha256" not in report:
                        report["checkpoint_sha256"] = checkpoint_sha
                    validate_evaluation(report, checkpoint=checkpoint, digest=checkpoint_sha, identity=identity,
                                        stage=chosen, step=record["actual_completed_steps"][chosen], config=config, budgets=budgets)
                    record["evaluation"]["status"] = "completed"
                    atomic_json(eval_path, report)
                elif outcome["status"] == "completed":
                    raise RuntimeError("evaluation exited successfully without a committed report")
            publish()

            probe_path = Path(entry["probe_dir"]) / "probe_results.json"
            reused = completed_probe(entry, manifest, config, original=False, digest=checkpoint_sha, device=device)
            if reused:
                record["probes"] = {"status": "completed", "reused": True, "path": str(probe_path)}
            else:
                command = probe_command(entry, manifest, original=False, checkpoint=checkpoint, device=device,
                                        deadline=slot_end - timedelta(seconds=cleanup))
                outcome = child(command, "probes", slot_end)
                result = read_json(probe_path) if probe_path.exists() else {}
                complete = completed_probe(entry, manifest, config, original=False, digest=checkpoint_sha, device=device)
                record["probes"] = {"status": "completed" if complete else (
                    outcome["status"] if outcome["status"] != "completed" else result.get("status", "failed")),
                    "path": str(probe_path), "reason": result.get("error") or outcome.get("reason")}
            required = [record["stages"][stage]["status"] for stage in ("fixed", "prefix")]
            required += [record[key]["status"] for key in ("evaluation", "original_probe", "probes")]
            final_status = "failed" if "failed" in required else ("completed" if all(status == "completed" for status in required) else "deadline")
            record.update(stage="finished", status=final_status,
                          reason=None if final_status == "completed" else (
                              "one or more evaluation/probe stages failed" if final_status == "failed" else
                              "wall-clock-limited stages; actual steps and partial metrics reported"))
    except Exception as error:
        record["status"] = "deadline" if isinstance(error, DeadlineExceeded) else "failed"
        record["reason"] = str(error)
        record["errors"].append({"stage": record["stage"], "message": str(error), "at": utcnow().isoformat()})
    publish()
    return record


def run_campaign(manifest_path: Path | str, experiment_ids: list[str], *, device: str = "cuda") -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest, experiments = load_manifest(manifest_path)
    selected = select_experiments(experiments, experiment_ids)
    deadline = aware_datetime(manifest["deadline"])
    records = []
    original_done: set[str] = set()
    for index, entry in enumerate(selected):
        remaining = selected[index:]
        boundaries = slot_boundaries(utcnow(), deadline, [item["max_codes"] / 256 for item in remaining])
        records.append(run_experiment(entry, manifest, manifest_path.parent, device=device,
                                      boundaries=boundaries, original_done=original_done))
    return {"format": STATE_FORMAT, "campaign_id": manifest["campaign_id"], "deadline": manifest["deadline"],
            "experiments": {record["id"]: record for record in records}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-ids", required=True, help="comma-separated explicit assigned subset")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run_campaign(args.manifest, [value.strip() for value in args.experiment_ids.split(",") if value.strip()], device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if any(entry["status"] == "failed" for entry in result["experiments"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
