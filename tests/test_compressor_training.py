"""Behavioral tests for compressor training/evaluation on a real (tiny) feature cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from compression import evaluate as evaluate_cli
from compression import train as train_cli
from compression.cache import FeatureCacheDataset, FeatureCacheWriter, cache_identity
from compression.model import load_compressor
from compression.train import (
    BatchStream,
    build_objective,
    compute_channel_stats,
    load_config,
    masked_sample_mse,
    objective_loss,
    plan_token_batches,
    train_stage,
)

DIM = 12
REPO = Path(__file__).resolve().parents[1]
GRIDS = [(2, 2), (3, 3), (2, 3), (3, 2)]


def write_toy_cache(root: Path, *, tag: str = "toy", train: int = 10, validation: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    records = [{"sample_id": f"train-{i:03d}", "split": "train", "layer": None} for i in range(train)]
    records += [{"sample_id": f"val-{i:03d}", "split": "validation", "layer": None} for i in range(validation)]
    provenance = {"model": "toy", "tag": tag, "storage_dtype": "float16"}
    output = root / "toy" / cache_identity(provenance, records)
    writer = FeatureCacheWriter(output, provenance, records, shard_tokens=32)
    offsets = rng.normal(0.0, 3.0, DIM)
    scales = rng.uniform(0.5, 4.0, DIM)
    for index, record in enumerate(records):
        rows, cols = GRIDS[index % len(GRIDS)]
        positions = [(0, r, c) for r in range(rows) for c in range(cols)]
        types = [0] * len(positions)
        if index % 2 == 0:  # CLS + two registers, as DINO-style special tokens
            positions = [(0, 0, 0), (0, 0, 0), (0, 0, 1)] + positions
            types = [1, 2, 2] + types
        features = rng.normal(size=(len(positions), DIM)) * scales + offsets
        writer.add(
            sample_id=record["sample_id"],
            split=record["split"],
            features=features.astype(np.float32),
            positions=np.asarray(positions, dtype=np.int64),
            token_types=np.asarray(types, dtype=np.int64),
            grid=(1, rows, cols),
        )
    writer.finish()
    return output


@pytest.fixture(scope="module")
def toy_cache(tmp_path_factory) -> Path:
    return write_toy_cache(tmp_path_factory.mktemp("cache"))


def tiny_config(*extra: str) -> dict:
    return load_config(
        None,
        [
            "name=toy",
            f"feature_dim={DIM}",
            "model.width=16",
            "model.max_codes=4",
            "model.encoder_layers=1",
            "model.decoder_layers=1",
            "model.heads=2",
            "model.ff_multiplier=2",
            "data.token_budget=40",
            "data.num_workers=0",
            "precision=fp32",
            "optim.warmup_steps=2",
            "log_every=1",
            "checkpoint_every=2",
            *extra,
        ],
    )


def run(config: dict, stage: str, cache: Path, run_dir: Path, **kwargs) -> dict:
    return train_stage(config, stage=stage, cache_dir=cache, run_dir=run_dir, device="cpu",
                       log=lambda message: None, **kwargs)


def model_state(path: Path) -> dict[str, torch.Tensor]:
    return load_compressor(path).model.state_dict()


def assert_same_state(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> None:
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


# -- objective ----------------------------------------------------------------


def test_masked_sample_mse_weights_samples_equally_and_ignores_padding():
    target = torch.zeros(2, 6, 3)
    prediction = torch.zeros(2, 6, 3)
    prediction[0, :2] = 1.0  # sample 0: 2 tokens, squared error 1
    prediction[1, :6] = 3.0 ** 0.5  # sample 1: 6 tokens, squared error 3
    valid = torch.zeros(2, 6, dtype=torch.bool)
    valid[0, :2] = True
    valid[1] = True
    prediction[0, 2:] = float("nan")  # padding garbage
    per_sample = masked_sample_mse(prediction, target, valid)
    torch.testing.assert_close(per_sample, torch.tensor([1.0, 3.0]))
    assert per_sample.mean().item() == pytest.approx(2.0)  # token-weighted would be 2.75


def test_default_prefix_objective_is_half_mean_short_plus_half_full():
    objective = build_objective(tiny_config("model.max_codes=8"), "prefix")
    weights = objective.expected_weights()
    assert weights[8] == pytest.approx(0.5)
    for k in range(1, 8):
        assert weights[k] == pytest.approx(0.5 / 7)
    assert sum(weights.values()) == pytest.approx(1.0)
    draws = [objective.sample_short_budget(0, step) for step in range(7000)]
    assert draws == [objective.sample_short_budget(0, step) for step in range(7000)]
    counts = np.bincount(draws, minlength=9)
    assert counts[0] == 0 and counts[8] == 0
    assert np.all(np.abs(counts[1:8] - 1000) < 150)
    assert draws != [objective.sample_short_budget(1, step) for step in range(7000)]


def test_custom_ladder_weights_match_sampling_and_special_cases():
    objective = build_objective(
        tiny_config("model.max_codes=8", "stages.prefix.short_budgets=[4,1,2]",
                    "stages.prefix.short_weights=[1,2,1]"),
        "prefix",
    )
    assert objective.short_budgets == (1, 2, 4)
    assert objective.short_probabilities == pytest.approx((0.5, 0.25, 0.25))
    assert objective.expected_weights() == pytest.approx({1: 0.25, 2: 0.125, 4: 0.125, 8: 0.5})
    draws = np.array([objective.sample_short_budget(3, step) for step in range(8000)])
    assert np.mean(draws == 1) == pytest.approx(0.5, abs=0.03)

    single = build_objective(tiny_config("model.max_codes=1"), "prefix")
    assert single.expected_weights() == {1: 1.0}
    assert single.sample_short_budget(0, 0) is None
    fixed = build_objective(tiny_config("stages.fixed.budget=3"), "fixed")
    assert fixed.encode_budget == 3 and fixed.expected_weights() == {3: 1.0}
    for bad in (["stages.prefix.short_budgets=[1,4]"], ["stages.prefix.short_budgets=[1,1]"],
                ["stages.prefix.short_budgets=[1,2]", "stages.prefix.short_weights=[1]"],
                ["stages.prefix.full_weight=0"]):
        with pytest.raises(ValueError):
            tiny_config(*bad)


@pytest.mark.parametrize("mode", ["separate", "joint"])
def test_objective_loss_equals_weighted_independent_prefix_distortions(toy_cache, mode):
    config = tiny_config(f"model.attention_mode={mode}", "model.max_codes=6")
    objective = build_objective(config, "prefix")
    dataset = FeatureCacheDataset(toy_cache, "train")
    batch = train_cli.collate_features([dataset[i] for i in range(5)])
    torch.manual_seed(0)
    model = train_cli.build_model({"input_dim": DIM, **config["model"]})
    valid = batch["valid_mask"]
    x = torch.where(valid[..., None], batch["features"] / 4.0, 0.0)
    args = (x, batch["positions"], batch["token_types"], valid)
    with torch.no_grad():
        loss, parts = objective_loss(model, objective, *args, short_budget=2)

        def distortion(k: int) -> torch.Tensor:
            prediction = model.decode(model.encode(*args, num_codes=k), *args[1:])
            return masked_sample_mse(prediction, x, valid).mean()

        torch.testing.assert_close(parts["d_full"], distortion(6), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(parts["d_short"], distortion(2), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(loss, 0.5 * distortion(6) + 0.5 * distortion(2), atol=1e-5, rtol=1e-5)
        fixed_loss, fixed_parts = objective_loss(model, build_objective(config, "fixed"), *args, short_budget=None)
        assert set(fixed_parts) == {"d_full"}
        torch.testing.assert_close(fixed_loss, distortion(6), atol=1e-5, rtol=1e-5)


# -- data and statistics --------------------------------------------------------


def test_channel_stats_stream_train_tokens_only(toy_cache):
    train = FeatureCacheDataset(toy_cache, "train")
    stats = compute_channel_stats(train, split="train", layer=None, std_floor=1e-6)
    tokens = np.concatenate([train[i]["features"].numpy().astype(np.float64) for i in range(len(train))])
    np.testing.assert_allclose(stats.mean.numpy(), tokens.mean(axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(stats.std.numpy(), tokens.std(axis=0), rtol=1e-10, atol=1e-10)
    assert stats.token_count == tokens.shape[0] and stats.sample_count == len(train)
    assert stats.cache_identity == train.identity

    validation = FeatureCacheDataset(toy_cache, "validation")
    everything = np.concatenate([tokens] + [validation[i]["features"].numpy().astype(np.float64)
                                            for i in range(len(validation))])
    assert not np.allclose(stats.mean.numpy(), everything.mean(axis=0), atol=1e-6)

    subset = compute_channel_stats(train, split="train", layer=None, std_floor=1e-6, max_samples=3)
    chosen = np.concatenate([train[i]["features"].numpy().astype(np.float64) for i in (0, 4, 9)])
    np.testing.assert_allclose(subset.mean.numpy(), chosen.mean(axis=0), rtol=1e-10, atol=1e-10)
    assert subset.sample_count == 3


def test_token_budget_batches_are_deterministic_complete_and_bounded():
    counts = [7, 12, 9, 7, 30, 12, 9, 4, 4, 11, 5]
    first = plan_token_batches(counts, 24, seed=5, epoch=0)
    assert first == plan_token_batches(counts, 24, seed=5, epoch=0)
    assert first != plan_token_batches(counts, 24, seed=5, epoch=1)
    assert sorted(i for batch in first for i in batch) == list(range(len(counts)))
    for batch in first:
        assert len(batch) == 1 or len(batch) * max(counts[i] for i in batch) <= 24
    assert [4] in first  # the oversized sample forms its own batch


def test_batch_stream_resumes_from_cursor(toy_cache):
    dataset = FeatureCacheDataset(toy_cache, "train")
    stream = BatchStream(dataset, 40, seed=2)
    iterator = iter(stream)
    reference = [next(iterator)["sample_ids"] for _ in range(9)]
    iterator.close()
    stream = BatchStream(dataset, 40, seed=2)
    iterator = iter(stream)
    for _ in range(4):
        next(iterator)
    cursor = (stream.epoch, stream.batch)
    iterator.close()
    resumed = iter(BatchStream(dataset, 40, seed=2, epoch=cursor[0], batch=cursor[1]))
    assert [next(resumed)["sample_ids"] for _ in range(5)] == reference[4:]
    assert stream.batches_per_epoch(0) < 9  # the reference crossed an epoch boundary


# -- training, resume, initialisation ------------------------------------------


@pytest.mark.parametrize("mode", ["separate", "joint"])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_interrupted_and_resumed_stages_match_uninterrupted_training(toy_cache, tmp_path, mode, dropout):
    config = tiny_config(f"model.attention_mode={mode}", f"model.dropout={dropout}")
    straight, split = tmp_path / "straight", tmp_path / "split"
    for run_dir in (straight, split):
        run(config, "fixed", toy_cache, run_dir, max_steps=4)
    assert_same_state(model_state(straight / "fixed" / "model.pt"), model_state(split / "fixed" / "model.pt"))

    run(config, "prefix", toy_cache, straight, max_steps=9)
    run(config, "prefix", toy_cache, split, max_steps=1)
    interrupted = torch.load(split / "prefix" / "checkpoint_latest.pt", weights_only=True)
    assert interrupted["cursor"] == {"epoch": 0, "batch": 1}
    summary = run(config, "prefix", toy_cache, split, max_steps=9, resume=True)
    assert summary["step"] == 9
    assert_same_state(model_state(straight / "prefix" / "model.pt"), model_state(split / "prefix" / "model.pt"))
    first = torch.load(straight / "prefix" / "checkpoint_latest.pt", weights_only=True)
    second = torch.load(split / "prefix" / "checkpoint_latest.pt", weights_only=True)
    assert first["cursor"] == second["cursor"] and first["step"] == second["step"] == 9
    for a, b in zip(first["optimizer_state"]["state"].values(), second["optimizer_state"]["state"].values()):
        assert torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])
    log = [json.loads(line) for line in (split / "prefix" / "train_log.jsonl").read_text().splitlines()]
    assert [entry["step"] for entry in log] == list(range(1, 10))
    assert all("d_short" in entry and 1 <= entry["mean_short_budget"] < 4 for entry in log)


def test_prefix_stage_initialises_from_fixed_control(toy_cache, tmp_path):
    config = tiny_config()
    run(config, "fixed", toy_cache, tmp_path, max_steps=3)
    summary = run(config, "prefix", toy_cache, tmp_path, max_steps=0)
    assert summary["init_from"]["stage"] == "fixed" and summary["init_from"]["step"] == 3
    assert_same_state(model_state(tmp_path / "fixed" / "model.pt"), model_state(tmp_path / "prefix" / "model.pt"))
    with pytest.raises(ValueError, match="model config"):
        run(tiny_config("model.width=32"), "prefix", toy_cache, tmp_path / "other", max_steps=0,
            init_from=tmp_path / "fixed" / "model.pt")


def test_existing_or_incompatible_checkpoints_are_never_overwritten(toy_cache, tmp_path):
    config = tiny_config()
    run(config, "fixed", toy_cache, tmp_path, max_steps=2)
    checkpoint = tmp_path / "fixed" / "checkpoint_latest.pt"
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        run(config, "fixed", toy_cache, tmp_path, max_steps=2)
    with pytest.raises(ValueError, match="optim"):
        run(tiny_config("optim.lr=0.001"), "fixed", toy_cache, tmp_path, max_steps=4, resume=True)
    with pytest.raises(ValueError, match="model"):
        run(tiny_config("model.max_codes=8"), "fixed", toy_cache, tmp_path, max_steps=4, resume=True)
    with pytest.raises(ValueError, match="computed for"):
        run(tiny_config("stats.max_samples=2"), "fixed", toy_cache, tmp_path, max_steps=4, resume=True)
    other_cache = write_toy_cache(tmp_path / "other-cache", tag="other")
    with pytest.raises(ValueError):
        run(config, "fixed", other_cache, tmp_path, max_steps=4, resume=True)
    assert checkpoint.read_bytes() == before
    run(config, "fixed", toy_cache, tmp_path, max_steps=3, resume=True)
    assert torch.load(checkpoint, weights_only=True)["step"] == 3


# -- evaluation and CLIs ---------------------------------------------------------


def test_cli_training_and_heldout_prefix_evaluation(toy_cache, tmp_path):
    cache_root = toy_cache.parent.parent
    run_root = tmp_path / "runs"
    common = ["--config", str(REPO / "configs/compression/qwen38.json"), "--cache-root", str(cache_root),
              "--run-root", str(run_root), "--device", "cpu", "--set", f"feature_dim={DIM}",
              "--set", "data.model=toy", "--set", "model.width=16", "--set", "model.max_codes=4",
              "--set", "model.encoder_layers=1", "--set", "model.decoder_layers=1", "--set", "model.heads=2",
              "--set", "data.token_budget=40", "--set", "data.num_workers=0", "--set", "eval.budgets=[1,2,4]"]
    assert train_cli.main(common + ["--stage", "fixed", "--max-steps", "2"]) == 0
    assert train_cli.main(common + ["--stage", "prefix", "--max-steps", "3"]) == 0
    run_dir = run_root / "qwen38_vision_block26"
    model_path = run_dir / "prefix" / "model.pt"
    output = tmp_path / "eval.json"
    assert evaluate_cli.main(["--checkpoint", str(model_path), "--cache-dir", str(toy_cache), "--device", "cpu",
                              "--budgets", "1,2,3,4", "--output", str(output), "--per-sample"]) == 0
    report = json.loads(output.read_text())
    assert report["split"] == "validation" and report["heldout"] is True
    assert report["num_samples"] == 4 and report["budgets"] == [1, 2, 3, 4]
    assert [m["code_scalars"] for m in report["metrics"]] == [16, 32, 48, 64]
    assert set(report["metrics"][0]["normalized_mse_by_token_type"]) == {"patch", "cls", "register"}

    bundle = load_compressor(model_path)
    dataset = FeatureCacheDataset(toy_cache, "validation")
    with torch.no_grad():
        for column, budget in enumerate(report["budgets"]):
            expected = []
            for index in range(len(dataset)):
                batch = train_cli.collate_features([dataset[index]])
                x = bundle.stats.normalize(batch["features"])
                args = (batch["positions"], batch["token_types"], batch["valid_mask"])
                prediction = bundle.model.decode(bundle.model.encode(x, *args, num_codes=budget), *args)
                expected.append(masked_sample_mse(prediction, x, batch["valid_mask"]).item())
                assert report["per_sample"]["sample_ids"][index] == batch["sample_ids"][0]
            np.testing.assert_allclose([row[column] for row in report["per_sample"]["normalized_mse"]], expected,
                                       rtol=1e-4, atol=1e-6)
            assert report["metrics"][column]["normalized_mse"]["mean"] == pytest.approx(np.mean(expected), rel=1e-4)
            cosine = report["metrics"][column]["native_cosine"]["mean"]
            assert -1.0 <= cosine <= 1.0

    with pytest.raises(SystemExit):  # refuses to overwrite metrics
        evaluate_cli.main(["--checkpoint", str(model_path), "--cache-dir", str(toy_cache), "--device", "cpu",
                           "--output", str(output)])
    foreign = write_toy_cache(tmp_path / "foreign", tag="foreign")
    with pytest.raises(ValueError, match="identity"):
        evaluate_cli.main(["--checkpoint", str(model_path), "--cache-dir", str(foreign), "--device", "cpu",
                           "--output", str(tmp_path / "foreign.json")])


def test_deadline_at_safe_step_forces_log_checkpoint_export_and_exact_resume(toy_cache, tmp_path, monkeypatch):
    config = tiny_config("log_every=1000", "checkpoint_every=1000", "model.dropout=0.2")
    straight, interrupted = tmp_path / "straight", tmp_path / "interrupted"
    run(config, "fixed", toy_cache, straight, max_steps=3)
    checks = iter((False, True))  # before optimizer step, then after its safe boundary
    monkeypatch.setattr(train_cli, "_deadline_reached", lambda deadline: next(checks))
    summary = run(config, "fixed", toy_cache, interrupted, max_steps=3, deadline="2026-10-15T08:00:00+03:00")
    assert summary["status"] == "deadline" and summary["stop_reason"] == "deadline"
    assert summary["step"] == 1 and summary["max_steps"] == 3
    checkpoint = torch.load(interrupted / "fixed" / "checkpoint_latest.pt", weights_only=True)
    assert checkpoint["step"] == 1 and checkpoint["cursor"]["batch"] == 1
    assert checkpoint["stop_reason"] == "deadline"
    assert load_compressor(interrupted / "fixed" / "model.pt").metadata["step"] == 1
    rows = [json.loads(line) for line in (interrupted / "fixed" / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1]
    monkeypatch.setattr(train_cli, "_deadline_reached", lambda deadline: False)
    resumed = run(config, "fixed", toy_cache, interrupted, max_steps=3, resume=True)
    assert resumed["status"] == "completed" and resumed["stop_reason"] == "max_steps"
    assert_same_state(model_state(straight / "fixed" / "model.pt"), model_state(interrupted / "fixed" / "model.pt"))


def test_expired_deadline_saves_zero_step_without_claiming_training(toy_cache, tmp_path):
    from compression.campaign import inspect_stage
    config = tiny_config("stages.fixed.max_steps=2")
    summary = run(config, "fixed", toy_cache, tmp_path, deadline="2000-01-01T00:00:00+00:00")
    assert summary["status"] == "deadline" and summary["step"] == 0
    assert load_compressor(tmp_path / "fixed" / "model.pt").metadata["step"] == 0
    info = inspect_stage({"run_dir": str(tmp_path)}, config, "fixed", summary["cache"]["identity"], "cpu")
    assert info["status"] == "partial" and not info.get("inference_ready", False)
    assert not (tmp_path / "fixed" / "train_log.jsonl").exists()


def test_naive_deadline_rejected_before_creating_run_artifacts(toy_cache, tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        run(tiny_config(), "fixed", toy_cache, tmp_path, deadline="2026-10-15T08:00:00")
    assert not list(tmp_path.iterdir())


def test_failed_resume_summary_reports_last_persisted_step_not_new_target(toy_cache, tmp_path, monkeypatch):
    config = tiny_config()
    run(config, "fixed", toy_cache, tmp_path, max_steps=2)
    checkpoint_path = tmp_path / "fixed" / "checkpoint_latest.pt"
    before = checkpoint_path.read_bytes()
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate non-finite objective")
    monkeypatch.setattr(train_cli, "objective_loss", fail)
    with pytest.raises(FloatingPointError):
        run(config, "fixed", toy_cache, tmp_path, max_steps=5, resume=True)
    summary = json.loads((tmp_path / "fixed" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["step"] == 2 and summary["max_steps"] == 5
    assert summary["model"] is None and checkpoint_path.read_bytes() == before


def test_campaign_stage_validation_reuses_only_proven_compatible_steps(toy_cache, tmp_path):
    from compression.campaign import inspect_stage
    config = tiny_config("stages.fixed.max_steps=4")
    summary = run(config, "fixed", toy_cache, tmp_path, max_steps=2)
    entry = {"run_dir": str(tmp_path)}
    identity = summary["cache"]["identity"]
    first = inspect_stage(entry, config, "fixed", identity, "cpu")
    assert first["status"] == "partial" and first["step"] == 2 and first["inference_ready"]
    stale_export = (tmp_path / "fixed" / "model.pt").read_bytes()
    run(config, "fixed", toy_cache, tmp_path, max_steps=4, resume=True)
    complete = inspect_stage(entry, config, "fixed", identity, "cpu")
    assert complete["status"] == "completed" and complete["step"] == 4
    with pytest.raises(ValueError, match="incompatible"):
        inspect_stage(entry, tiny_config("stages.fixed.max_steps=4", "optim.lr=0.002"), "fixed", identity, "cpu")
    with pytest.raises(ValueError, match="incompatible"):
        inspect_stage(entry, config, "fixed", "foreign-cache", "cpu")
    (tmp_path / "fixed" / "model.pt").write_bytes(stale_export)
    interrupted = inspect_stage(entry, config, "fixed", identity, "cpu")
    assert interrupted["status"] == "partial" and interrupted["step"] == 4
    assert not interrupted["inference_ready"]
    run(config, "fixed", toy_cache, tmp_path, max_steps=4, resume=True)
    assert inspect_stage(entry, config, "fixed", identity, "cpu")["status"] == "completed"
