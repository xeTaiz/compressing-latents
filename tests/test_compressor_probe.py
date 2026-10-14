"""Scientific/provenance regressions for real frozen linear-probe execution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from compression import probe
from compression.cache import FeatureCacheDataset, FeatureCacheWriter
from compression.model import ChannelStats, OrderedCompressor, export_compressor, load_compressor


def labeled_cache(root: Path, *, model: str = "qwen", missing_label: bool = False,
                  train_samples: int = 6, variable_length: bool = False) -> Path:
    labels = {"train": [10 if i % 2 == 0 else 20 for i in range(train_samples)], "validation": [10, 20, 99]}
    records = [{"sample_id": f"{split}-{i}", "split": split, "layer": None}
               for split, values in labels.items() for i in range(len(values))]
    writer = FeatureCacheWriter(root, {"model": model, "storage_dtype": "float16"}, records,
                                shard_tokens=8)
    for record in records:
        split, index = record["split"], int(record["sample_id"].rsplit("-", 1)[1])
        label = labels[split][index]
        sign = -1.0 if label == 10 else 1.0
        features = np.array([[sign * 2, index * .1], [sign * 4, index * .2]], dtype=np.float32)
        if variable_length:
            count = 2 + index % 4
            features = np.array([[sign * (token + 1), index * .1 + token * .2]
                                 for token in range(count)], dtype=np.float32)
            types = np.zeros(count, dtype=np.int64)
            types[0] = 1
            if count > 2:
                types[-1] = 2
        else:
            types = np.array([0, 0], dtype=np.int64)
        count = len(features)
        positions = np.zeros((count, 3), dtype=np.int64)
        patch_count = int((types == 0).sum())
        positions[types == 0, 2] = np.arange(patch_count)
        metadata = {} if missing_label else {"manifest_record": {"label": label}}
        writer.add(sample_id=record["sample_id"], split=split, features=features,
                   positions=positions, token_types=types, grid=(1, 1, patch_count), metadata=metadata)
    writer.finish()
    return root


def arguments(cache: Path, root: Path, *extra: str):
    return probe.build_parser().parse_args([
        "--cache-dir", str(cache), "--run-dir", str(root), "--representations", "original",
        "--epochs", "3", "--batch-size", "2", "--device", "cpu", *extra])


def checkpoint(cache: Path, path: Path, *, identity: str | None = None) -> Path:
    dataset = FeatureCacheDataset(cache, "train")
    try:
        torch.manual_seed(7)
        model = OrderedCompressor(input_dim=2, width=4, max_codes=4, heads=1,
                                  encoder_layers=1, decoder_layers=1, ff_multiplier=1)
        cache_identity = identity or dataset.identity
        stats = ChannelStats(mean=torch.zeros(2, dtype=torch.float64),
                             std=torch.ones(2, dtype=torch.float64), token_count=12,
                             sample_count=6, split="train", cache_identity=cache_identity,
                             layer=None, std_floor=1e-6)
        export_compressor(path, model, stats, {"stage": "prefix", "step": 17,
                                               "cache": {"identity": cache_identity, "layer": None}})
    finally:
        dataset.close()
    return path


def test_native_pooling_matches_token_semantics_and_ignores_padding_nan():
    features = torch.tensor([[[3., 5.], [100., 200.], [7., 9.], [1000., 2000.], [float("nan"), float("nan")]]])
    types = torch.tensor([[1, 0, 0, 2, 1]])
    valid = torch.tensor([[True, True, True, True, False]])
    torch.testing.assert_close(probe.pool_native(features, types, valid, "cls"), torch.tensor([[3., 5.]]))
    torch.testing.assert_close(probe.pool_native(features, types, valid, "patch_mean"), torch.tensor([[53.5, 104.5]]))
    with pytest.raises(ValueError, match="exactly one"):
        probe.pool_native(features, types, torch.ones_like(valid), "cls")
    with pytest.raises(ValueError, match="patch tokens"):
        probe.pool_native(features, torch.ones_like(types), valid, "patch_mean")


def test_direct_codes_are_ordered_prefix_coordinates_not_means_or_decodes():
    codes = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    codes[:, 2:] = np.nan  # Discarded suffix must never reach the classifier.
    selected = probe.feature_batch(codes, np.array([2, 0]), 2)
    np.testing.assert_array_equal(selected, np.array([[16, 17, 18, 19], [0, 1, 2, 3]], dtype=np.float32))
    assert selected.shape == (2, 4)
    assert np.isfinite(selected).all()


def test_normalization_is_train_only_stable_and_constant_channels_are_finite():
    train = np.array([[1., 7.], [3., 7.], [5., 7.]], dtype=np.float32)
    mean, std = probe.fit_normalization(train, None, 2)
    np.testing.assert_allclose(mean.numpy(), train.mean(axis=0))
    np.testing.assert_allclose(std.numpy(), np.maximum(train.std(axis=0), 1e-6), rtol=1e-6)
    validation = torch.tensor([[1003., 7.]])
    normalized = (validation - mean) / std
    assert normalized[0, 0] > 500  # Validation does not recenter the training affine map.
    assert normalized[0, 1] == 0
    assert torch.isfinite(normalized).all()


def test_all_validation_accuracy_includes_unseen_classes_and_ce_is_explicit():
    head = nn.Linear(1, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.], [-1.]]))
        head.bias.zero_()
    array = np.array([[2.], [-2.], [10.]], dtype=np.float32)
    metrics = probe.validation_metrics(head, array, torch.tensor([0, 1, -1]),
                                       torch.zeros(1), torch.ones(1), None, 2, "cpu")
    assert metrics["top1"] == pytest.approx(2 / 3)
    assert metrics["top5"] == pytest.approx(2 / 3)
    assert metrics["known_label_samples"] == metrics["cross_entropy_samples"] == 2
    assert metrics["validation_samples"] == 3 and metrics["unseen_label_samples"] == 1
    expected_ce = torch.nn.functional.cross_entropy(torch.tensor([[2., -2.], [-2., 2.]]), torch.tensor([0, 1]))
    assert metrics["cross_entropy"] == pytest.approx(float(expected_ce))
    unseen = probe.validation_metrics(head, array, torch.full((3,), -1),
                                      torch.zeros(1), torch.ones(1), None, 2, "cpu")
    assert unseen["top1"] == unseen["top5"] == 0
    assert unseen["cross_entropy"] is None and unseen["cross_entropy_samples"] == 0


def test_missing_labels_fail_instead_of_synthesizing_targets(tmp_path):
    cache = labeled_cache(tmp_path / "cache", missing_label=True)
    with pytest.raises(ValueError, match="missing manifest label"):
        probe.run_probe(arguments(cache, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_seeded_subset_preserves_official_split_order_and_reports_ids(tmp_path):
    cache = labeled_cache(tmp_path / "cache")
    dataset = FeatureCacheDataset(cache, "train")
    try:
        first = probe.selected_indices(dataset, 3, 11)
        assert first == probe.selected_indices(dataset, 3, 11)
        assert first == sorted(first) and len(set(first)) == 3
        assert all(dataset.sample_ids[i].startswith("train-") for i in first)
    finally:
        dataset.close()
    result = probe.run_probe(arguments(cache, tmp_path / "run", "--epochs", "1", "--max-train-samples", "3"))
    assert result["protocol"]["bounded_sample_smoke"] is True
    assert result["train_samples"] == 3 and result["validation_samples"] == 3
    assert len(result["sample_selection"]["train"]["sample_ids"]) == 3
    assert result["sample_selection"]["train"]["sample_ids_sha256"] == probe.identity(result["sample_selection"]["train"]["sample_ids"])


def test_original_real_training_provenance_and_checkpoint_independent_reuse(tmp_path):
    cache = labeled_cache(tmp_path / "cache")
    root = tmp_path / "baseline"
    result = probe.run_probe(arguments(cache, root))
    assert result["format"] == probe.FORMAT and result["status"] == "completed"
    assert result["checkpoint"] is None and result["checkpoint_sha256"] is None
    assert result["pooling"] == "patch_mean"
    assert result["class_coverage"]["train_labels"] == [10, 20]
    assert result["class_coverage"]["unseen_validation_labels"] == [99]
    assert result["class_coverage"]["per_class"][-1] == {"label": 99, "train_samples": 0, "validation_samples": 1}
    metric = result["metrics"][0]
    assert metric["feature_dim"] == 2 and metric["classifier_parameters"] == 6
    assert metric["epochs_completed"] == 3 and metric["budget"] is None
    assert metric["validation_samples"] == 3 and metric["unseen_label_samples"] == 1
    log = [json.loads(line) for line in (root / "probe_log.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in log] == [1, 2, 3]
    assert log[-1]["lr"] == pytest.approx(.025)
    before = (root / "heads/original.pt").read_bytes()
    # A different compressor/budget is immaterial to the reusable original baseline.
    second = probe.run_probe(arguments(cache, root, "--checkpoint", str(tmp_path / "nonexistent.pt"),
                                       "--budgets", "512", "--deadline", "2000-01-01T00:00:00+00:00"))
    assert second["status"] == "completed" and second["metrics"] == result["metrics"]
    assert (root / "heads/original.pt").read_bytes() == before


def test_resume_reconciles_log_and_snapshot_from_committed_epoch(tmp_path):
    cache = labeled_cache(tmp_path / "cache")
    root = tmp_path / "run"
    result = probe.run_probe(arguments(cache, root, "--epochs", "1"))
    (root / "probe_log.jsonl").write_text('{"epoch":999}\n{"incomplete"')
    (root / "probe_results.json").write_text('{"status":"running","metrics":[]}')
    resumed = probe.run_probe(arguments(cache, root, "--epochs", "1"))
    assert resumed["metrics"] == result["metrics"]
    records = [json.loads(line) for line in (root / "probe_log.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["epoch"] == 1


def test_resume_rejects_protocol_changes_without_overwriting_work(tmp_path):
    cache = labeled_cache(tmp_path / "cache")
    root = tmp_path / "run"
    probe.run_probe(arguments(cache, root, "--epochs", "1"))
    before = (root / "probe_results.json").read_bytes()
    with pytest.raises(ValueError, match="configuration identity mismatch"):
        probe.run_probe(arguments(cache, root, "--epochs", "1", "--lr", ".2"))
    assert (root / "probe_results.json").read_bytes() == before


def test_resume_rejects_changed_checkpoint_bytes_even_same_cache_and_stage(tmp_path):
    cache = labeled_cache(tmp_path / "cache")
    model_path = checkpoint(cache, tmp_path / "model.pt")
    root = tmp_path / "run"
    args = arguments(cache, root, "--checkpoint", str(model_path), "--representations", "compressed",
                     "--budgets", "1", "--epochs", "1")
    probe.run_probe(args)
    before = (root / "probe_results.json").read_bytes()
    payload = torch.load(model_path, weights_only=True)
    # A real tensor change; filenames/stage/step/cache identity remain identical.
    key = next(iter(payload["model_state"]))
    payload["model_state"][key] = payload["model_state"][key] + .125
    torch.save(payload, model_path)
    with pytest.raises(ValueError, match="configuration identity mismatch"):
        probe.run_probe(args)
    assert (root / "probe_results.json").read_bytes() == before


def test_partial_epoch_deadline_resume_is_identical_to_uninterrupted_sgd(tmp_path, monkeypatch):
    cache = labeled_cache(tmp_path / "cache")
    full, partial = tmp_path / "full", tmp_path / "partial"
    expected = probe.run_probe(arguments(cache, full))
    original_step, original_check = torch.optim.SGD.step, probe.check_deadline
    steps = 0
    def counted_step(self, *args, **kwargs):
        nonlocal steps
        value = original_step(self, *args, **kwargs)
        steps += 1
        return value
    def boundary(deadline):
        if steps == 1:
            raise probe.ProbeDeadline()
        return original_check(deadline)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.SGD, "step", counted_step)
        patch.setattr(probe, "check_deadline", boundary)
        stopped = probe.run_probe(arguments(cache, partial))
    assert stopped["status"] == "deadline" and stopped["metrics"] == []
    saved = torch.load(partial / "heads/original.pt", weights_only=True)
    assert saved["epoch"] == 0 and saved["cursor"] == 2
    resumed = probe.run_probe(arguments(cache, partial))
    assert resumed["status"] == "completed" and resumed["metrics"] == expected["metrics"]
    a = torch.load(full / "heads/original.pt", weights_only=True)
    b = torch.load(partial / "heads/original.pt", weights_only=True)
    for key in a["head"]:
        assert torch.equal(a["head"][key], b["head"][key])
    for index in a["optimizer"]["state"]:
        assert torch.equal(a["optimizer"]["state"][index]["momentum_buffer"],
                           b["optimizer"]["state"][index]["momentum_buffer"])


def test_real_compressor_encodes_once_then_physically_slices_and_pools(tmp_path, monkeypatch):
    cache = labeled_cache(tmp_path / "cache")
    model_path = checkpoint(cache, tmp_path / "model.pt")
    bundle = load_compressor(model_path)
    real_encode, real_decode = bundle.encode_native, bundle.decode_native
    encoded, decoded = [], []
    def encode(*args, **kwargs):
        assert kwargs["num_codes"] == 4
        codes = real_encode(*args, **kwargs)
        encoded.append(codes.detach().clone())
        return codes
    def decode(codes, *args, **kwargs):
        k = codes.shape[1]
        assert k in (1, 4)
        assert torch.equal(codes, encoded[-1][:, :k])
        decoded.append(k)
        return real_decode(codes, *args, **kwargs)
    monkeypatch.setattr(bundle, "encode_native", encode)
    monkeypatch.setattr(bundle, "decode_native", decode)
    monkeypatch.setattr(probe, "load_compressor", lambda *args, **kwargs: bundle)
    root = tmp_path / "run"
    result = probe.run_probe(arguments(cache, root, "--checkpoint", str(model_path),
                                        "--representations", "original,reconstructed,compressed",
                                        "--budgets", "1,4", "--epochs", "1"))
    assert result["status"] == "completed" and len(result["metrics"]) == 5
    assert [len(batch) for batch in encoded] == [6, 3] and decoded == [1, 4] * 2
    assert result["checkpoint_stage"] == "prefix" and result["checkpoint_step"] == 17
    assert result["checkpoint_sha256"] == probe.sha256_file(model_path)
    codes = np.load(root / "features/train-codes.npy", mmap_mode="r")
    assert codes.shape == (6, 4, 4) and codes.dtype == np.float32
    np.testing.assert_array_equal(codes[0], encoded[0][0].numpy())
    assert not (root / "features/train-compressed-1.npy").exists()
    assert not (root / "features/train-compressed-4.npy").exists()
    dataset = FeatureCacheDataset(cache, "train")
    try:
        batch = probe.collate_features([dataset[0]])
        with torch.no_grad():
            reconstruction = real_decode(encoded[0][:1, :1], batch["positions"], batch["token_types"], batch["valid_mask"])
            expected = probe.pool_native(reconstruction, batch["token_types"], batch["valid_mask"], "patch_mean")
        np.testing.assert_allclose(np.load(root / "features/train-reconstructed-1.npy")[0],
                                   expected[0].numpy(), rtol=1e-5, atol=1e-6)
    finally:
        dataset.close()
        codes._mmap.close()
    compressed = [row for row in result["metrics"] if row["condition"] == "compressed"]
    assert [(row["budget"], row["feature_dim"], row["classifier_parameters"]) for row in compressed] == [(1, 4, 10), (4, 16, 34)]
    before = len(encoded)
    probe.run_probe(arguments(cache, root, "--checkpoint", str(model_path),
                              "--representations", "original,reconstructed,compressed",
                              "--budgets", "1,4", "--epochs", "1"))
    assert len(encoded) == before  # Prepared mmap vectors are actually reused.


@pytest.mark.parametrize("wrong_identity,budgets,message", [
    (None, "1,8", "exceeds checkpoint"), ("foreign-cache", "1,4", "identity mismatch")])
def test_invalid_checkpoint_or_budget_is_not_silently_clamped(tmp_path, wrong_identity, budgets, message):
    cache = labeled_cache(tmp_path / "cache")
    model_path = checkpoint(cache, tmp_path / "model.pt", identity=wrong_identity)
    with pytest.raises(ValueError, match=message):
        probe.run_probe(arguments(cache, tmp_path / "run", "--checkpoint", str(model_path),
                                  "--representations", "compressed", "--budgets", budgets))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("interruption", ["deadline", "hard_failure", "flush_failure"])
def test_feature_resume_preserves_durable_prefix_and_overwrites_partial_tail(tmp_path, monkeypatch, interruption):
    cache = labeled_cache(tmp_path / "cache", train_samples=10, variable_length=True)
    bundle = load_compressor(checkpoint(cache, tmp_path / "model.pt"))
    datasets = {split: FeatureCacheDataset(cache, split) for split in ("train", "validation")}
    indices = {split: list(range(len(dataset))) for split, dataset in datasets.items()}
    full, partial = tmp_path / "full", tmp_path / "partial"
    full.mkdir()
    partial.mkdir()
    monkeypatch.setattr(probe, "_FEATURE_BATCH_SAMPLES", 2)
    monkeypatch.setattr(probe, "_FEATURE_CHECKPOINT_ROWS", 4)
    monkeypatch.setattr(probe, "_FEATURE_CHECKPOINT_SECONDS", float("inf"))
    args = (datasets, indices, ["original", "reconstructed", "compressed"], [1, 4],
            "patch_mean", bundle, "cpu", {"regression": "feature-resume"}, None)
    try:
        expected = probe.prepare_features(full, *args)
        real_encode, real_decode, real_flush = bundle.encode_native, bundle.decode_native, np.memmap.flush
        encoded_batches = 0
        def encode(*values, **kwargs):
            nonlocal encoded_batches
            encoded_batches += 1
            return real_encode(*values, **kwargs)
        def decode(codes, *values, **kwargs):
            result = real_decode(codes, *values, **kwargs)
            if encoded_batches == 4 and codes.shape[1] == 4:
                # The fourth batch has original/direct/short-budget outputs,
                # but not a complete all-condition result.
                if interruption == "deadline":
                    raise probe.ProbeDeadline()
                raise RuntimeError("interrupted native decode")
            return result
        def flush(array, *values, **kwargs):
            if str(array.filename).endswith("/train-reconstructed-4.npy"):
                raise OSError("failed durable feature flush")
            return real_flush(array, *values, **kwargs)
        with monkeypatch.context() as patch:
            if interruption == "flush_failure":
                patch.setattr(np.memmap, "flush", flush)
                exception = OSError
            else:
                patch.setattr(bundle, "encode_native", encode)
                patch.setattr(bundle, "decode_native", decode)
                exception = probe.ProbeDeadline if interruption == "deadline" else RuntimeError
            with pytest.raises(exception):
                probe.prepare_features(partial, *args)
        manifest = json.loads((partial / "features_manifest.json").read_text())
        committed = {"deadline": 6, "hard_failure": 4, "flush_failure": 0}[interruption]
        assert manifest["rows_completed"] == {"train": committed, "validation": 0}
        assert manifest["completed"] is False
        for name in expected["arrays"]:
            actual = np.load(partial / "features" / f"{name}.npy", mmap_mode="r+")
            reference = np.load(full / "features" / f"{name}.npy")
            if name.startswith("train-"):
                np.testing.assert_array_equal(actual[:committed], reference[:committed])
                actual[committed:] = -12345  # Even a plausible on-disk tail is not authoritative.
            else:
                actual[:] = -12345
            actual.flush()
            actual._mmap.close()
        resumed = probe.prepare_features(partial, *args)
        assert resumed == expected
        for name in expected["arrays"]:
            np.testing.assert_array_equal(np.load(partial / "features" / f"{name}.npy"),
                                          np.load(full / "features" / f"{name}.npy"))
    finally:
        for dataset in datasets.values():
            dataset.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("pooling", ["patch_mean", "cls"])
def test_batched_native_features_match_explicit_unpadded_rows(tmp_path, monkeypatch, device, pooling):
    cache = labeled_cache(tmp_path / "cache", variable_length=True)
    bundle = load_compressor(checkpoint(cache, tmp_path / "model.pt"), map_location=device)
    datasets = {split: FeatureCacheDataset(cache, split) for split in ("train", "validation")}
    indices = {split: list(range(len(dataset))) for split, dataset in datasets.items()}
    root = tmp_path / "features-run"
    root.mkdir()
    # Force both variable-length padding and multiple token-bounded batches,
    # independently of the sample-count bound.
    monkeypatch.setattr(probe, "_FEATURE_TOKEN_BUDGET", 12)
    real_encode = bundle.encode_native
    batch_shapes = []
    def encode(native, *values, **kwargs):
        batch_shapes.append(tuple(native.shape[:2]))
        return real_encode(native, *values, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(bundle, "encode_native", encode)
            manifest = probe.prepare_features(root, datasets, indices,
                                              ["original", "reconstructed", "compressed"], [1, 4],
                                              pooling, bundle, device, {"regression": "batched-native"}, None)
        assert any(batch > 1 for batch, _ in batch_shapes)
        assert all(batch * tokens <= 12 for batch, tokens in batch_shapes)
        tolerance = {"rtol": 3e-4, "atol": 3e-5} if device == "cuda" else {"rtol": 1e-5, "atol": 1e-6}
        with torch.inference_mode():
            for split, dataset in datasets.items():
                expected = {name: [] for name in manifest["arrays"] if name.startswith(split + "-")}
                for row in indices[split]:
                    batch = probe.collate_features([dataset[row]])
                    native, positions, types, valid = (batch[key].to(device) for key in
                                                      ("features", "positions", "token_types", "valid_mask"))
                    codes = bundle.encode_native(native, positions, types, valid, num_codes=4)
                    expected[f"{split}-original"].append(
                        probe.pool_native(native, types, valid, pooling)[0].cpu().numpy())
                    expected[f"{split}-codes"].append(codes[0].cpu().numpy())
                    for budget in (1, 4):
                        reconstructed = bundle.decode_native(codes[:, :budget], positions, types, valid)
                        expected[f"{split}-reconstructed-{budget}"].append(
                            probe.pool_native(reconstructed, types, valid, pooling)[0].cpu().numpy())
                for name, rows in expected.items():
                    np.testing.assert_allclose(np.load(root / "features" / f"{name}.npy"),
                                               np.stack(rows), **tolerance)
    finally:
        for dataset in datasets.values():
            dataset.close()


def test_deadline_requires_timezone_and_elapsed_boundary_is_real():
    with pytest.raises(ValueError, match="timezone"):
        probe.parse_deadline("2026-10-15T08:00:00")
    assert probe.parse_deadline("2026-10-15T08:00:00+03:00") == probe.parse_deadline("2026-10-15T05:00:00Z")
    with pytest.raises(probe.ProbeDeadline):
        probe.check_deadline(probe.parse_deadline("2000-01-01T00:00:00Z"))
