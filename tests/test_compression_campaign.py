"""Campaign allocation, provenance, concurrent state and owned-child boundaries."""
from datetime import datetime, timedelta, timezone
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from compression.campaign import (
    MANIFEST_FORMAT, PROBE_DEFAULTS, STATE_FORMAT, aware_datetime, evaluation_path,
    load_manifest, requested_budgets, run_child, select_experiments, slot_boundaries,
    snapshot_model, update_state, validate_evaluation,
)


def manifest_payload():
    experiments = []
    for model, k in (("qwen38", 512), ("dinov3", 256), ("qwen38", 256), ("dinov3", 512)):
        name = f"{model}-k{k}"
        experiments.append({"id": name, "model": model, "max_codes": k, "attention_mode": "separate",
                            "scope": "training", "status": "pending", "cache_dir": f"cache/{model}",
                            "config_path": f"configs/{model}.json", "run_dir": f"runs/{name}",
                            "probe_dir": f"probes/{name}", "original_probe_dir": f"original/{model}",
                            "eval_path": f"runs/{name}/evaluation.json"})
    return {"format": MANIFEST_FORMAT, "campaign_id": "test-campaign",
            "created_at": "2026-10-14T20:00:00+03:00", "deadline": "2026-10-15T08:00:00+03:00",
            "scope": "training", "experiments": experiments}


def test_manifest_paths_priority_and_assigned_subset_are_explicit(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps(manifest_payload()))
    config, experiments = load_manifest(manifest)
    assert config["probe_protocol"] == PROBE_DEFAULTS
    assert experiments[0]["cache_dir"] == str(tmp_path / "cache/qwen38")
    selected = select_experiments(experiments, ["qwen38-k512", "qwen38-k256", "dinov3-k256"])
    assert [entry["id"] for entry in selected] == ["dinov3-k256", "qwen38-k256", "qwen38-k512"]
    with pytest.raises(ValueError, match="unknown assigned"):
        select_experiments(experiments, ["not-in-manifest"])
    with pytest.raises(ValueError, match="unique assigned"):
        select_experiments(experiments, ["dinov3-k256", "dinov3-k256"])
    assert requested_budgets(256) == [4, 32, 64, 128, 256]
    assert requested_budgets(512) == [4, 32, 64, 128, 256, 512]
    assert requested_budgets(1024) == [4, 32, 64, 128, 256, 512, 1024]
    with pytest.raises(ValueError, match="never clamp"):
        requested_budgets(128)


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data.update(deadline="2026-10-15T08:00:00"), "timezone"),
    (lambda data: data["experiments"][0].update(scope="full"), "scope"),
    (lambda data: data["experiments"][0].update(max_codes=128), "budget"),
    (lambda data: data["experiments"][2].update(original_probe_dir="another"), "single original"),
    (lambda data: data["experiments"][1].update(run_dir=data["experiments"][0]["run_dir"]), "share run_dir"),
])
def test_manifest_rejects_ambiguous_or_unsafe_assignments(tmp_path, mutation, match):
    payload = manifest_payload()
    mutation(payload)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_manifest(path)


def test_remaining_slot_fractions_carry_time_and_weight_larger_experiments():
    now = aware_datetime("2026-10-15T06:00:00+03:00")
    deadline = now + timedelta(seconds=1000)
    fixed, prefix, end = slot_boundaries(now, deadline, [1, 2])
    assert (fixed - now).total_seconds() == pytest.approx(1000 / 3 * 0.22, abs=1e-6)
    assert (prefix - now).total_seconds() == pytest.approx(1000 / 3 * 0.70, abs=1e-6)
    assert (end - now).total_seconds() == pytest.approx(1000 / 3, abs=1e-6)
    # The next experiment starts early after an already completed first run.
    early = now + timedelta(seconds=10)
    next_fixed, next_prefix, next_end = slot_boundaries(early, deadline, [2])
    assert next_end == deadline
    assert (next_fixed - early).total_seconds() == pytest.approx(990 * 0.22)
    assert (next_prefix - early).total_seconds() == pytest.approx(990 * 0.70)
    assert slot_boundaries(deadline + timedelta(seconds=1), deadline, [1]) == (deadline + timedelta(seconds=1),) * 3


def test_state_merges_disjoint_assignments_and_preserves_foreign_state(tmp_path):
    manifest = manifest_payload()
    update_state(tmp_path, manifest, {"id": "a", "status": "deadline", "actual_completed_steps": {"fixed": 17, "prefix": 0}})
    update_state(tmp_path, manifest, {"id": "b", "status": "failed", "reason": "test"})
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["format"] == STATE_FORMAT and set(state["experiments"]) == {"a", "b"}
    assert state["experiments"]["a"]["actual_completed_steps"]["fixed"] == 17
    before = (tmp_path / "state.json").read_bytes()
    with pytest.raises(ValueError, match="incompatible campaign"):
        update_state(tmp_path, {**manifest, "campaign_id": "foreign"}, {"id": "c"})
    assert (tmp_path / "state.json").read_bytes() == before


def test_checkpoint_snapshot_survives_atomic_replacement_and_reuses_inode(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"real artifact version one")
    snapshot, digest = snapshot_model(source, tmp_path)
    assert snapshot.stat().st_ino == source.stat().st_ino
    same, same_digest = snapshot_model(source, tmp_path)
    assert same == snapshot and same_digest == digest
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"real artifact version two")
    os.replace(replacement, source)
    newer, newer_digest = snapshot_model(source, tmp_path)
    assert newer_digest != digest and newer != snapshot
    assert snapshot.read_bytes() == b"real artifact version one"
    assert newer.read_bytes() == b"real artifact version two"


@pytest.mark.parametrize("error_number", [errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP])
def test_snapshot_falls_back_to_atomic_independent_copy(tmp_path, monkeypatch, error_number):
    from compression import campaign
    source = tmp_path / "model.pt"
    original = b"checkpoint bytes before future in-place modification"
    source.write_bytes(original)
    def unsupported(*args):
        raise OSError(error_number, "hardlinks unavailable")
    monkeypatch.setattr(campaign.os, "link", unsupported)
    snapshot, digest = snapshot_model(source, tmp_path)
    assert snapshot.read_bytes() == original
    assert campaign.sha256_file(snapshot) == digest
    assert snapshot.stat().st_ino != source.stat().st_ino
    source.write_bytes(b"changed mutable source")
    assert snapshot.read_bytes() == original
    assert not list(snapshot.parent.glob(".model-copy-*.tmp"))


@pytest.mark.parametrize("failure", ["corrupted_copy", "copy_error"])
def test_snapshot_copy_failure_checks_digest_and_removes_only_owned_temporary(tmp_path, monkeypatch, failure):
    from compression import campaign
    source = tmp_path / "model.pt"
    source.write_bytes(b"expected immutable checkpoint")
    directory = tmp_path / "campaign-checkpoints" / campaign.sha256_file(source)
    directory.mkdir(parents=True)
    foreign = directory / ".model-copy-not-ours.tmp"
    foreign.write_bytes(b"unrelated")
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "hardlinks unavailable")
    def broken_copy(input_file, output, **kwargs):
        output.write(b"corrupted")
        if failure == "copy_error":
            raise OSError(errno.ENOSPC, "no space")
    monkeypatch.setattr(campaign.os, "link", unsupported)
    monkeypatch.setattr(campaign.shutil, "copyfileobj", broken_copy)
    with pytest.raises(ValueError if failure == "corrupted_copy" else OSError):
        snapshot_model(source, tmp_path)
    assert not (directory / "model.pt").exists()
    assert list(directory.iterdir()) == [foreign]
    assert foreign.read_bytes() == b"unrelated"


def test_missing_evaluation_fingerprint_cannot_inherit_current_mutable_model_digest(tmp_path):
    checkpoint = tmp_path / "immutable/model.pt"
    source = tmp_path / "prefix/model.pt"
    config = {"feature_dim": 3, "model": {"width": 2, "max_codes": 256}}
    report = {"format": "ordered-compressor-eval/v1", "checkpoint": str(source),
              "cache_identity": "cache", "checkpoint_stage": "prefix", "checkpoint_step": 7,
              "model_config": {"input_dim": 3, **config["model"]}, "split": "validation",
              "heldout": True, "max_samples": None, "budgets": [4], "num_samples": 2,
              "metrics": [{"budget": 4, **{name: {"mean": 0.2, "std": 0.0, "count": 2}
                                         for name in ("normalized_mse", "native_cosine", "native_mse")}}]}
    arguments = dict(checkpoint=checkpoint, digest="current-weights", identity="cache", stage="prefix",
                     step=7, config=config, budgets=[4], source_model=source)
    with pytest.raises(ValueError, match="unverifiable"):
        validate_evaluation(report, **arguments)
    with pytest.raises(ValueError, match="unverifiable"):
        validate_evaluation({**report, "checkpoint_sha256": "previous-weights"}, **arguments)
    validate_evaluation({**report, "checkpoint_sha256": "current-weights"}, **arguments)


def test_evaluation_selection_preserves_unfingerprinted_base_and_versioned_reports(tmp_path):
    base = tmp_path / "evaluation.json"
    legacy = {"checkpoint_stage": "prefix", "checkpoint_step": 7, "checkpoint": "mutable/model.pt"}
    base.write_text(json.dumps(legacy))
    versioned = tmp_path / "checkpoint-current-weights/evaluation.json"
    versioned.parent.mkdir()
    versioned.write_text(json.dumps(legacy))
    base_before, versioned_before = base.read_bytes(), versioned.read_bytes()
    selected = evaluation_path(base, "current-weights", "prefix", 7)
    assert selected not in (base, versioned) and not selected.exists()
    assert base.read_bytes() == base_before and versioned.read_bytes() == versioned_before
    selected.write_text(json.dumps({**legacy, "checkpoint_sha256": "current-weights"}))
    assert evaluation_path(base, "current-weights", "prefix", 7) == selected


def test_manifest_deadline_is_aware_but_not_pinned_to_one_calendar_date(tmp_path):
    payload = manifest_payload()
    payload.update(created_at="2031-04-08T20:00:00+03:00", deadline="2031-04-09T08:00:00+03:00")
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps(payload))
    loaded, _ = load_manifest(manifest)
    assert aware_datetime(loaded["deadline"]) == aware_datetime("2031-04-09T05:00:00+00:00")


def test_expired_child_deadline_does_not_launch_a_process(tmp_path):
    marker = tmp_path / "must-not-exist"
    result = run_child([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                       hard_deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
                       log_path=tmp_path / "child.log")
    assert result["status"] == "deadline" and result["returncode"] is None
    assert not marker.exists() and not (tmp_path / "child.log").exists()


def test_child_hard_timeout_kills_only_owned_group_and_returns_real_failure(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        start = time.monotonic()
        result = run_child([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
                           hard_deadline=datetime.now(timezone.utc) + timedelta(seconds=0.5),
                           termination_seconds=0.1, log_path=tmp_path / "timeout.log")
        assert result["status"] == "deadline" and result["returncode"] < 0
        assert time.monotonic() - start < 3.0
        assert unrelated.poll() is None
        failed = run_child([sys.executable, "-c", "raise SystemExit(7)"],
                           hard_deadline=datetime.now(timezone.utc) + timedelta(seconds=5),
                           termination_seconds=0.1, log_path=tmp_path / "failed.log")
        assert failed["status"] == "failed" and failed["returncode"] == 7
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_real_cpu_campaign_completes_all_conditions_and_reuses_checkpoint_on_restart(tmp_path, monkeypatch):
    import numpy as np
    from compression import campaign
    from compression.cache import FeatureCacheWriter, cache_identity
    from compression.train import load_config
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    deadline = datetime.now(timezone.utc) + timedelta(seconds=180)
    records = [{"sample_id": f"{split}-{i}", "split": split, "layer": None}
               for split, count in (("train", 4), ("validation", 2)) for i in range(count)]
    provenance = {"model": "qwen38", "storage_dtype": "float16", "scope": "test"}
    cache = tmp_path / "cache" / cache_identity(provenance, records)
    writer = FeatureCacheWriter(cache, provenance, records, shard_tokens=16)
    rng = np.random.default_rng(7)
    for index, record in enumerate(records):
        writer.add(sample_id=record["sample_id"], split=record["split"],
                   features=rng.normal(size=(4, 4)).astype(np.float32),
                   positions=np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]),
                   token_types=np.zeros(4, dtype=np.int64), grid=(1, 2, 2),
                   metadata={"manifest_record": {"label": index % 2}})
    writer.finish()
    config = load_config(None, ["name=campaign-test", "feature_dim=4", "data.model=qwen38",
                               "data.token_budget=16", "data.num_workers=0", "model.width=8",
                               "model.max_codes=256", "model.encoder_layers=1", "model.decoder_layers=1",
                               "model.heads=1", "model.ff_multiplier=2", "model.position_frequencies=2",
                               "precision=fp32", "optim.warmup_steps=0", "stages.fixed.max_steps=1",
                               "stages.prefix.max_steps=1", "checkpoint_every=1", "log_every=1"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    payload = manifest_payload()
    entry = payload["experiments"][2]
    entry.update(cache_dir=str(cache), config_path=str(config_path), scope="pilot")
    payload.update(experiments=[entry], scope="pilot", created_at=datetime.now(timezone.utc).isoformat(),
                   deadline=deadline.isoformat(), cleanup_seconds=2, termination_seconds=0.2,
                   probe_protocol={"epochs": 1, "batch_size": 2, "lr": 0.1, "weight_decay": 0, "seed": 0})
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps(payload))
    first = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert first["status"] == "completed", first
    assert first["actual_completed_steps"] == {"fixed": 1, "prefix": 1}
    assert first["artifacts"]["model"] != str(tmp_path / entry["run_dir"] / "prefix/model.pt")
    evaluation = json.loads(Path(first["artifacts"]["eval_path"]).read_text())
    probes = json.loads((Path(first["artifacts"]["probe_dir"]) / "probe_results.json").read_text())
    original = json.loads((Path(first["artifacts"]["original_probe_dir"]) / "probe_results.json").read_text())
    assert evaluation["budgets"] == [4, 32, 64, 128, 256] and evaluation["heldout"]
    assert {(metric["condition"], metric["budget"]) for metric in probes["metrics"]} == {
        (condition, budget) for condition in ("reconstructed", "compressed") for budget in (4, 32, 64, 128, 256)}
    assert len(original["metrics"]) == 1 and original["metrics"][0]["condition"] == "original"
    training_model = tmp_path / entry["run_dir"] / "prefix/model.pt"
    model_stat = training_model.stat()
    second = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert second["status"] == "completed", second
    assert second["stages"]["fixed"]["reused"] and second["stages"]["prefix"]["reused"]
    assert second["evaluation"]["reused"]
    assert training_model.stat().st_mtime_ns == model_stat.st_mtime_ns
    assert second["checkpoint_sha256"] == first["checkpoint_sha256"]
    legacy_path = Path(second["artifacts"]["eval_path"])
    legacy_report = json.loads(legacy_path.read_text())
    legacy_report.pop("checkpoint_sha256")
    legacy_report["checkpoint"] = str(training_model)  # old evaluate CLI used a mutable export
    legacy_path.write_text(json.dumps(legacy_report))
    unverifiable_bytes = legacy_path.read_bytes()
    regenerated = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert regenerated["status"] == "completed", regenerated
    assert regenerated["artifacts"]["eval_path"] != str(legacy_path)
    assert legacy_path.read_bytes() == unverifiable_bytes
    verified_report = json.loads(Path(regenerated["artifacts"]["eval_path"]).read_text())
    assert verified_report["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert not regenerated["evaluation"].get("reused", False)
    old_probe_path = Path(first["artifacts"]["probe_dir"]) / "probe_results.json"
    old_probe_bytes = old_probe_path.read_bytes()
    config["stages"]["fixed"]["max_steps"] = 2
    config["stages"]["prefix"]["max_steps"] = 2
    config_path.write_text(json.dumps(config))
    extended = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert extended["status"] == "completed", extended
    assert extended["actual_completed_steps"] == {"fixed": 2, "prefix": 2}
    assert extended["checkpoint_sha256"] != first["checkpoint_sha256"]
    assert extended["artifacts"]["probe_dir"] != first["artifacts"]["probe_dir"]
    assert extended["original_probe"]["reused"]
    assert old_probe_path.read_bytes() == old_probe_bytes
    monkeypatch.setattr(campaign, "utcnow", lambda: deadline + timedelta(seconds=1))
    expired = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert expired["status"] == "completed", expired
    assert not expired.get("children")  # completion is proven offline; no post-deadline children
    import torch
    head_path = Path(extended["artifacts"]["probe_dir"]) / "heads/compressed-256.pt"
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    state["epoch"] = 0  # a filename and completed JSON cannot prove actual classifier training
    torch.save(state, head_path)
    corrupt_bytes = head_path.read_bytes()
    corrupt = campaign.run_campaign(manifest, [entry["id"]], device="cpu")["experiments"][entry["id"]]
    assert corrupt["status"] == "failed" and "committed classifier" in corrupt["reason"]
    assert head_path.read_bytes() == corrupt_bytes
