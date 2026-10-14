"""Read live compression artifacts without importing Torch or model backbones.

Relative manifest paths resolve against the manifest's parent. Every call rereads
files, including atomically replaced snapshots. Missing artifacts are pending;
invalid committed JSON/JSONL is an explicit error, never a fabricated metric.
Campaign state must match both its versioned format and the manifest campaign
identity before any status, step, or artifact-path override is accepted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """A committed artifact cannot be interpreted as its declared format."""


def read_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read append-only records, ignoring only an incomplete trailing record.

    A valid final record without a newline is readable. An invalid unterminated
    final record may still be being written; any invalid newline-terminated
    record (including the last one) is reported with its exact line number.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return rows, issues
    except OSError as exc:
        return rows, [{"path": str(path), "severity": "error", "message": str(exc)}]
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            if index == len(lines) and not line.endswith((b"\n", b"\r")):
                issues.append({"path": str(path), "line": index, "severity": "warning",
                               "message": "Incomplete final JSONL record ignored"})
                continue
            issues.append({"path": str(path), "line": index, "severity": "error",
                           "message": f"Malformed committed JSONL record: {exc}"})
            continue
        if not isinstance(row, dict):
            issues.append({"path": str(path), "line": index, "severity": "error",
                           "message": "JSONL record must be an object"})
            continue
        rows.append(row)
    return rows, issues


class _Reader:
    def __init__(self) -> None:
        self.issues: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []

    def track(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            self.issues.append({"path": str(path), "severity": "error", "message": str(exc)})
            return False
        self.artifacts.append({"path": str(path), "modified_at": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc).isoformat(), "size_bytes": stat.st_size})
        return True

    def json(self, path: Path) -> dict[str, Any] | None:
        if not self.track(path):
            return None
        try:
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                raise ArtifactError("JSON snapshot must be an object")
            return value
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.issues.append({"path": str(path), "severity": "error",
                                "message": f"Malformed JSON snapshot: {exc}"})
            return None

    def jsonl(self, path: Path) -> list[dict[str, Any]]:
        self.track(path)
        rows, issues = read_jsonl(path)
        self.issues.extend(issues)
        return rows


def _path(root: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else root / path


def _experiment(reader: _Reader, root: Path, entry: dict[str, Any],
                state: dict[str, Any]) -> dict[str, Any]:
    artifacts = state.get("artifacts", {})
    run = _path(root, artifacts.get("run_dir") or entry.get("run_dir"), root / str(entry["id"]))
    stages = []
    for name in ("fixed", "prefix"):
        folder = run / name
        config = reader.json(folder / "config.json")
        summary = reader.json(folder / "summary.json")
        log = reader.jsonl(folder / "train_log.jsonl")
        stage_state = state.get("stages", {}).get(name, {})
        actual_steps = stage_state.get("step")
        if actual_steps is None:
            actual_steps = state.get("actual_completed_steps", {}).get(name)
        if actual_steps is None:
            actual_steps = (summary or {}).get("step")
        if actual_steps is None and log:
            actual_steps = log[-1].get("step")
        stages.append({"stage": name, "path": str(folder), "config": config,
                       "summary": summary, "curves": log, "state": stage_state,
                       "actual_completed_steps": actual_steps,
                       "configured_steps": (config or {}).get("max_steps"),
                       "status": stage_state.get("status") or (summary or {}).get("status") or
                                 ("completed" if summary else "running" if log else "pending")})
    eval_path = _path(root, artifacts.get("eval_path") or entry.get("eval_path"),
                      run / "prefix" / "validation.json")
    evaluation = reader.json(eval_path)
    probes = []
    seen: set[Path] = set()
    for key, default in (("original_probe_dir", run / "original-probe"),
                         ("probe_dir", run / "probe")):
        folder = _path(root, artifacts.get(key) or entry.get(key), default)
        if folder in seen:
            continue
        seen.add(folder)
        result = reader.json(folder / "probe_results.json")
        config = reader.json(folder / "probe_config.json")
        curves = reader.jsonl(folder / "probe_log.jsonl")
        extraction = reader.json(folder / "features_manifest.json")
        probes.append({"kind": key, "path": str(folder), "results": result,
                       "config": config, "curves": curves, "features_manifest": extraction,
                       "status": (result or {}).get("status", "running" if curves else "pending")})
    return {"id": entry["id"], "model": entry.get("model"),
            "scope": entry.get("scope", "unspecified"),
            "status": state.get("status", entry.get("status", "pending")),
            "run_dir": str(run), "manifest_entry": entry, "state": state,
            "stages": stages, "evaluation": evaluation, "eval_path": str(eval_path),
            "probes": probes, "smoke_proof": reader.json(run / "smoke-proof.json")}


def _finish(reader: _Reader, output: dict[str, Any]) -> dict[str, Any]:
    output.update({"loaded_at": datetime.now(timezone.utc).isoformat(),
                   "artifacts": reader.artifacts, "issues": reader.issues,
                   "warnings": [i for i in reader.issues if i["severity"] == "warning"],
                   "errors": [i for i in reader.issues if i["severity"] == "error"]})
    output["latest_artifact_at"] = max((a["modified_at"] for a in reader.artifacts), default=None)
    if output["errors"]:
        output["status"] = "artifact_error"
    return output


def load_campaign(manifest_path: str | Path) -> dict[str, Any]:
    """Normalize the explicit campaign manifest/state and current live artifacts."""
    manifest_path = Path(manifest_path).expanduser().absolute()
    root = manifest_path.parent
    reader = _Reader()
    manifest = reader.json(manifest_path)
    state = reader.json(root / "state.json")
    output: dict[str, Any] = {"manifest_path": str(manifest_path), "manifest": manifest,
                              "state": state, "experiments": [], "status": "pending"}
    if manifest is None:
        return _finish(reader, output)
    if manifest.get("format") != "compression-campaign/v1":
        reader.issues.append({"path": str(manifest_path), "severity": "error",
                              "message": "Unsupported campaign manifest format"})
    campaign_id = manifest.get("campaign_id")
    if state is not None and (
        state.get("format") != "compression-campaign-state/v1"
        or not isinstance(campaign_id, str) or not campaign_id
        or state.get("campaign_id") != campaign_id
    ):
        reader.issues.append({"path": str(root / "state.json"), "severity": "error",
                              "code": "campaign_state_identity",
                              "message": "Campaign state format/identity mismatch; state ignored",
                              "expected_format": "compression-campaign-state/v1",
                              "actual_format": state.get("format"),
                              "expected_campaign_id": campaign_id,
                              "actual_campaign_id": state.get("campaign_id")})
        output["rejected_state"] = state
        state = None
        output["state"] = None
    entries = manifest.get("experiments", [])
    if not isinstance(entries, list):
        reader.issues.append({"path": str(manifest_path), "severity": "error",
                              "message": "Manifest experiments must be a list"})
        entries = []
    states = (state or {}).get("experiments", {})
    if not isinstance(states, dict):
        reader.issues.append({"path": str(root / "state.json"), "severity": "error",
                              "message": "Campaign state experiments must be an object"})
        states = {}
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            reader.issues.append({"path": str(manifest_path), "severity": "error",
                                  "message": "Each experiment needs a string id"})
            continue
        if entry["id"] in ids:
            reader.issues.append({"path": str(manifest_path), "severity": "error",
                                  "message": f"Duplicate experiment id: {entry['id']}"})
            continue
        ids.add(entry["id"])
        experiment_state = states.get(entry["id"], {})
        if not isinstance(experiment_state, dict):
            reader.issues.append({"path": str(root / "state.json"), "severity": "error",
                                  "message": f"Invalid state for experiment: {entry['id']}"})
            experiment_state = {}
        output["experiments"].append(_experiment(reader, root, entry, experiment_state))
    output.update({"campaign_id": manifest.get("campaign_id"), "deadline": manifest.get("deadline"),
                   "scope": manifest.get("scope"), "status": (state or {}).get("status", "active")})
    if output["deadline"]:
        try:
            deadline = datetime.fromisoformat(output["deadline"].replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                raise ValueError("deadline must include a timezone")
            output["deadline_reached"] = datetime.now(timezone.utc) >= deadline
        except (ValueError, TypeError) as exc:
            reader.issues.append({"path": str(manifest_path), "severity": "error",
                                  "message": f"Invalid campaign deadline: {exc}"})
    return _finish(reader, output)


def load_smokes(smoke_root: str | Path) -> dict[str, Any]:
    """Read only the four explicitly identified GPU lifecycle smokes, not quality runs."""
    root = Path(smoke_root).expanduser().absolute()
    reader = _Reader()
    experiments = []
    for model in ("dinov3", "qwen38", "vggt", "vjepa21"):
        experiments.append(_experiment(reader, root, {"id": f"smoke-{model}", "model": model,
                           "scope": "smoke", "status": "pending", "run_dir": str(root / model)}, {}))
        experiment = experiments[-1]
        if experiment["smoke_proof"]:
            experiment["status"] = "completed_smoke"
    return _finish(reader, {"status": "smoke", "experiments": experiments,
                           "scope": "smoke", "smoke_root": str(root),
                           "quality_evidence": False})
