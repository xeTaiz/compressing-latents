import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import os
    import sys
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    import marimo as mo
    import altair as alt
    project_root = Path(os.environ.get("COMPRESSION_PROJECT_ROOT", str(mo.notebook_location().parent))).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from compression.results import load_campaign, load_smokes

    return (
        alt,
        datetime,
        json,
        load_campaign,
        load_smokes,
        mo,
        os,
        project_root,
        timezone,
    )


@app.cell(hide_code=True)
def controls(mo, os, project_root):
    mo.md("# Live ordered-compression results\nTraining, reconstruction and frozen linear probes — actual artifacts only.")
    manifest_input = mo.ui.text(value=os.environ.get("COMPRESSION_CAMPAIGN_MANIFEST", str(project_root / "runs/overnight-20261014/campaign.json")), label="Campaign manifest (change without editing code)", full_width=True)
    smoke_input = mo.ui.text(value=os.environ.get("COMPRESSION_SMOKE_ROOT", str(project_root / "runs/gpu-smoke-20261014")), label="Explicit four-model GPU smoke root", full_width=True)
    refresh = mo.ui.refresh(options=["15s", "30s", "1m"], default_interval="15s", label="Auto / manual refresh")
    manual_refresh = mo.ui.run_button(label="Read artifacts now")
    mo.vstack([mo.md("# Live ordered-compression results"), manifest_input, smoke_input, mo.hstack([refresh, manual_refresh]), mo.md("**Smoke ≠ quality evidence.** The four existing width-64 / K16 lifecycle smokes use 2 train + 2 validation samples. Their K256 production-shaped checks are not trained checkpoints. No missing result is replaced by a zero.")])
    return manifest_input, manual_refresh, refresh, smoke_input


@app.cell(hide_code=True)
def read_artifacts(
    datetime,
    load_campaign,
    load_smokes,
    manifest_input,
    manual_refresh,
    refresh,
    smoke_input,
    timezone,
):
    refresh.value
    manual_refresh.value
    campaign_data = load_campaign(manifest_input.value)
    smoke_data = load_smokes(smoke_input.value)
    all_experiments = campaign_data["experiments"] + smoke_data["experiments"]
    read_at = datetime.now(timezone.utc)
    print(f"Read {len(campaign_data['experiments'])} campaign experiments and {len(smoke_data['experiments'])} explicit smokes at {read_at.isoformat()}; {sum(len(s['curves']) for e in all_experiments for s in e['stages'])} actual training log records.")
    return all_experiments, campaign_data, read_at, smoke_data


@app.cell(hide_code=True)
def filters(mo):
    run_filter = mo.ui.text(value="", label="Run IDs: comma-separated (empty = all)")
    model_filter = mo.ui.multiselect(options=["dinov3", "qwen38", "vggt", "vjepa21"], value=[], label="Models (empty = all)")
    stage_filter = mo.ui.multiselect(options=["fixed", "prefix"], value=["fixed", "prefix"], label="Training stages")
    scope_filter = mo.ui.multiselect(options=["training", "pilot", "smoke", "unspecified"], value=[], label="Scopes (empty = all)")
    mo.hstack([run_filter, model_filter, stage_filter, scope_filter])
    return model_filter, run_filter, scope_filter, stage_filter


@app.cell(hide_code=True)
def selection(
    all_experiments,
    model_filter,
    run_filter,
    scope_filter,
    stage_filter,
):
    selected_experiments = [e for e in all_experiments if (not run_filter.value or e["id"] in [v.strip() for v in run_filter.value.split(",")]) and (not model_filter.value or e["model"] in model_filter.value) and (not scope_filter.value or e["scope"] in scope_filter.value)]
    training_rows = [{**r, "run": e["id"], "model": e["model"], "scope": e["scope"], "stage": s["stage"], "series": e["id"] + " / " + s["stage"]} for e in selected_experiments for s in e["stages"] if s["stage"] in stage_filter.value for r in s["curves"]]
    reconstruction_rows = [{**r, "run": e["id"], "model": e["model"], "scope": e["scope"], "checkpoint_stage": e["evaluation"].get("checkpoint_stage"), "split": e["evaluation"].get("split"), "heldout": e["evaluation"].get("heldout")} for e in selected_experiments if e["evaluation"] for r in e["evaluation"].get("metrics", [])]
    probe_rows = [{**r, "run": e["id"], "model": e["model"], "scope": e["scope"], "pooling": p["results"].get("pooling"), "compressed_pooling": p["results"].get("compressed_pooling"), "probe_scope": "bounded_sample_smoke" if p["results"].get("protocol", {}).get("bounded_sample_smoke") else e["scope"], "status": p["status"], "series": e["id"] + " / " + r["condition"]} for e in selected_experiments for p in e["probes"] if p["results"] for r in p["results"].get("metrics", [])]
    probe_epoch_rows = [{**r, "run": e["id"], "scope": e["scope"], "series": e["id"] + " / " + r.get("condition", "unknown") + " / K=" + str(r.get("budget"))} for e in selected_experiments for p in e["probes"] for r in p["curves"]]
    return (
        probe_epoch_rows,
        probe_rows,
        reconstruction_rows,
        selected_experiments,
        training_rows,
    )


@app.cell(hide_code=True)
def status(
    campaign_data,
    datetime,
    json,
    mo,
    read_at,
    selected_experiments,
    smoke_data,
    stage_filter,
):
    _deadline = campaign_data.get("deadline") or "2026-10-15T08:00:00+03:00"
    try:
        _deadline_dt = datetime.fromisoformat(_deadline.replace("Z", "+00:00"))
        _remaining = (_deadline_dt - read_at).total_seconds() if _deadline_dt.tzinfo else None
    except (ValueError, TypeError):
        _remaining = None
    _latest = max(filter(None, [campaign_data.get("latest_artifact_at"), smoke_data.get("latest_artifact_at")]), default=None)
    _lag = round((read_at - datetime.fromisoformat(_latest)).total_seconds(), 1) if _latest else None
    _status_rows = [{"run": e["id"], "model": e["model"], "scope": e["scope"], "status": e["status"], "stage": s["stage"], "stage_status": s["status"], "actual_steps": s["actual_completed_steps"], "configured_steps": s["configured_steps"], "probe_status": ", ".join(p["status"] for p in e["probes"]), "reason": e["state"].get("reason"), "errors": json.dumps(e["state"].get("errors", []))} for e in selected_experiments for s in e["stages"] if s["stage"] in stage_filter.value]
    _issues = campaign_data["issues"] + smoke_data["issues"]
    mo.vstack([mo.md(f"## Status\n**Loaded:** {read_at.isoformat()} · **Artifact lag:** {_lag if _lag is not None else 'pending'} s · **Campaign:** {campaign_data['status']}\n\n**Hard deadline:** {_deadline} · **{'DEADLINE REACHED' if _remaining is not None and _remaining <= 0 else 'time remaining'}:** {round(_remaining / 3600, 2) if _remaining is not None else 'invalid'} hours\n\n{'Campaign manifest pending: only explicitly labelled GPU smoke artifacts are shown.' if campaign_data['manifest'] is None else 'Deadline-limited actual steps are not convergence claims.'}"), mo.ui.table(_status_rows, selection=None), mo.md("### Artifact warnings / failures"), mo.ui.table(_issues, selection=None) if _issues else mo.md("No malformed committed artifacts reported.")])
    return


@app.cell(hide_code=True)
def training(alt, mo, training_rows):
    def line_chart(rows, x, y, title, color="series", facet=None):
        chart = alt.Chart(alt.Data(values=rows)).mark_line(point=True).encode(x=alt.X(x, title=x.split(":")[0]), y=alt.Y(y, scale=alt.Scale(zero=False)), color=alt.Color(color + ":N"), tooltip=[alt.Tooltip(x), alt.Tooltip(y), alt.Tooltip(color + ":N")]).properties(title=title, width=550, height=220)
        return chart.facet(row=facet + ":N") if facet else chart
    _training_views = [mo.md("## Training curves — fixed and sampled-prefix remain separate")]
    if training_rows:
        _training_views.extend([mo.ui.altair_chart(line_chart(training_rows, "step:Q", "loss:Q", "Loss (actual logged objective)", facet="stage")), mo.ui.altair_chart(line_chart(training_rows, "step:Q", "lr:Q", "Learning rate", facet="stage")), mo.ui.altair_chart(line_chart(training_rows, "step:Q", "tokens_per_s:Q", "Throughput: tokens / second", facet="stage"))])
    else:
        _training_views.append(mo.md("Training pending — no selected log records."))
    mo.vstack(_training_views)
    return (line_chart,)


@app.cell(hide_code=True)
def reconstruction(line_chart, mo, reconstruction_rows):
    _recon_views = [mo.md("## Held-out reconstruction vs ordered code budget\nNormalized MSE / native cosine / native MSE are reconstruction metrics, not classification accuracy.")]
    if reconstruction_rows:
        for _metric in ("normalized_mse", "native_cosine", "native_mse"):
            _rows = [{"budget": r["budget"], "value": r[_metric]["mean"], "series": r["run"] + " / " + r["scope"]} for r in reconstruction_rows if isinstance(r.get(_metric), dict) and r[_metric].get("mean") is not None]
            if _rows:
                _recon_views.append(mo.ui.altair_chart(line_chart(_rows, "budget:Q", "value:Q", _metric)))
        _recon_views.append(mo.ui.table(reconstruction_rows, selection=None))
    else:
        _recon_views.append(mo.md("Reconstruction evaluation pending — no selected real evaluation metrics."))
    mo.vstack(_recon_views)
    return


@app.cell(hide_code=True)
def probes(alt, line_chart, mo, probe_epoch_rows, probe_rows):
    _probe_views = [mo.md("## Frozen linear probes: top-1 / top-5 vs budget\nNative pooling: DINO CLS (not registers), Qwen type-0 patch mean. Reconstructed uses the same native pooling. **Compressed sees flattened ordered code coordinates directly, never decoded features.** K × width changes head capacity; these are not capacity-matched comparisons. Original-feature baselines are horizontal dashed lines and have no code budget. Top-1 / top-5 are fractions [0,1], not percentages. Top-5 uses min(5, training class count). Bounded-sample probes are smoke-only, not quality evidence.")]
    if probe_rows:
        for _metric in ("top1", "top5"):
            _budget = [{**r, "value": r[_metric]} for r in probe_rows if r.get("budget") is not None and r.get(_metric) is not None]
            _original = [{**r, "value": r[_metric]} for r in probe_rows if r["condition"] == "original" and r.get(_metric) is not None]
            _layers = []
            if _budget:
                _layers.append(alt.Chart(alt.Data(values=_budget)).mark_line(point=True).encode(x=alt.X("budget:Q"), y=alt.Y("value:Q", scale=alt.Scale(zero=False)), color="series:N", tooltip=["run:N", "condition:N", "budget:Q", "value:Q", "feature_dim:Q", "classifier_parameters:Q"]))
            if _original:
                _layers.append(alt.Chart(alt.Data(values=_original)).mark_rule(strokeDash=[7, 4]).encode(y="value:Q", color="series:N", tooltip=["run:N", "condition:N", "value:Q"]))
            if _layers:
                _probe_views.append(mo.ui.altair_chart(alt.layer(*_layers).properties(title=_metric + " (original = dashed)", width=650, height=250)))
        _probe_views.extend([mo.md("### Actual probe metrics and classifier capacity"), mo.ui.table(probe_rows, selection=None)])
    else:
        _probe_views.append(mo.md("**Probes pending.** No completed probe epoch/metric artifact is available for the selected runs; no accuracy is inferred from reconstruction."))
    if probe_epoch_rows:
        _probe_views.append(mo.ui.altair_chart(line_chart(probe_epoch_rows, "epoch:Q", "val_top1:Q", "Probe validation top-1 by epoch")))
    mo.vstack(_probe_views)
    return


@app.cell(hide_code=True)
def provenance(campaign_data, json, mo, selected_experiments, smoke_data):
    _provenance = [{"id": e["id"], "model": e["model"], "scope": e["scope"], "manifest_entry": e["manifest_entry"], "state": e["state"], "stages": [{"stage": s["stage"], "config": s["config"], "summary": s["summary"]} for s in e["stages"]], "evaluation_provenance": {k: v for k, v in (e["evaluation"] or {}).items() if k != "metrics"}, "probe_provenance": [{"path": p["path"], "status": p["status"], "config": p["config"], "results": p["results"], "features_manifest": p["features_manifest"]} for p in e["probes"]], "smoke_proof": e["smoke_proof"]} for e in selected_experiments]
    mo.vstack([mo.md("## Split, class coverage, protocol and provenance\nOfficial train / validation membership is preserved. Any feature normalization is fit on training only. Unseen validation labels and known-label cross-entropy denominator are reported by probes. Read the explicit sampled class coverage: this selected source subset is not asserted to be class-balanced or all 1,000 classes. Empty metric lists mean pending, not zero accuracy."), mo.accordion({"Full artifact provenance / capacity / pooling / sample selection": mo.md("```json\n" + json.dumps(_provenance, indent=2) + "\n```"), "Read paths and modification times": mo.ui.table(campaign_data["artifacts"] + smoke_data["artifacts"], selection=None)})])
    return


if __name__ == "__main__":
    app.run()
