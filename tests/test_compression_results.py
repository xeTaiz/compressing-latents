"""Pure-data result reading, live append semantics, and explicit provenance."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from compression.results import load_campaign, load_smokes, read_jsonl


def snapshot(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def manifest(tmp_path):
    path = tmp_path / 'campaign.json'
    snapshot(path, {'format': 'compression-campaign/v1', 'campaign_id': 'actual',
                    'deadline': '2026-10-15T08:00:00+03:00', 'scope': 'training',
                    'experiments': [{'id': 'dino256', 'model': 'dinov3', 'scope': 'training',
                                     'run_dir': 'runs/dino', 'eval_path': 'eval/dino.json',
                                     'probe_dir': 'probes/dino', 'original_probe_dir': 'probes/original'}]})
    return path


def test_incomplete_tail_only_is_ignored(tmp_path):
    path = tmp_path / 'log.jsonl'
    path.write_bytes(b'{"step":1}\nnot-json\n{"step":2}\n{"step":')
    rows, issues = read_jsonl(path)
    assert rows == [{'step': 1}, {'step': 2}]
    assert [(i['line'], i['severity']) for i in issues] == [(2, 'error'), (4, 'warning')]
    path.write_bytes(b'{"step":1}\n{"step":\n')
    rows, issues = read_jsonl(path)
    assert len(rows) == 1
    assert issues[0]['severity'] == 'error'


def test_valid_unterminated_record_and_nonobject(tmp_path):
    path = tmp_path / 'log.jsonl'
    path.write_bytes(b'[]\n{"step":3}')
    rows, issues = read_jsonl(path)
    assert rows == [{'step': 3}]
    assert issues[0]['line'] == 1
    assert issues[0]['severity'] == 'error'


def test_pending_campaign_has_no_invented_results(tmp_path):
    data = load_campaign(tmp_path / 'campaign.json')
    assert data['status'] == 'pending'
    assert data['experiments'] == []
    assert data['errors'] == []


def test_paths_state_curves_evaluation_and_probe_provenance(tmp_path):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {'format': 'compression-campaign-state/v1',
        'campaign_id': 'actual', 'experiments': {'dino256': {
        'status': 'deadline', 'reason': 'hard deadline', 'actual_completed_steps': {'fixed': 7},
        'stages': {'fixed': {'status': 'deadline', 'step': 7}}, 'errors': []}}})
    snapshot(tmp_path / 'runs/dino/fixed/config.json', {'max_steps': 10000,
             'cache': {'identity': 'train-only', 'train_samples': 10000},
             'config': {'model': {'max_codes': 256, 'width': 512}}})
    (tmp_path / 'runs/dino/fixed/train_log.jsonl').write_text('{"step":7,"loss":0.9}\n')
    snapshot(tmp_path / 'eval/dino.json', {'split': 'validation', 'heldout': True,
             'checkpoint_stage': 'prefix', 'metrics': [{'budget': 4, 'normalized_mse': {'mean': 0.5}}]})
    result = {'format': 'compression-linear-probe/v1', 'status': 'completed',
              'pooling': 'cls', 'compressed_pooling': 'ordered_prefix_flatten',
              'class_coverage': {'unseen_validation_labels': [99]},
              'metrics': [{'condition': 'compressed', 'budget': 4, 'feature_dim': 2048,
                           'classifier_parameters': 4098, 'top1': 0.2, 'top5': 0.5}]}
    snapshot(tmp_path / 'probes/dino/probe_results.json', result)
    data = load_campaign(path)
    exp = data['experiments'][0]
    assert exp['run_dir'] == str(tmp_path / 'runs/dino')
    assert exp['status'] == 'deadline'
    assert exp['state']['reason'] == 'hard deadline'
    assert exp['stages'][0]['actual_completed_steps'] == 7
    assert exp['stages'][0]['configured_steps'] == 10000
    assert exp['evaluation']['heldout'] is True
    assert exp['probes'][1]['results'] == result
    assert data['manifest']['experiments'][0]['scope'] == 'training'
    assert data['artifacts'] and data['latest_artifact_at']


def test_live_append_and_atomic_replacement_are_reread(tmp_path):
    path = manifest(tmp_path)
    log = tmp_path / 'runs/dino/fixed/train_log.jsonl'
    log.parent.mkdir(parents=True)
    log.write_text('{"step":1}\n{"step":')
    first = load_campaign(path)
    assert len(first['experiments'][0]['stages'][0]['curves']) == 1
    with log.open('a') as stream:
        stream.write('2}\n')
    replacement = tmp_path / 'replacement.json'
    snapshot(replacement, {'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
                          'experiments': {'dino256': {'status': 'failed', 'reason': 'actual error'}}})
    replacement.replace(tmp_path / 'state.json')
    second = load_campaign(path)
    assert len(second['experiments'][0]['stages'][0]['curves']) == 2
    assert second['experiments'][0]['status'] == 'failed'
    assert not second['warnings']


def test_committed_corruption_surfaces_as_error_state(tmp_path):
    path = manifest(tmp_path)
    stage = tmp_path / 'runs/dino/fixed'
    stage.mkdir(parents=True)
    (stage / 'train_log.jsonl').write_text('{"step":1}\nbroken\n')
    (stage / 'summary.json').write_text('{broken}')
    data = load_campaign(path)
    assert data['status'] == 'artifact_error'
    assert len(data['errors']) == 2
    assert data['experiments'][0]['stages'][0]['curves'] == [{'step': 1}]


def test_four_smokes_are_explicit_not_shape_quality_evidence(tmp_path):
    snapshot(tmp_path / 'dinov3/smoke-proof.json', {'production': [{'code_shape': [1, 256, 512]}]})
    data = load_smokes(tmp_path)
    assert [e['model'] for e in data['experiments']] == ['dinov3', 'qwen38', 'vggt', 'vjepa21']
    assert all(e['scope'] == 'smoke' for e in data['experiments'])
    assert data['quality_evidence'] is False
    assert data['experiments'][0]['status'] == 'completed_smoke'
    assert data['experiments'][1]['status'] == 'pending'


def test_state_checkpoint_versioned_artifacts_override_manifest_defaults(tmp_path):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'eval/dino.json', {'checkpoint_sha256': 'old', 'metrics': []})
    snapshot(tmp_path / 'versions/new/eval.json', {'checkpoint_sha256': 'new', 'metrics': []})
    snapshot(tmp_path / 'versions/new/probe/probe_results.json',
             {'status': 'deadline', 'checkpoint_sha256': 'new', 'metrics': []})
    snapshot(tmp_path / 'probes/original/probe_results.json',
             {'status': 'completed', 'metrics': [{'condition': 'original', 'budget': None}]})
    snapshot(tmp_path / 'state.json', {'format': 'compression-campaign-state/v1',
        'campaign_id': 'actual', 'experiments': {'dino256': {
        'actual_completed_steps': {'prefix': 19}, 'artifacts': {
            'eval_path': 'versions/new/eval.json', 'probe_dir': 'versions/new/probe',
            'model': 'versions/new/model.pt'}}}})
    experiment = load_campaign(path)['experiments'][0]
    assert experiment['evaluation']['checkpoint_sha256'] == 'new'
    assert experiment['probes'][1]['results']['checkpoint_sha256'] == 'new'
    assert experiment['probes'][0]['results']['metrics'][0]['condition'] == 'original'
    assert experiment['stages'][1]['actual_completed_steps'] == 19
    assert experiment['state']['artifacts']['model'] == 'versions/new/model.pt'


def test_live_optimizer_steps_advance_past_stale_state_and_summary(tmp_path):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': {
            'actual_completed_steps': {'fixed': 0, 'prefix': 23},
            'stages': {'fixed': {'status': 'running', 'step': 0}}}}})
    snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': 'completed', 'step': 0})
    log = tmp_path / 'runs/dino/fixed/train_log.jsonl'
    log.write_text('{"step":1000}\n{"step":1050}\n')
    first = load_campaign(path)['experiments'][0]['stages']
    assert first[0]['actual_completed_steps'] == 1050
    assert first[0]['status'] == 'running'
    assert first[1]['actual_completed_steps'] == 23
    with log.open('a') as stream:
        stream.write('{"step":1100}\n{"event":"checkpoint"}\n')
    second = load_campaign(path)['experiments'][0]['stages']
    assert second[0]['actual_completed_steps'] == 1100
    assert second[0]['status'] == 'running'
    assert second[1]['actual_completed_steps'] == 23


@pytest.mark.parametrize('stage_step,state_step,summary_step,expected', [
    (1073, 0, 0, 1073),
    (0, 1081, 0, 1081),
    (0, 0, 1087, 1087),
    (1090, 1100, 1110, 1110),
])
def test_unlogged_completed_steps_survive_older_optimizer_logs(
    tmp_path, stage_step, state_step, summary_step, expected
):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': {
            'actual_completed_steps': {'fixed': state_step},
            'stages': {'fixed': {'status': 'deadline', 'step': stage_step}}}}})
    snapshot(tmp_path / 'runs/dino/fixed/summary.json',
             {'status': 'completed', 'step': summary_step})
    (tmp_path / 'runs/dino/fixed/train_log.jsonl').write_text('{"step":1050}\n')
    stage = load_campaign(path)['experiments'][0]['stages'][0]
    assert stage['actual_completed_steps'] == expected
    assert stage['status'] == 'deadline'


@pytest.mark.parametrize('valid_step', [None, 0, 7.0])
def test_only_valid_completed_counts_are_progress_evidence(tmp_path, valid_step):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': {
            'actual_completed_steps': {'fixed': -1},
            'stages': {'fixed': {'step': True}}}}})
    snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'step': float('inf')})
    rows = [{'step': step} for step in
            [None, '9999', False, -2, 3.5, float('nan'), float('inf'), float('-inf')]]
    if valid_step is not None:
        rows.insert(0, {'step': valid_step})
    (tmp_path / 'runs/dino/fixed/train_log.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows))
    data = load_campaign(path)
    assert data['experiments'][0]['stages'][0]['actual_completed_steps'] == valid_step
    assert data['experiments'][0]['stages'][1]['actual_completed_steps'] is None
    assert not data['errors']


@pytest.mark.parametrize('status', ['running', 'deadline', 'failed', 'completed'])
@pytest.mark.parametrize('with_state', [False, True])
def test_explicit_summary_status_when_stage_state_has_no_status(tmp_path, status, with_state):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': status, 'step': 3})
    if with_state:
        snapshot(tmp_path / 'state.json', {
            'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
            'experiments': {'dino256': {'stages': {'fixed': {'step': 3}}}}})
    stage = load_campaign(path)['experiments'][0]['stages'][0]
    assert stage['status'] == status
    assert stage['actual_completed_steps'] == 3


def test_explicit_stage_state_status_precedes_summary_and_legacy_summary_fallback(tmp_path):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': 'completed', 'step': 3})
    snapshot(tmp_path / 'runs/dino/prefix/summary.json', {'step': 2})
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': {'stages': {'fixed': {'status': 'deadline', 'step': 4}}}}})
    stages = load_campaign(path)['experiments'][0]['stages']
    assert stages[0]['status'] == 'deadline'
    assert stages[0]['actual_completed_steps'] == 4
    assert stages[1]['status'] == 'completed'


@pytest.mark.parametrize('experiment_id,model,step', [
    ('dinov3-k256', 'dinov3', 1750),
    ('qwen38-k256', 'qwen38', 3150),
])
def test_current_child_promotes_observed_preflight_snapshot_only(
    tmp_path, experiment_id, model, step
):
    path = tmp_path / 'campaign.json'
    run = tmp_path / experiment_id
    queued_id = experiment_id.replace('k256', 'k512')
    snapshot(path, {
        'format': 'compression-campaign/v1', 'campaign_id': 'overnight-20261014',
        'experiments': [
            {'id': experiment_id, 'model': model, 'scope': 'pilot', 'run_dir': str(run)},
            {'id': queued_id, 'model': model, 'status': 'queued'}]})
    # The live campaign record publishes stage=fixed/status=running while
    # retaining inspect_stage's zero-step pending snapshot until child exit.
    record = {
        'id': experiment_id, 'status': 'running', 'stage': 'fixed', 'scope': 'pilot',
        'actual_completed_steps': {'fixed': 0, 'prefix': None},
        'stages': {'fixed': {
            'checkpoint': str(run / 'fixed/checkpoint_latest.pt'),
            'model': str(run / 'fixed/model.pt'), 'status': 'pending', 'step': 0,
            'summary': str(run / 'fixed/summary.json'), 'target_steps': 10000}},
        'artifacts': {'run_dir': str(run), 'eval_path': str(run / 'validation.json'),
                      'probe_dir': str(run / 'linear-probes'),
                      'original_probe_dir': str(tmp_path / 'baselines' / model)},
        'cache_identity': 'train-cache-identity', 'train_samples': 10000, 'errors': [],
        'updated_at': '2026-10-14T19:22:11.571569+00:00'}
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'overnight-20261014',
        'experiments': {experiment_id: record}})
    snapshot(run / 'fixed/config.json', {'max_steps': 10000})
    (run / 'fixed/train_log.jsonl').write_text(f'{{"step":{step}}}\n')
    active, queued = load_campaign(path)['experiments']
    assert active['status'] == 'running'
    assert active['stages'][0]['status'] == 'running'
    assert active['stages'][0]['actual_completed_steps'] == step
    assert active['stages'][0]['state'] == record['stages']['fixed']
    assert active['stages'][1]['status'] == 'pending'
    assert queued['status'] == 'queued'
    assert [stage['status'] for stage in queued['stages']] == ['pending', 'pending']


@pytest.mark.parametrize('experiment_status,current_stage,expected', [
    ('running', 'fixed', ['running', 'pending']),
    ('running', 'prefix', ['pending', 'running']),
    ('running', 'original_probe', ['pending', 'pending']),
    ('queued', 'fixed', ['pending', 'pending']),
    ('completed', 'fixed', ['pending', 'pending']),
    ('failed', 'fixed', ['pending', 'pending']),
    ('deadline', 'fixed', ['pending', 'pending']),
])
def test_only_running_current_training_stage_can_promote_pending(
    tmp_path, experiment_status, current_stage, expected
):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': {
            'status': experiment_status, 'stage': current_stage,
            'stages': {'fixed': {'status': 'pending'}, 'prefix': {'status': 'pending'}}}}})
    stages = load_campaign(path)['experiments'][0]['stages']
    assert [stage['status'] for stage in stages] == expected


@pytest.mark.parametrize('terminal_status', ['completed', 'failed', 'deadline'])
@pytest.mark.parametrize('source', ['state', 'summary'])
def test_current_running_marker_does_not_override_explicit_terminal_status(
    tmp_path, terminal_status, source
):
    path = manifest(tmp_path)
    record = {'status': 'running', 'stage': 'fixed'}
    if source == 'state':
        record['stages'] = {'fixed': {'status': terminal_status}}
        snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': 'running'})
    else:
        snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': terminal_status})
    snapshot(tmp_path / 'state.json', {
        'format': 'compression-campaign-state/v1', 'campaign_id': 'actual',
        'experiments': {'dino256': record}})
    stage = load_campaign(path)['experiments'][0]['stages'][0]
    assert stage['status'] == terminal_status


@pytest.mark.parametrize('state_format,campaign_id', [
    ('compression-campaign-state/v1', 'foreign'),
    ('compression-campaign-state/v2', 'actual'),
])
def test_foreign_current_child_marker_cannot_promote_pending_stage(
    tmp_path, state_format, campaign_id
):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'state.json', {
        'format': state_format, 'campaign_id': campaign_id,
        'experiments': {'dino256': {
            'status': 'running', 'stage': 'fixed',
            'actual_completed_steps': {'fixed': 999},
            'stages': {'fixed': {'status': 'pending', 'step': 999}}}}})
    data = load_campaign(path)
    experiment = data['experiments'][0]
    assert data['errors'][0]['code'] == 'campaign_state_identity'
    assert experiment['state'] == {}
    assert experiment['status'] == 'pending'
    assert [stage['status'] for stage in experiment['stages']] == ['pending', 'pending']
    assert experiment['stages'][0]['actual_completed_steps'] is None


@pytest.mark.parametrize('state_format,state_campaign_id', [
    ('compression-campaign-state/v1', 'foreign'),
    ('compression-campaign-state/v2', 'actual'),
    (None, 'actual'),
    ('compression-campaign-state/v1', None),
])
def test_reject_state_before_any_foreign_status_steps_or_artifact_overrides(
    tmp_path, state_format, state_campaign_id
):
    path = manifest(tmp_path)
    snapshot(tmp_path / 'runs/dino/fixed/summary.json', {'status': 'deadline', 'step': 3})
    snapshot(tmp_path / 'eval/dino.json', {'checkpoint_sha256': 'actual', 'metrics': []})
    snapshot(tmp_path / 'probes/dino/probe_results.json',
             {'status': 'running', 'checkpoint_sha256': 'actual', 'metrics': []})
    snapshot(tmp_path / 'foreign/fixed/summary.json', {'status': 'completed', 'step': 99})
    snapshot(tmp_path / 'foreign/eval.json', {'checkpoint_sha256': 'foreign', 'metrics': []})
    snapshot(tmp_path / 'foreign/probe/probe_results.json',
             {'status': 'completed', 'checkpoint_sha256': 'foreign', 'metrics': []})
    foreign_state = {'format': state_format, 'campaign_id': state_campaign_id,
                     'status': 'completed', 'experiments': {'dino256': {
                         'status': 'completed', 'actual_completed_steps': {'fixed': 99},
                         'stages': {'fixed': {'status': 'completed', 'step': 99}},
                         'artifacts': {'run_dir': 'foreign', 'eval_path': 'foreign/eval.json',
                                       'probe_dir': 'foreign/probe'}}}}
    snapshot(tmp_path / 'state.json', foreign_state)
    data = load_campaign(path)
    experiment = data['experiments'][0]
    assert data['status'] == 'artifact_error'
    assert data['state'] is None
    assert data['rejected_state'] == foreign_state
    assert data['errors'][0]['code'] == 'campaign_state_identity'
    assert experiment['state'] == {}
    assert experiment['status'] == 'pending'
    assert experiment['run_dir'] == str(tmp_path / 'runs/dino')
    assert experiment['stages'][0]['status'] == 'deadline'
    assert experiment['stages'][0]['actual_completed_steps'] == 3
    assert experiment['evaluation']['checkpoint_sha256'] == 'actual'
    assert experiment['probes'][1]['results']['checkpoint_sha256'] == 'actual'
    assert not any('/foreign/' in artifact['path'] for artifact in data['artifacts'])


def test_reader_does_not_import_torch_or_backbones():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, '-c',
        "import sys; import compression.results; assert 'torch' not in sys.modules; "
        "assert not any(k.startswith(('transformers', 'timm')) for k in sys.modules)"],
        cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
