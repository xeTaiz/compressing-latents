"""Behavioral tests of the feature cache format (writer, resume, reader, collation)."""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from compression.cache import (CacheError, CacheIdentityError, CacheIntegrityError, DuplicateRecordError,
                               FeatureCacheDataset,
                               FeatureCacheWriter, IncompleteCacheError, cache_identity, collate_features,
                               verify_cache)

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = {'model': 'tiny-backbone', 'checkpoint_sha256': '0' * 64, 'source_pin': 'abc123',
              'feature_boundary': 'final-norm', 'compute_dtype': 'float32', 'storage_dtype': 'float16',
              'manifest_sha256': '1' * 64, 'transform': {'resize': 32}}


def make_sample(seed, *, views=1, rows=2, cols=3, dim=8, registers=2, cls=True, camera=False):
    """Native token layout: optional CLS, then per view camera, registers and row-major patches."""
    positions, types = [], []
    if cls:
        positions.append((0, 0, 0))
        types.append(1)
    for view in range(views):
        if camera:
            positions.append((view, 0, 0))
            types.append(3)
        for ordinal in range(registers):
            positions.append((view, 0, ordinal))
            types.append(2)
        for row in range(rows):
            for col in range(cols):
                positions.append((view, row, col))
                types.append(0)
    features = np.random.default_rng(seed).standard_normal((len(types), dim)).astype(np.float32)
    return {'features': features, 'positions': np.asarray(positions, dtype=np.int64),
            'token_types': np.asarray(types, dtype=np.int64), 'grid': (views, rows, cols)}


def records(train=('a', 'b', 'c'), validation=('v1',), layers=(None,)):
    return [{'sample_id': sid, 'split': split, 'layer': layer}
            for split, ids in (('train', train), ('validation', validation)) for sid in ids for layer in layers]


SAMPLES = {'a': make_sample(1), 'b': make_sample(2, rows=3, cols=4), 'c': make_sample(3, rows=1, cols=1, registers=0),
           'v1': make_sample(4, views=2, cls=False, camera=True)}


def write_cache(path, sample_ids=('a', 'b', 'c', 'v1'), **writer_kwargs):
    writer = FeatureCacheWriter(path, PROVENANCE, records(), **writer_kwargs)
    for sid in sample_ids:
        writer.add(sample_id=sid, split='val' if sid == 'v1' else 'train', metadata={'source': f'{sid}.jpg'},
                   **SAMPLES[sid])
    return writer


def expected_positions(sample):
    return sample['positions'].astype(np.float32) / np.maximum(np.asarray(sample['grid']) - 1, 1)


def test_roundtrip_returns_float16_rounded_features_normalized_positions_and_split_rows(tmp_path):
    path = write_cache(tmp_path / 'cache', shard_tokens=20).finish()
    train = FeatureCacheDataset(path, 'train')
    assert train.identity == cache_identity(PROVENANCE, records())
    assert train.layer is None and train.split == 'train' and train.feature_dim == 8
    assert train.sample_ids == ['a', 'b', 'c']
    assert train.token_counts == [len(SAMPLES[sid]['token_types']) for sid in ('a', 'b', 'c')]
    assert train.metadata[1] == {'source': 'b.jpg'}
    for index, sid in enumerate(train.sample_ids):
        item, sample = train[index], SAMPLES[sid]
        assert item['sample_id'] == sid
        assert item['features'].dtype == torch.float32 and item['token_types'].dtype == torch.int64
        np.testing.assert_array_equal(item['features'].numpy(), sample['features'].astype(np.float16).astype(np.float32))
        np.testing.assert_allclose(item['positions'].numpy(), expected_positions(sample))
        np.testing.assert_array_equal(item['token_types'].numpy(), sample['token_types'])
        assert item['grid'].tolist() == list(sample['grid'])
    validation = FeatureCacheDataset(path, 'val')
    assert validation.split == 'validation' and validation.sample_ids == ['v1']
    camera = validation[0]
    assert camera['positions'][camera['token_types'] == 3].tolist() == [[0, 0, 0], [1, 0, 0]]
    summary = verify_cache(path, full=True)
    assert summary['records'] == 4 and summary['records_by_split_layer'] == {'train': {'all': 3}, 'validation': {'all': 1}}
    for shard in path.rglob('*.features.npy'):
        assert np.load(shard, mmap_mode='r').shape[0] <= 20


def test_finish_and_reader_refuse_partial_cache_until_every_record_is_committed(tmp_path):
    writer = write_cache(tmp_path / 'cache', sample_ids=('a', 'b', 'v1'))
    with pytest.raises(IncompleteCacheError, match='1 of 4'):
        writer.finish()
    for split in ('train', 'validation'):
        with pytest.raises(IncompleteCacheError):
            FeatureCacheDataset(tmp_path / 'cache', split)
    writer.add(sample_id='c', split='train', **SAMPLES['c'])
    writer.finish()
    assert FeatureCacheDataset(tmp_path / 'cache', 'train').sample_ids == ['a', 'b', 'c']


CRASHING_WRITER = '''
import json, os, sys
import numpy as np
from compression.cache import FeatureCacheWriter
output, provenance, expected, data = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), np.load(sys.argv[4])
def sample(name):
    return {key: data[name + '__' + key] for key in ('features', 'positions', 'token_types')} | {
        'grid': tuple(int(v) for v in data[name + '__grid'])}
writer = FeatureCacheWriter(output, provenance, expected, shard_tokens=10_000)
writer.add(sample_id='a', split='train', **sample('a'))
writer.flush()
writer.add(sample_id='b', split='train', **sample('b'))
replace = os.replace
def killed_before_commit(source, target):
    if str(target).endswith('.json') and '/shards/' in str(target):
        os._exit(17)
    replace(source, target)
os.replace = killed_before_commit
writer.flush()
'''


def test_resume_after_hard_kill_recommits_only_lost_records_without_duplicates(tmp_path):
    path, arrays = tmp_path / 'cache', tmp_path / 'samples.npz'
    np.savez(arrays, **{f'{sid}__{key}': np.asarray(value) for sid, sample in SAMPLES.items()
                        for key, value in sample.items()})
    crashed = subprocess.run([sys.executable, '-c', CRASHING_WRITER, str(path), json.dumps(PROVENANCE),
                              json.dumps(records()), str(arrays)], cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))})
    assert crashed.returncode == 17, crashed.stderr
    assert any('.tmp-' in item.name for item in path.rglob('*')), 'the kill must leave a torn shard behind'
    assert len(list(path.rglob('*.features.npy'))) == 2
    with pytest.raises(IncompleteCacheError):
        FeatureCacheDataset(path, 'train')

    writer = FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
    assert writer.contains('a') and not writer.contains('b')
    assert [record['sample_id'] for record in writer.missing()] == ['b', 'c', 'v1']
    assert not any('.tmp-' in item.name for item in path.rglob('*'))
    assert len(list(path.rglob('*.features.npy'))) == 1
    with pytest.raises(DuplicateRecordError):
        writer.add(sample_id='a', split='train', **SAMPLES['a'])
    for sid in ('b', 'c', 'v1'):
        writer.add(sample_id=sid, split='validation' if sid == 'v1' else 'train', **SAMPLES[sid])
    with pytest.raises(DuplicateRecordError):
        writer.add(sample_id='b', split='train', **SAMPLES['b'])
    writer.finish()
    train = FeatureCacheDataset(path, 'train')
    assert train.sample_ids == ['a', 'b', 'c']
    np.testing.assert_array_equal(train[1]['features'].numpy(), SAMPLES['b']['features'].astype(np.float16))
    assert verify_cache(path, full=True)['records'] == 4


def test_resume_of_completed_cache_is_idempotent(tmp_path):
    path = write_cache(tmp_path / 'cache').finish()
    writer = FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
    assert all(writer.contains(sid) for sid in SAMPLES) and writer.missing() == []
    with pytest.raises(CacheError, match='complete'):
        writer.add(sample_id='a', split='train', **SAMPLES['a'])
    assert writer.finish() == path
    assert len(FeatureCacheDataset(path, 'train')) == 3


def test_writer_rejects_existing_mismatched_or_foreign_directories(tmp_path):
    path = tmp_path / 'cache'
    write_cache(path, sample_ids=('a',)).flush()
    with pytest.raises(FileExistsError):
        FeatureCacheWriter(path, PROVENANCE, records())
    with pytest.raises(CacheIdentityError):
        FeatureCacheWriter(path, {**PROVENANCE, 'checkpoint_sha256': 'f' * 64}, records(), resume=True)
    with pytest.raises(CacheIdentityError):
        FeatureCacheWriter(path, PROVENANCE, records(train=('a', 'b', 'c', 'd')), resume=True)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'notes.txt').write_text('keep me')
    with pytest.raises(FileExistsError):
        FeatureCacheWriter(foreign, PROVENANCE, records(), resume=True)
    assert sorted(item.name for item in foreign.iterdir()) == ['notes.txt']
    assert (foreign / 'notes.txt').read_text() == 'keep me'


def test_corrupted_committed_shard_is_rejected_and_never_deleted(tmp_path):
    path = write_cache(tmp_path / 'cache').finish()
    shard = sorted(path.rglob('*.features.npy'))[0]
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(CacheIntegrityError):
        FeatureCacheDataset(path, 'train' if '/train/' in shard.as_posix() else 'validation')
    with pytest.raises(CacheIntegrityError):
        verify_cache(path)
    with pytest.raises(CacheIntegrityError):
        FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
    assert shard.exists()


def test_reader_rejects_tampered_or_duplicated_index(tmp_path):
    path = write_cache(tmp_path / 'cache').finish()
    index = (path / 'index.jsonl').read_bytes()
    (path / 'index.jsonl').write_bytes(index + index.splitlines(keepends=True)[0])
    with pytest.raises(CacheIntegrityError, match='hash'):
        FeatureCacheDataset(path, 'train')
    complete = json.loads((path / 'COMPLETE.json').read_text())
    complete['index_sha256'] = hashlib.sha256((path / 'index.jsonl').read_bytes()).hexdigest()
    (path / 'COMPLETE.json').write_text(json.dumps(complete))
    with pytest.raises(CacheIntegrityError, match='twice'):
        FeatureCacheDataset(path, 'train')


def test_add_rejects_non_finite_values_and_float16_overflow(tmp_path):
    writer = FeatureCacheWriter(tmp_path / 'cache', PROVENANCE, records())
    for bad in (np.nan, np.inf, 1e6):
        sample = {**SAMPLES['a'], 'features': SAMPLES['a']['features'].copy()}
        sample['features'][0, 0] = bad
        with pytest.raises(ValueError):
            writer.add(sample_id='a', split='train', **sample)
    assert {record['sample_id'] for record in writer.missing()} == set(SAMPLES)
    writer.add(sample_id='a', split='train', **SAMPLES['a'])


def mutate(name, **changes):
    sample = {key: np.array(value) if isinstance(value, np.ndarray) else value for key, value in SAMPLES[name].items()}
    return {**sample, **changes}


def bad_layouts():
    a = SAMPLES['a']
    outside = a['positions'].copy()
    outside[-1] = (0, 2, 2)  # row 2 does not exist in a 2x3 grid
    cls_moved = a['positions'].copy()
    cls_moved[0] = (0, 1, 0)
    duplicate = a['positions'].copy()
    duplicate[-1] = duplicate[-2]
    unknown_type = a['token_types'].copy()
    unknown_type[0] = 5
    camera = SAMPLES['v1']['positions'].copy()
    camera[0] = (0, 0, 1)
    return {
        'patch outside grid': mutate('a', positions=outside),
        'CLS not at origin': mutate('a', positions=cls_moved),
        'duplicate token position': mutate('a', positions=duplicate),
        'missing patch': mutate('a', features=a['features'][:-1], positions=a['positions'][:-1],
                                token_types=a['token_types'][:-1]),
        'float positions': mutate('a', positions=a['positions'].astype(np.float32)),
        'unknown token type': mutate('a', token_types=unknown_type),
        'wrong grid rank': mutate('a', grid=(2, 3)),
        'camera with ordinal': mutate('v1', positions=camera),
    }


@pytest.mark.parametrize('case', sorted(bad_layouts()))
def test_add_rejects_invalid_token_layouts(tmp_path, case):
    writer = FeatureCacheWriter(tmp_path / 'cache', PROVENANCE, records())
    sid = 'v1' if case == 'camera with ordinal' else 'a'
    with pytest.raises(ValueError):
        writer.add(sample_id=sid, split='validation' if sid == 'v1' else 'train', **bad_layouts()[case])
    writer.add(sample_id=sid, split='validation' if sid == 'v1' else 'train', **SAMPLES[sid])
    writer.flush()
    assert writer.contains(sid)


def test_add_rejects_unexpected_records_wrong_split_and_feature_dim_changes(tmp_path):
    writer = FeatureCacheWriter(tmp_path / 'cache', PROVENANCE, records())
    with pytest.raises(KeyError):
        writer.add(sample_id='unknown', split='train', **SAMPLES['a'])
    with pytest.raises(KeyError):
        writer.add(sample_id='a', split='train', layer=23, **SAMPLES['a'])
    with pytest.raises(ValueError):
        writer.add(sample_id='a', split='validation', **SAMPLES['a'])
    with pytest.raises(ValueError):
        writer.add(sample_id='a', split='holdout', **SAMPLES['a'])
    writer.add(sample_id='a', split='train', **SAMPLES['a'])
    with pytest.raises(ValueError, match='dim'):
        writer.add(sample_id='b', split='train', **make_sample(2, rows=3, cols=4, dim=9))


def test_reader_rejects_invalid_or_absent_split(tmp_path):
    path = write_cache(tmp_path / 'cache').finish()
    with pytest.raises(ValueError):
        FeatureCacheDataset(path, 'holdout')
    with pytest.raises(ValueError):
        FeatureCacheDataset(path, 'test')


def test_identity_is_canonical_and_selection_sensitive():
    base = cache_identity(PROVENANCE, records())
    aliased = [{**record, 'split': 'val'} if record['split'] == 'validation' else record for record in records()]
    assert cache_identity(dict(reversed(list(PROVENANCE.items()))), list(reversed(aliased))) == base
    assert cache_identity({**PROVENANCE, 'transform': {'resize': 64}}, records()) != base
    assert cache_identity(PROVENANCE, records(train=('a', 'b'))) != base
    assert cache_identity(PROVENANCE, records(layers=(23,))) != base
    invalid = [records() + [records()[0]],
               records() + [{'sample_id': 'a', 'split': 'validation', 'layer': 4}],
               records(layers=(None,)) + records(train=('x',), validation=(), layers=(4,)),
               [{**records()[0], 'path': 'a.jpg'}],
               [{'sample_id': 'a', 'split': 'holdout'}]]
    for expected in invalid:
        with pytest.raises(ValueError):
            cache_identity(PROVENANCE, expected)
    with pytest.raises(ValueError):
        cache_identity({**PROVENANCE, 'storage_dtype': 'float32'}, records())
    with pytest.raises(ValueError):
        cache_identity({**PROVENANCE, 'scale': float('nan')}, records())


def multi_layer_cache(path, layers):
    writer = FeatureCacheWriter(path, PROVENANCE, records(train=('g1', 'g2'), validation=('g3',), layers=layers))
    for index, sid in enumerate(('g1', 'g2', 'g3')):
        for layer in layers:
            writer.add(sample_id=sid, split='validation' if sid == 'g3' else 'train', layer=layer,
                       **make_sample(100 * index + layer, views=4, camera=True, cls=False, registers=4, dim=6))
    return writer


def test_layer_selection_defaults_to_23_and_never_mixes_layers(tmp_path):
    path = multi_layer_cache(tmp_path / 'omega', (4, 11, 17, 23)).finish()
    default = FeatureCacheDataset(path, 'train')
    assert default.layer == 23 and len(default) == 2
    np.testing.assert_array_equal(default[0]['features'].numpy(),
                                  make_sample(23, views=4, camera=True, cls=False, registers=4, dim=6)['features']
                                  .astype(np.float16))
    layer4 = FeatureCacheDataset(path, 'train', layer=4)
    assert layer4.layer == 4 and layer4.sample_ids == ['g1', 'g2']
    np.testing.assert_array_equal(layer4[1]['features'].numpy(),
                                  make_sample(104, views=4, camera=True, cls=False, registers=4, dim=6)['features']
                                  .astype(np.float16))
    with pytest.raises(ValueError):
        FeatureCacheDataset(path, 'train', layer=7)
    ambiguous = multi_layer_cache(tmp_path / 'ambiguous', (4, 11)).finish()
    with pytest.raises(ValueError, match='layer'):
        FeatureCacheDataset(ambiguous, 'train')


def test_resume_tracks_committed_layers_of_one_group_independently(tmp_path):
    expected = records(train=('g1',), validation=(), layers=(4, 23))
    writer = FeatureCacheWriter(tmp_path / 'cache', PROVENANCE, expected)
    writer.add(sample_id='g1', split='train', layer=4, **make_sample(5, views=4, camera=True, cls=False))
    writer.flush()
    resumed = FeatureCacheWriter(tmp_path / 'cache', PROVENANCE, expected, resume=True)
    assert resumed.contains('g1', 4) and not resumed.contains('g1', 23)
    assert resumed.missing() == [{'sample_id': 'g1', 'split': 'train', 'layer': 23}]


def test_collate_pads_with_mask_and_zero_padding(tmp_path):
    dataset = FeatureCacheDataset(write_cache(tmp_path / 'cache').finish(), 'train')
    items = [dataset[index] for index in range(len(dataset))]
    batch = collate_features(items)
    longest = max(dataset.token_counts)
    assert batch['features'].shape == (3, longest, 8) and batch['positions'].shape == (3, longest, 3)
    assert batch['sample_ids'] == ['a', 'b', 'c'] and batch['grid'].tolist() == [[1, 2, 3], [1, 3, 4], [1, 1, 1]]
    assert batch['valid_mask'].sum(1).tolist() == dataset.token_counts
    for row, (item, count) in enumerate(zip(items, dataset.token_counts)):
        assert torch.equal(batch['features'][row, :count], item['features'])
        assert torch.equal(batch['token_types'][row, :count], item['token_types'])
        assert not batch['valid_mask'][row, count:].any()
        assert not batch['features'][row, count:].any() and not batch['positions'][row, count:].any()
    with pytest.raises(ValueError):
        collate_features([items[0], {**items[1], 'features': torch.zeros(items[1]['features'].shape[0], 9)}])


def test_dataset_pickles_without_mapped_feature_data(tmp_path):
    path = tmp_path / 'cache'
    big = make_sample(9, rows=16, cols=16, dim=256)
    writer = FeatureCacheWriter(path, PROVENANCE, records(train=('big',), validation=()))
    writer.add(sample_id='big', split='train', **big)
    dataset = FeatureCacheDataset(writer.finish(), 'train')
    first = dataset[0]['features']
    blob = pickle.dumps(dataset)
    assert len(blob) < big['features'].astype(np.float16).nbytes // 4
    assert torch.equal(pickle.loads(blob)[0]['features'], first)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_features)
    assert next(iter(loader))['features'].shape == (1, 16 * 16 + 3, 256)


@pytest.mark.parametrize('kind', ('features', 'positions', 'types'))
@pytest.mark.parametrize('complete', (False, True))
def test_resume_rejects_same_size_committed_payload_mutation(tmp_path, kind, complete):
    path = tmp_path / 'cache'
    writer = write_cache(path, sample_ids=('a',))
    if complete:
        for sid in ('b', 'c', 'v1'):
            writer.add(sample_id=sid, split='validation' if sid == 'v1' else 'train', **SAMPLES[sid])
        writer.finish()
    else:
        writer.flush()
    shard = sorted(path.rglob(f'*.{kind}.npy'))[0]
    size = shard.stat().st_size
    array = np.load(shard, mmap_mode='r+')
    array.flat[0] = np.nan if kind == 'features' else int(array.flat[0]) + 1
    array.flush()
    array._mmap.close()
    assert shard.stat().st_size == size
    with pytest.raises(CacheIntegrityError, match='sha256'):
        FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
    assert shard.exists() and shard.stat().st_size == size


@pytest.mark.parametrize('location', ('root', 'shard'))
def test_resume_preserves_unrelated_temporary_looking_files(tmp_path, location):
    path = tmp_path / 'cache'
    write_cache(path, sample_ids=('a',)).flush()
    directory = path if location == 'root' else next(path.rglob('*.features.npy')).parent
    foreign = directory / '.notes.tmp-deadbeef'
    foreign.write_text('unrelated user content')
    if location == 'root':
        resumed = FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
        assert resumed.contains('a')
    else:
        with pytest.raises(CacheIntegrityError, match='Unexpected file'):
            FeatureCacheWriter(path, PROVENANCE, records(), resume=True)
    assert foreign.read_text() == 'unrelated user content'


BOUNDED_READER = '''
import os, pickle, resource, sys
import numpy as np
from compression.cache import FeatureCacheDataset
dataset = FeatureCacheDataset(sys.argv[1], 'train')
baseline = len(os.listdir('/proc/self/fd'))
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = baseline + 48
if hard != resource.RLIM_INFINITY:
    limit = min(limit, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
first = dataset[0]
for index in list(range(len(dataset))) + list(reversed(range(len(dataset)))):
    item = dataset[index]
    assert item['sample_id'] == f'{index:04d}'
    np.testing.assert_array_equal(item['features'].numpy(), np.full((1, 2), index, dtype=np.float32))
    assert len(os.listdir('/proc/self/fd')) <= baseline + 30
np.testing.assert_array_equal(first['features'].numpy(), np.zeros((1, 2), dtype=np.float32))
restored = pickle.loads(pickle.dumps(dataset))
np.testing.assert_array_equal(restored[-1]['features'].numpy(), np.full((1, 2), 99, dtype=np.float32))
restored.close()
dataset.close()
assert len(os.listdir('/proc/self/fd')) <= baseline
np.testing.assert_array_equal(dataset[0]['features'].numpy(), np.zeros((1, 2), dtype=np.float32))
dataset.close()
'''


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='requires Linux descriptor limits')
def test_reader_visits_many_shards_under_bounded_file_descriptor_limit(tmp_path):
    path = tmp_path / 'cache'
    expected = records(train=tuple(f'{index:04d}' for index in range(100)), validation=())
    writer = FeatureCacheWriter(path, PROVENANCE, expected, shard_tokens=1)
    for index in range(100):
        writer.add(sample_id=f'{index:04d}', split='train', features=np.full((1, 2), index, dtype=np.float32),
                   positions=np.zeros((1, 3), dtype=np.int64), token_types=np.zeros(1, dtype=np.int64),
                   grid=(1, 1, 1))
    writer.finish()
    result = subprocess.run(
        [sys.executable, '-c', BOUNDED_READER, str(path)], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT), os.environ.get('PYTHONPATH')]))})
    assert result.returncode == 0, result.stderr
