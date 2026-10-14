"""Offline frozen-feature cache: bounded float16 NumPy shards with a relative JSON index.

One cache directory (normally ``<shared-root>/features/cache/<model>/<identity>``) holds::

    cache.json                                   immutable header: identity, provenance, expected records
    shards/<split>/<all|layerN>/NNNNNN-xxxxxxxx.features.npy   float16 [tokens, D]
    shards/<split>/<all|layerN>/NNNNNN-xxxxxxxx.positions.npy  int32 [tokens, 3] native coordinates
    shards/<split>/<all|layerN>/NNNNNN-xxxxxxxx.types.npy      int8 [tokens] token types
    shards/<split>/<all|layerN>/NNNNNN-xxxxxxxx.json           commit marker of that shard
    index.jsonl                                  all commit markers, written by ``finish()``
    COMPLETE.json                                completion marker, written last by ``finish()``

A physical shard holds whole records of one split and one layer and at most ``shard_tokens``
tokens (a single larger record gets its own shard). Every file is written under a hidden
``.<name>.tmp-<hex>`` name, flushed, fsynced and renamed into place; the marker is renamed only
after its three arrays, so a shard is committed exactly when its marker exists. Resume removes
only recognized temporary files and marker-less shard arrays written by this format, and refuses
any committed shard whose marker, sizes, headers or SHA256 disagree (it never deletes committed data).
Readers accept only caches whose ``COMPLETE.json`` matches the header identity and the index hash
and whose index covers every expected record exactly once.
The dataset retains at most eight shard mappings in LRU order, closes evicted mappings after
copying the requested sample, and drops all mappings from its pickled state. ``dataset.close()``
releases retained mappings explicitly; subsequent indexing reopens them lazily.

Shared rclone/FUSE mounts make renames visible locally at once but may upload asynchronously.
Run exactly one writer per cache directory and let it finish (or resume it on the same host)
before reading the cache from another host; readers then check every selected file's size and
``python -m compression.cache verify DIR --full`` re-hashes all shards.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import operator
import os
import re
import secrets
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT = 'compression-feature-cache'
FORMAT_VERSION = 1
STORAGE_DTYPE = 'float16'
SPLITS = ('train', 'validation', 'test')
SPLIT_ALIASES = {'val': 'validation'}
TOKEN_TYPES = {'patch': 0, 'cls': 1, 'register': 2, 'camera': 3}
DEFAULT_LAYER = 23
HEADER, INDEX, COMPLETE, SHARDS = 'cache.json', 'index.jsonl', 'COMPLETE.json', 'shards'
ARRAYS = {'features': np.dtype('<f2'), 'positions': np.dtype('<i4'), 'types': np.dtype('i1')}
_SHARD_FILE = re.compile(r'^(\d{6}-[0-9a-f]{8})\.(json|features\.npy|positions\.npy|types\.npy)$')
_SHARD_PATH = re.compile(r'^shards/(train|validation|test)/(all|layer\d+)/(\d{6})-[0-9a-f]{8}$')
_LAYER_DIR = re.compile(r'^(all|layer\d+)$')
_ROOT_TEMP = re.compile(r'^\.(cache\.json|index\.jsonl|COMPLETE\.json)\.tmp-[0-9a-f]{8}$')
_SHARD_TEMP = re.compile(
    r'^\.\d{6}-[0-9a-f]{8}\.(json|features\.npy|positions\.npy|types\.npy)\.tmp-[0-9a-f]{8}$')
_MAX_MAPPED_SHARDS = 8
_INT32_MAX = np.iinfo(np.int32).max


class CacheError(RuntimeError):
    """Base class for cache format failures."""


class CacheIdentityError(CacheError):
    """The directory holds a cache for different provenance or expected records."""


class IncompleteCacheError(CacheError):
    """Not every expected record is committed, or completion was never published."""


class CacheIntegrityError(CacheError):
    """Committed cache content is missing, modified, duplicated or foreign."""


class DuplicateRecordError(CacheError):
    """A record was added although it is already committed or buffered."""


def normalize_split(split: str) -> str:
    """Return the canonical split name ('val' is an alias of 'validation')."""
    if not isinstance(split, str):
        raise ValueError(f'Split must be a string, got {split!r}')
    split = SPLIT_ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError(f'Invalid split {split!r}; expected one of {SPLITS} (or alias val)')
    return split


def _layer(layer: Any) -> int | None:
    if layer is None:
        return None
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) or layer < 0:
        raise ValueError(f'Layer must be a non-negative int or None, got {layer!r}')
    return int(layer)


def _layer_dir(layer: int | None) -> str:
    return 'all' if layer is None else f'layer{layer}'


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Value is not finite JSON-serializable data: {error}') from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError('Expected records must be a sequence of mappings')
    canonical, seen, splits = [], set(), {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f'Expected record must be a mapping, got {record!r}')
        extra = set(record) - {'sample_id', 'split', 'layer'}
        if extra:
            raise ValueError(f'Expected record has unsupported keys {sorted(extra)}: {record!r}')
        sample_id = record.get('sample_id')
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f'Expected record needs a non-empty string sample_id: {record!r}')
        split, layer = normalize_split(record.get('split')), _layer(record.get('layer'))
        if (sample_id, layer) in seen:
            raise ValueError(f'Duplicate expected record sample_id={sample_id!r} layer={layer}')
        if splits.setdefault(sample_id, split) != split:
            raise ValueError(f'Sample {sample_id!r} is expected in both {splits[sample_id]} and {split}')
        seen.add((sample_id, layer))
        canonical.append({'sample_id': sample_id, 'split': split, 'layer': layer})
    if not canonical:
        raise ValueError('A cache needs at least one expected record')
    if len({record['layer'] is None for record in canonical}) > 1:
        raise ValueError('Expected records mix layer=None with integer layers')
    canonical.sort(key=lambda r: (r['split'], r['sample_id'], -1 if r['layer'] is None else r['layer']))
    return canonical


def _identity_payload(provenance: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError('Provenance must be a non-empty mapping')
    provenance = json.loads(_canonical_json(dict(provenance)))
    if provenance.get('storage_dtype', STORAGE_DTYPE) != STORAGE_DTYPE:
        raise ValueError(f'This cache format stores {STORAGE_DTYPE}; provenance says {provenance["storage_dtype"]!r}')
    return {'format': FORMAT, 'format_version': FORMAT_VERSION, 'storage_dtype': STORAGE_DTYPE,
            'provenance': provenance, 'expected_records': records}


def _payload_identity(payload: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(payload).encode())


def cache_identity(provenance: Mapping[str, Any], expected_records: Iterable[Mapping[str, Any]]) -> str:
    """Stable SHA256 of the canonical provenance, storage dtype, format and sorted expected records."""
    return _payload_identity(_identity_payload(provenance, _canonical_records(expected_records)))


def _grid(grid: Sequence[int]) -> list[int]:
    try:
        values = list(grid)
        if len(values) != 3 or any(isinstance(v, (bool, np.bool_)) for v in values):
            raise TypeError
        values = [operator.index(v) for v in values]
    except TypeError:
        raise ValueError(f'Grid must be three integers (time/views, patch rows, patch cols), got {grid!r}') from None
    if min(values) < 1:
        raise ValueError(f'Grid entries must be positive, got {values}')
    return values


def _check_layout(positions: np.ndarray, types: np.ndarray, grid: Sequence[int]) -> None:
    """Validate one record's native integer token layout against its grid; raises ValueError."""
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise ValueError(f'Positions must be [N, 3] with N > 0, got {positions.shape}')
    if types.shape != (positions.shape[0],):
        raise ValueError(f'Token types must be [{positions.shape[0]}], got {types.shape}')
    if not np.isin(types, list(TOKEN_TYPES.values())).all():
        raise ValueError('Token types must be 0 patch, 1 CLS, 2 register or 3 camera')
    if (positions < 0).any() or (positions > _INT32_MAX).any():
        raise ValueError('Positions must be non-negative int32 coordinates')
    limit = np.asarray(grid, dtype=np.int64)
    patch = positions[types == TOKEN_TYPES['patch']]
    if patch.shape[0] != int(np.prod(limit)):
        raise ValueError(f'{patch.shape[0]} patch tokens do not cover grid {list(grid)} exactly')
    if (patch >= limit).any():
        raise ValueError(f'Patch position outside grid {list(grid)}')
    if positions[types == TOKEN_TYPES['cls']].any():
        raise ValueError('CLS position must be (0, 0, 0)')
    special = positions[(types == TOKEN_TYPES['register']) | (types == TOKEN_TYPES['camera'])]
    if (special[:, 0] >= limit[0]).any() or special[:, 1].any():
        raise ValueError('Register/camera positions must be (view, 0, ordinal) with view < grid[0]')
    if positions[types == TOKEN_TYPES['camera']][:, 2].any():
        raise ValueError('Camera position must be (view, 0, 0)')
    keyed = np.concatenate([types[:, None].astype(np.int64), positions.astype(np.int64)], axis=1)
    if np.unique(keyed, axis=0).shape[0] != keyed.shape[0]:
        raise ValueError('Duplicate (token type, position) in one record')


def _prepare(features: Any, positions: Any, token_types: Any, grid: Sequence[int]):
    features = np.asarray(features)
    if features.ndim != 2 or 0 in features.shape:
        raise ValueError(f'Features must be a non-empty [N, D] array, got {features.shape}')
    if features.dtype.kind != 'f':
        raise ValueError(f'Features must be floating point (convert bfloat16 tensors first), got {features.dtype}')
    if not np.isfinite(features).all():
        raise ValueError('Features contain non-finite values')
    stored = np.ascontiguousarray(features.astype(ARRAYS['features']))
    if not np.isfinite(stored).all():
        raise ValueError(f'Features exceed the {STORAGE_DTYPE} range (non-finite after storage cast)')
    positions, types = np.asarray(positions), np.asarray(token_types)
    if positions.dtype.kind not in 'iu' or types.dtype.kind not in 'iu':
        raise ValueError(f'Positions and token types must be integer arrays, got {positions.dtype}/{types.dtype}')
    if positions.shape != (features.shape[0], 3):
        raise ValueError(f'Positions must be [{features.shape[0]}, 3], got {positions.shape}')
    grid = _grid(grid)
    _check_layout(positions.astype(np.int64), types.astype(np.int64), grid)
    return (stored, np.ascontiguousarray(positions.astype(ARRAYS['positions'])),
            np.ascontiguousarray(types.astype(ARRAYS['types'])), grid)


def _temporary(path: Path) -> Path:
    return path.with_name(f'.{path.name}.tmp-{secrets.token_hex(4)}')


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass  # Some FUSE mounts do not support directory fsync.
    finally:
        os.close(descriptor)


def _publish(blocks: Iterable[Any], path: Path, *, overwrite: bool = False) -> tuple[int, str]:
    """Write byte blocks to a temporary sibling, fsync and rename; return (bytes, sha256)."""
    if not overwrite and path.exists():
        raise CacheIntegrityError(f'Refusing to overwrite existing cache file {path}')
    temporary, digest, size = _temporary(path), hashlib.sha256(), 0
    try:
        with open(temporary, 'xb') as handle:
            for block in blocks:
                view = memoryview(block if isinstance(block, bytes) else block.reshape(-1).view(np.uint8))
                handle.write(view)
                digest.update(view)
                size += view.nbytes
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)
    return size, digest.hexdigest()


def _publish_array(path: Path, dtype: np.dtype, shape: tuple[int, ...], parts: list[np.ndarray]) -> dict[str, Any]:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False, 'shape': shape})
    size, digest = _publish([header.getvalue(), *parts], path)
    return {'bytes': size, 'sha256': digest, 'dtype': dtype.str, 'shape': list(shape)}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CacheIntegrityError(f'Unreadable cache metadata {path}: {error}') from error


def _shard_file(cache_dir: Path, shard: str, kind: str) -> Path:
    return cache_dir / f'{shard}.{kind}.npy'


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_marker(marker: Any, identity: str, expected: Mapping[tuple[str, int | None], str],
                  seen: dict[tuple[str, int | None], str], dims: dict[int | None, int]) -> None:
    """Validate one shard commit marker and register its records in ``seen``/``dims``."""
    name = marker.get('shard') if isinstance(marker, dict) else None
    try:
        match = _SHARD_PATH.match(name) if isinstance(name, str) else None
        if match is None:
            raise ValueError(f'invalid shard path {name!r}')
        if marker['format'] != FORMAT or marker['format_version'] != FORMAT_VERSION:
            raise ValueError('unsupported shard format')
        if marker['identity'] != identity:
            raise ValueError(f'shard belongs to cache {marker["identity"]}, not {identity}')
        split, layer = marker['split'], marker['layer']
        if match[1] != split or match[2] != _layer_dir(layer):
            raise ValueError('shard directory disagrees with its split/layer')
        dim, tokens = marker['feature_dim'], marker['tokens']
        if not _positive_int(dim) or not _positive_int(tokens):
            raise ValueError('feature_dim and tokens must be positive integers')
        if dims.setdefault(layer, dim) != dim:
            raise ValueError(f'feature_dim {dim} differs from {dims[layer]} of other layer {layer} shards')
        shapes = {'features': [tokens, dim], 'positions': [tokens, 3], 'types': [tokens]}
        if set(marker['files']) != set(ARRAYS):
            raise ValueError('shard must list features, positions and types files')
        for kind, info in marker['files'].items():
            if info['dtype'] != ARRAYS[kind].str or info['shape'] != shapes[kind] or not _positive_int(info['bytes']):
                raise ValueError(f'{kind} file description is inconsistent')
            if not isinstance(info['sha256'], str) or len(info['sha256']) != 64:
                raise ValueError(f'{kind} file has no sha256')
        cursor = 0
        if not marker['records']:
            raise ValueError('shard has no records')
        for record in marker['records']:
            key = (record['sample_id'], layer)
            if expected.get(key) != split:
                raise ValueError(f'record {key} is not an expected {split} record')
            if key in seen:
                raise ValueError(f'duplicate record {key}, also in {seen[key]}')
            if record['start'] != cursor or not _positive_int(record['count']):
                raise ValueError(f'record {key} does not start at token {cursor}')
            _grid(record['grid'])
            if not isinstance(record['metadata'], dict):
                raise ValueError(f'record {key} metadata must be an object')
            cursor += record['count']
            seen[key] = name
        if cursor != tokens:
            raise ValueError(f'records cover {cursor} of {tokens} tokens')
    except (KeyError, TypeError, ValueError) as error:
        raise CacheIntegrityError(f'Invalid shard marker {name!r}: {error}') from error


def _npy_header(path: Path) -> tuple[list[int], np.dtype, int]:
    with open(path, 'rb') as handle:
        version = np.lib.format.read_magic(handle)
        read = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
        if read is None:
            raise ValueError(f'unsupported npy version {version}')
        shape, fortran, dtype = read(handle)
        if fortran:
            raise ValueError('Fortran-ordered array')
        return list(shape), dtype, handle.tell()


def _check_files(cache_dir: Path, marker: Mapping[str, Any], *, headers: bool) -> None:
    for kind, info in marker['files'].items():
        path = _shard_file(cache_dir, marker['shard'], kind)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise CacheIntegrityError(f'Committed shard file is missing: {path} ({error})') from error
        if size != info['bytes']:
            raise CacheIntegrityError(f'Committed shard file {path} has {size} bytes, marker records {info["bytes"]}')
        if headers:
            try:
                shape, dtype, offset = _npy_header(path)
            except (OSError, ValueError) as error:
                raise CacheIntegrityError(f'Unreadable shard array {path}: {error}') from error
            if shape != info['shape'] or dtype.str != info['dtype'] or offset + int(np.prod(shape)) * dtype.itemsize != size:
                raise CacheIntegrityError(f'Shard array {path} header disagrees with its marker')


def _expected_map(records: list[dict[str, Any]]) -> dict[tuple[str, int | None], str]:
    return {(record['sample_id'], record['layer']): record['split'] for record in records}


def _load_header(cache_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = cache_dir / HEADER
    if not path.is_file():
        raise CacheError(f'{cache_dir} is not a feature cache (no {HEADER})')
    header = _read_json(path)
    try:
        if header['format'] != FORMAT or header['format_version'] != FORMAT_VERSION:
            raise CacheIdentityError(f'{cache_dir} has unsupported cache format {header["format"]!r} '
                                     f'v{header["format_version"]}')
        records = _canonical_records(header['expected_records'])
        payload = _identity_payload(header['provenance'], records)
    except (KeyError, TypeError, ValueError) as error:
        raise CacheIntegrityError(f'Invalid cache header {path}: {error}') from error
    if records != header['expected_records'] or _payload_identity(payload) != header.get('identity'):
        raise CacheIntegrityError(f'Cache header {path} does not hash to its recorded identity')
    return header, records


def _load_completed(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load and cross-check header, completion marker and index of a finished cache."""
    header, records = _load_header(cache_dir)
    identity = header['identity']
    if not (cache_dir / COMPLETE).is_file():
        raise IncompleteCacheError(f'{cache_dir} has no {COMPLETE}; the writer never finished (resume it)')
    complete = _read_json(cache_dir / COMPLETE)
    if not isinstance(complete, dict) or complete.get('identity') != identity:
        raise CacheIntegrityError(f'{COMPLETE} in {cache_dir} does not belong to identity {identity}')
    try:
        index = (cache_dir / INDEX).read_bytes()
    except OSError as error:
        raise CacheIntegrityError(f'Completed cache {cache_dir} has no readable {INDEX}: {error}') from error
    if _sha256(index) != complete.get('index_sha256'):
        raise CacheIntegrityError(f'{INDEX} in {cache_dir} does not match the hash in {COMPLETE}')
    try:
        markers = [json.loads(line) for line in index.decode().splitlines()]
    except ValueError as error:
        raise CacheIntegrityError(f'Corrupt {INDEX} in {cache_dir}: {error}') from error
    expected, seen, dims = _expected_map(records), {}, {}
    names = [marker.get('shard') if isinstance(marker, dict) else None for marker in markers]
    if len(set(names)) != len(names):
        raise CacheIntegrityError(f'{INDEX} in {cache_dir} lists a shard twice')
    for marker in markers:
        _check_marker(marker, identity, expected, seen, dims)
    if len(seen) != len(expected):
        raise CacheIntegrityError(f'{INDEX} in {cache_dir} covers {len(seen)} of {len(expected)} expected records')
    if complete.get('records') != len(seen) or complete.get('shards') != len(markers):
        raise CacheIntegrityError(f'{COMPLETE} counts disagree with {INDEX} in {cache_dir}')
    return header, complete, markers


class FeatureCacheWriter:
    """Append-only writer for one cache directory; see the module docstring for the format.

    ``contains`` reports committed records only. Records added but not yet flushed into a shard are
    lost on interruption and are reported by ``missing()`` again after ``resume=True``.
    """

    def __init__(self, output: str | os.PathLike, provenance: Mapping[str, Any],
                 expected_records: Iterable[Mapping[str, Any]], *, shard_tokens: int = 65536, resume: bool = False):
        if not _positive_int(shard_tokens):
            raise ValueError(f'shard_tokens must be a positive integer, got {shard_tokens!r}')
        self.output = Path(output)
        self.shard_tokens = shard_tokens
        records = _canonical_records(expected_records)
        payload = _identity_payload(provenance, records)
        self.identity = _payload_identity(payload)
        self.provenance = payload['provenance']
        self._records = records
        self._expected = _expected_map(records)
        self._committed: dict[tuple[str, int | None], str] = {}
        self._dims: dict[int | None, int] = {}
        self._markers: list[dict[str, Any]] = []
        self._buffers: dict[tuple[str, int | None], dict[str, Any]] = {}
        self._pending: set[tuple[str, int | None]] = set()
        self._next_shard = 0
        self._complete = self._closed = self._failed = False
        if (self.output / HEADER).exists():
            if not resume:
                raise FileExistsError(f'{self.output} already holds a feature cache; pass resume=True to continue it')
            header, _ = _load_header(self.output)
            if header['identity'] != self.identity:
                raise CacheIdentityError(f'{self.output} holds cache {header["identity"]}, '
                                         f'but this provenance/selection is {self.identity}')
            self._recover()
        else:
            self._create(payload)

    def _create(self, payload: dict[str, Any]) -> None:
        if self.output.exists():
            if not self.output.is_dir():
                raise FileExistsError(f'Cache output {self.output} exists and is not a directory')
            entries = list(self.output.iterdir())
            foreign = [entry.name for entry in entries if not (entry.is_file()
                       and (match := _ROOT_TEMP.fullmatch(entry.name)) and match[1] == HEADER)]
            if foreign:
                raise FileExistsError(f'Refusing to write a cache into non-empty directory {self.output} '
                                      f'(contains {sorted(foreign)[:5]})')
            for entry in entries:
                entry.unlink()
        self.output.mkdir(parents=True, exist_ok=True)
        text = json.dumps({**payload, 'identity': self.identity}, sort_keys=True, indent=1,
                          ensure_ascii=False, allow_nan=False)
        _publish([(text + '\n').encode()], self.output / HEADER)

    def _recover(self) -> None:
        for entry in self.output.iterdir():
            if _ROOT_TEMP.fullmatch(entry.name) and entry.is_file():
                entry.unlink()
        groups: dict[tuple[str, str], set[str]] = {}
        root = self.output / SHARDS
        if root.is_dir():
            for directory, subdirectories, names in os.walk(root):
                relative = Path(directory).relative_to(self.output).as_posix()
                depth = relative.count('/')
                if (depth == 1 and Path(directory).name not in SPLITS) or depth > 2 or (
                        depth == 2 and not _LAYER_DIR.match(Path(directory).name)):
                    raise CacheIntegrityError(f'Unexpected directory {directory} in cache {self.output}')
                subdirectories.sort()
                for name in sorted(names):
                    if depth == 2 and _SHARD_TEMP.fullmatch(name):
                        (Path(directory) / name).unlink()
                        continue
                    match = _SHARD_FILE.match(name)
                    if depth != 2 or match is None:
                        raise CacheIntegrityError(f'Unexpected file {Path(directory) / name} in cache {self.output}')
                    groups.setdefault((relative, match[1]), set()).add(match[2])
        for (relative, base), kinds in sorted(groups.items()):
            self._next_shard = max(self._next_shard, int(base[:6]) + 1)
            if 'json' not in kinds:
                for kind in kinds:  # Arrays whose commit marker never landed: own uncommitted output.
                    (self.output / relative / f'{base}.{kind}').unlink()
                continue
            marker = _read_json(self.output / relative / f'{base}.json')
            _check_marker(marker, self.identity, self._expected, self._committed, self._dims)
            if marker['shard'] != f'{relative}/{base}':
                raise CacheIntegrityError(f'Marker {relative}/{base}.json names shard {marker["shard"]!r}')
            _check_files(self.output, marker, headers=True)
            _check_digests(self.output, marker)
            self._markers.append(marker)
        if (self.output / COMPLETE).exists():
            _, _, markers = _load_completed(self.output)
            if sorted(m['shard'] for m in markers) != sorted(m['shard'] for m in self._markers):
                raise CacheIntegrityError(f'{INDEX} of completed cache {self.output} disagrees with its shards')
            self._complete = True

    def _require_open(self) -> None:
        if self._failed:
            raise CacheError(f'Writer for {self.output} failed while publishing a shard; reopen with resume=True')
        if self._complete or self._closed:
            raise CacheError(f'Writer for {self.output} is {"complete" if self._complete else "closed"}')

    def _key(self, sample_id: str, layer: int | None) -> tuple[str, int | None]:
        key = (sample_id, _layer(layer))
        if key not in self._expected:
            raise KeyError(f'sample_id={sample_id!r} layer={key[1]} is not an expected record of cache {self.identity}')
        return key

    def contains(self, sample_id: str, layer: int | None = None) -> bool:
        """True when the record is committed in a published shard (buffered records are not)."""
        return self._key(sample_id, layer) in self._committed

    def missing(self) -> list[dict[str, Any]]:
        """Expected records neither committed nor buffered, in canonical order."""
        return [dict(record) for record in self._records
                if (record['sample_id'], record['layer']) not in self._committed
                and (record['sample_id'], record['layer']) not in self._pending]

    def add(self, *, sample_id: str, split: str, features: Any, positions: Any, token_types: Any,
            grid: Sequence[int], metadata: Mapping[str, Any] | None = None, layer: int | None = None) -> None:
        """Validate one record, cast features to float16 and buffer it for a bounded shard."""
        self._require_open()
        split = normalize_split(split)
        key = self._key(sample_id, layer)
        if self._expected[key] != split:
            raise ValueError(f'{sample_id!r} is expected in split {self._expected[key]}, not {split}')
        if key in self._committed or key in self._pending:
            raise DuplicateRecordError(f'sample_id={sample_id!r} layer={key[1]} was already added to {self.output}')
        features, positions, token_types, grid = _prepare(features, positions, token_types, grid)
        dim = features.shape[1]
        if self._dims.get(key[1], dim) != dim:
            raise ValueError(f'Feature dim {dim} differs from {self._dims[key[1]]} of other layer {key[1]} records')
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f'Metadata must be a mapping, got {type(metadata).__name__}')
        metadata = json.loads(_canonical_json(dict(metadata or {})))
        self._dims[key[1]] = dim
        buffer_key = (split, key[1])
        buffer = self._buffers.get(buffer_key)
        if buffer and buffer['tokens'] + features.shape[0] > self.shard_tokens:
            self._flush(buffer_key)
        buffer = self._buffers.setdefault(buffer_key, {'tokens': 0, 'records': []})
        buffer['records'].append((sample_id, features, positions, token_types, grid, metadata))
        buffer['tokens'] += features.shape[0]
        self._pending.add(key)
        if buffer['tokens'] >= self.shard_tokens:
            self._flush(buffer_key)

    def _flush(self, buffer_key: tuple[str, int | None]) -> None:
        buffer = self._buffers.pop(buffer_key, None)
        if not buffer:
            return
        split, layer = buffer_key
        records, tokens = buffer['records'], buffer['tokens']
        dim = records[0][1].shape[1]
        relative = f'{SHARDS}/{split}/{_layer_dir(layer)}'
        shard = f'{relative}/{self._next_shard:06d}-{secrets.token_hex(4)}'
        self._next_shard += 1
        try:
            (self.output / relative).mkdir(parents=True, exist_ok=True)
            files = {kind: _publish_array(_shard_file(self.output, shard, kind), ARRAYS[kind], shape,
                                          [record[column] for record in records])
                     for kind, column, shape in (('features', 1, (tokens, dim)), ('positions', 2, (tokens, 3)),
                                                 ('types', 3, (tokens,)))}
            entries, start = [], 0
            for sample_id, features, _, _, grid, metadata in records:
                entries.append({'sample_id': sample_id, 'start': start, 'count': features.shape[0],
                                'grid': grid, 'metadata': metadata})
                start += features.shape[0]
            marker = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'identity': self.identity, 'shard': shard,
                      'split': split, 'layer': layer, 'feature_dim': dim, 'tokens': tokens, 'files': files,
                      'records': entries}
            _publish([(_canonical_json(marker) + '\n').encode()], self.output / f'{shard}.json')
        except BaseException:
            self._failed = True
            raise
        for entry in entries:
            key = (entry['sample_id'], layer)
            self._pending.discard(key)
            self._committed[key] = shard
        self._markers.append(marker)

    def flush(self) -> None:
        """Commit every buffered record (possibly as shards smaller than shard_tokens)."""
        self._require_open()
        for buffer_key in sorted(self._buffers, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
            self._flush(buffer_key)

    def close(self) -> None:
        """Flush buffered records and stop accepting new ones (does not mark the cache complete)."""
        if not (self._closed or self._complete or self._failed):
            self.flush()
        self._closed = True

    def __enter__(self) -> 'FeatureCacheWriter':
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def finish(self) -> Path:
        """Publish index and completion marker once every expected record is committed."""
        if self._complete:
            _load_completed(self.output)
            return self.output
        if self._failed:
            self._require_open()
        for buffer_key in sorted(self._buffers, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
            self._flush(buffer_key)
        absent = [key for key in self._expected if key not in self._committed]
        if absent:
            raise IncompleteCacheError(f'{len(absent)} of {len(self._expected)} expected records are not cached '
                                       f'in {self.output}, e.g. {absent[:3]}')
        markers = sorted(self._markers, key=lambda marker: marker['shard'])
        index = ''.join(_canonical_json(marker) + '\n' for marker in markers).encode()
        _publish([index], self.output / INDEX, overwrite=True)
        splits: dict[str, int] = {}
        for split in self._expected.values():
            splits[split] = splits.get(split, 0) + 1
        complete = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'identity': self.identity,
                    'records': len(self._committed), 'shards': len(markers),
                    'tokens': sum(marker['tokens'] for marker in markers), 'splits': splits,
                    'index_sha256': _sha256(index)}
        _publish([(json.dumps(complete, sort_keys=True, indent=1) + '\n').encode()], self.output / COMPLETE)
        self._complete = True
        _load_completed(self.output)
        return self.output


def _select_layer(available: list[int | None], requested: Any) -> int | None:
    if requested is not None:
        requested = _layer(requested)
        if requested not in available:
            raise ValueError(f'Layer {requested} is not cached; available layers {available}')
        return requested
    if available == [None]:
        return None
    if DEFAULT_LAYER in available:
        return DEFAULT_LAYER
    if len(available) == 1:
        return available[0]
    raise ValueError(f'Cache holds layers {available} without default layer {DEFAULT_LAYER}; pass layer=')


class FeatureCacheDataset(Dataset):
    """Read-only, memory-mapped view of one split and one layer of a completed cache.

    Items: features float32 [N, D] (native target space), positions float32 [N, 3] (native integer
    coordinates divided by max(grid - 1, 1) per axis), token_types int64 [N], grid int64 [3], sample_id.
    """

    def __init__(self, cache_dir: str | os.PathLike, split: str, *, layer: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.split = normalize_split(split)
        header, _, markers = _load_completed(self.cache_dir)
        self.identity: str = header['identity']
        self.provenance: dict[str, Any] = header['provenance']
        available = sorted({record['layer'] for record in header['expected_records']},
                           key=lambda value: -1 if value is None else value)
        self.layer = _select_layer(available, layer)
        shards = [marker for marker in markers if marker['split'] == self.split and marker['layer'] == self.layer]
        if not shards:
            raise ValueError(f'Cache {self.identity} has no {self.split} records for layer {self.layer}')
        for marker in shards:
            _check_files(self.cache_dir, marker, headers=False)
        self.feature_dim: int = shards[0]['feature_dim']
        self._shards = [(marker['shard'], marker['files']) for marker in shards]
        items = sorted((record['sample_id'], position, record['start'], record['count'], record['grid'],
                        record['metadata']) for position, marker in enumerate(shards) for record in marker['records'])
        self.sample_ids: list[str] = [item[0] for item in items]
        self.token_counts: list[int] = [item[3] for item in items]
        self.grids: list[list[int]] = [list(item[4]) for item in items]
        self.metadata: list[dict[str, Any]] = [item[5] for item in items]
        self._items = [(shard, start) for _, shard, start, _, _, _ in items]
        self._maps: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def _arrays(self, shard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = self._maps.get(shard)
        if arrays is None:
            name, files = self._shards[shard]
            loaded = []
            try:
                for kind in ('features', 'positions', 'types'):
                    path = _shard_file(self.cache_dir, name, kind)
                    try:
                        array = np.load(path, mmap_mode='r', allow_pickle=False)
                    except (OSError, ValueError) as error:
                        raise CacheIntegrityError(f'Unreadable shard array {path}: {error}') from error
                    loaded.append(array)
                    if list(array.shape) != files[kind]['shape'] or array.dtype.str != files[kind]['dtype']:
                        raise CacheIntegrityError(f'Shard array {path} disagrees with its marker')
            except BaseException:
                self._close_arrays(loaded)
                raise
            arrays = self._maps[shard] = tuple(loaded)
        self._maps.move_to_end(shard)
        return arrays

    @staticmethod
    def _close_arrays(arrays: Iterable[np.ndarray]) -> None:
        for array in arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()

    def _trim_maps(self) -> None:
        while len(self._maps) > _MAX_MAPPED_SHARDS:
            _, arrays = self._maps.popitem(last=False)
            self._close_arrays(arrays)

    def close(self) -> None:
        """Release all retained mappings; later indexing reopens them lazily."""
        for arrays in self._maps.values():
            self._close_arrays(arrays)
        self._maps.clear()

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, start = self._items[index]
        stop = start + self.token_counts[index]
        features, positions, types = self._arrays(shard)
        grid = np.asarray(self.grids[index], dtype=np.int64)
        scale = np.maximum(grid - 1, 1).astype(np.float32)
        try:
            return {'features': torch.from_numpy(np.array(features[start:stop], dtype=np.float32)),
                    'positions': torch.from_numpy(np.array(positions[start:stop], dtype=np.float32) / scale),
                    'token_types': torch.from_numpy(np.array(types[start:stop], dtype=np.int64)),
                    'grid': torch.from_numpy(grid),
                    'sample_id': self.sample_ids[index]}
        finally:
            # Tensors own copies, so releasing older mappings cannot invalidate returned samples.
            self._trim_maps()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['_maps'] = OrderedDict()  # Memory maps are reopened lazily in each DataLoader worker.
        return state


def collate_features(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad samples to the longest token count; valid_mask is True for real tokens, padding is zero."""
    samples = list(samples)
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    dims = {sample['features'].shape[-1] for sample in samples}
    if len(dims) != 1:
        raise ValueError(f'Samples have different feature dims {sorted(dims)}')
    batch, length, dim = len(samples), max(sample['features'].shape[0] for sample in samples), dims.pop()
    features = torch.zeros(batch, length, dim, dtype=samples[0]['features'].dtype)
    positions = torch.zeros(batch, length, 3, dtype=torch.float32)
    token_types = torch.zeros(batch, length, dtype=torch.int64)
    valid_mask = torch.zeros(batch, length, dtype=torch.bool)
    for row, sample in enumerate(samples):
        count = sample['features'].shape[0]
        if sample['positions'].shape != (count, 3) or sample['token_types'].shape != (count,):
            raise ValueError(f'Sample {sample["sample_id"]!r} has inconsistent token shapes')
        features[row, :count] = sample['features']
        positions[row, :count] = sample['positions']
        token_types[row, :count] = sample['token_types']
        valid_mask[row, :count] = True
    collated = {'features': features, 'positions': positions, 'token_types': token_types,
                'valid_mask': valid_mask, 'sample_ids': [sample['sample_id'] for sample in samples]}
    if all('grid' in sample for sample in samples):
        collated['grid'] = torch.stack([torch.as_tensor(sample['grid'], dtype=torch.int64) for sample in samples])
    return collated


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _check_digests(cache_dir: Path, marker: Mapping[str, Any]) -> None:
    for kind, info in marker['files'].items():
        path = _shard_file(cache_dir, marker['shard'], kind)
        try:
            digest = _file_sha256(path)
        except OSError as error:
            raise CacheIntegrityError(f'Unreadable committed shard array {path}: {error}') from error
        if digest != info['sha256']:
            raise CacheIntegrityError(f'{path} does not match its recorded sha256')


def verify_cache(cache_dir: str | os.PathLike, *, full: bool = False) -> dict[str, Any]:
    """Verify a completed cache; ``full`` re-hashes every shard and re-checks values and layouts."""
    cache_dir = Path(cache_dir)
    header, complete, markers = _load_completed(cache_dir)
    records: dict[str, dict[str, int]] = {}
    dims: dict[str, int] = {}
    for marker in markers:
        _check_files(cache_dir, marker, headers=True)
        layer = _layer_dir(marker['layer'])
        dims[layer] = marker['feature_dim']
        per_split = records.setdefault(marker['split'], {})
        per_split[layer] = per_split.get(layer, 0) + len(marker['records'])
        if not full:
            continue
        _check_digests(cache_dir, marker)
        features, positions, types = (np.load(_shard_file(cache_dir, marker['shard'], kind), mmap_mode='r')
                                      for kind in ('features', 'positions', 'types'))
        for start in range(0, features.shape[0], 8192):
            if not np.isfinite(features[start:start + 8192]).all():
                raise CacheIntegrityError(f'{marker["shard"]} contains non-finite features')
        for record in marker['records']:
            stop = record['start'] + record['count']
            try:
                _check_layout(np.asarray(positions[record['start']:stop], dtype=np.int64),
                              np.asarray(types[record['start']:stop], dtype=np.int64), record['grid'])
            except ValueError as error:
                raise CacheIntegrityError(f'{marker["shard"]} record {record["sample_id"]!r}: {error}') from error
    return {'cache_dir': str(cache_dir), 'identity': header['identity'], 'records': complete['records'],
            'shards': complete['shards'], 'tokens': complete['tokens'], 'records_by_split_layer': records,
            'feature_dims': dims, 'full': full}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Feature cache utilities')
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='check that a cache is complete and intact')
    verify.add_argument('cache_dir', type=Path)
    verify.add_argument('--full', action='store_true', help='re-hash shards and re-check values/layouts')
    args = parser.parse_args(argv)
    try:
        summary = verify_cache(args.cache_dir, full=args.full)
    except (CacheError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
