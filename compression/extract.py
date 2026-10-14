"""Write frozen backbone features into resumable compression caches.

One process loads one backbone once (strict native loaders from experiments/<track>), preprocesses manifest
records on a bounded thread prefetcher, runs batched inference, and commits every selected record through
compression.cache.FeatureCacheWriter under <output-root>/<model>/<identity>. Tokens are stored in native
order with special tokens included (type 0 patch, 1 CLS, 2 register, 3 camera).

    python -m compression.extract --model dinov3 --manifest train.jsonl --manifest validation.jsonl \
        --device cuda --compute-dtype bfloat16 --batch-size 16 --resume
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import hashlib
import importlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORKSTATION_ROOT = '/home/engeld/mnt/datawaha/compressing-latents'
SCHEMA = 'compression.extract/1'
MODELS = ('dinov3', 'qwen38', 'vggt', 'vjepa21')
PATCH, CLS, REGISTER, CAMERA = 0, 1, 2, 3
TOKEN_TYPES = {'patch': PATCH, 'cls': CLS, 'register': REGISTER, 'camera': CAMERA}
SPLIT_ALIASES = {'val': 'validation'}
SPLITS = ('train', 'validation', 'test')
IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# Pinned V-JEPA 2.1 SSv2 frozen evaluation (configs/eval_2_1/vitb-384/ssv2.yaml at the source revision).
SSV2_EVAL_CONFIG = 'configs/eval_2_1/vitb-384/ssv2.yaml'
SSV2_EVAL = {'frames_per_clip': 16, 'frame_step': 4, 'num_segments': 2, 'num_views_per_segment': 3,
             'resolution': 384}


@dataclasses.dataclass(frozen=True)
class Record:
    sample_id: str
    split: str
    paths: tuple
    source: dict


class PathMap:
    """Explicit prefix rewriting of manifest/asset paths; longest matching OLD prefix wins."""

    def __init__(self, pairs):
        olds = [old for old, _ in pairs]
        if len(set(olds)) != len(olds):
            raise ValueError('Each --path-map OLD prefix may appear once')
        for old, new in pairs:
            if not (old.startswith('/') and new.startswith('/')):
                raise ValueError(f'--path-map needs absolute OLD=NEW prefixes, got {old}={new}')
        self.pairs = sorted(((old.rstrip('/') or '/', new.rstrip('/') or '/') for old, new in pairs),
                            key=lambda pair: -len(pair[0]))

    def __call__(self, path):
        text = str(path)
        for old, new in self.pairs:
            if text == old or text.startswith(old + '/'):
                return Path(new + text[len(old):])
        return Path(text)


def build_path_map(root, values):
    pairs = [] if str(root).rstrip('/') == WORKSTATION_ROOT else [(WORKSTATION_ROOT, str(root))]
    for value in values:
        old, separator, new = value.partition('=')
        if not separator:
            raise ValueError(f'--path-map expects OLD=NEW, got {value!r}')
        pairs.append((old, new))
    return PathMap(pairs)


def normalize_split(split):
    split = SPLIT_ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError(f'Unsupported cache split {split!r}; expected one of {SPLITS} (or alias val)')
    return split


def load_manifests(manifests, path_map, limit_per_split=None):
    """Read JSONL manifests in order; content SHA256 (not paths after mapping) identifies the selection."""
    records, identities, seen = [], [], set()
    for manifest in manifests:
        data = manifest.read_bytes()
        rows = 0
        for line_number, line in enumerate(data.decode().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f'{manifest}:{line_number}'
            sample_id, paths = row.get('sample_id'), row.get('paths')
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f'{where}: missing sample_id')
            if sample_id in seen:
                raise ValueError(f'{where}: duplicate sample_id {sample_id}')
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
                raise ValueError(f'{where}: paths must be a nonempty list of strings')
            seen.add(sample_id)
            resolved = tuple(path_map(p) if Path(p).is_absolute() else manifest.parent / p for p in paths)
            records.append(Record(sample_id, normalize_split(row.get('split')), resolved, row))
            rows += 1
        identities.append({'sha256': hashlib.sha256(data).hexdigest(), 'records': rows})
    if limit_per_split is not None:
        counts = collections.Counter()
        kept = []
        for record in records:
            if counts[record.split] < limit_per_split:
                counts[record.split] += 1
                kept.append(record)
        records = kept
    if not records:
        raise ValueError('Manifests select no records')
    return records, identities


def video_eval_clip_indices(num_frames, *, frames_per_clip=16, frame_step=4, num_segments=2,
                            allow_segment_overlap=True, segment=0):
    """Frame indices of one temporal segment from pinned VideoDataset.loadvideo_decord.

    Exact upstream linspace/clip/padding arithmetic with random_clip_sampling=False (window ends at
    clip_len inside its partition). Returns (int64 indices[frames_per_clip], branch name).
    """
    if num_frames < 2:
        raise ValueError('Upstream sampler yields index -1 for videos shorter than two frames')
    if not 0 <= segment < num_segments:
        raise ValueError('segment must index one of num_segments')
    fpc, fstp = frames_per_clip, frame_step
    clip_len = int(fpc * fstp)
    partition_len = num_frames // num_segments
    if partition_len > clip_len:
        branch = 'partition_window'
        end_indx = clip_len
        start_indx = end_indx - clip_len
        indices = np.linspace(start_indx, end_indx, num=fpc)
        indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
        indices = indices + segment * partition_len
    elif not allow_segment_overlap:
        branch = 'short_partition_pad'
        indices = np.linspace(0, partition_len, num=partition_len // fstp)
        indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len))
        indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
        indices = indices + segment * partition_len
    else:
        branch = 'overlapping_short_pad'
        sample_len = min(clip_len, num_frames) - 1
        indices = np.linspace(0, sample_len, num=sample_len // fstp)
        indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len))
        indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
        clip_step = 0
        if num_frames > clip_len:
            clip_step = (num_frames - clip_len) // (num_segments - 1)
        indices = indices + segment * clip_step
    if indices.shape != (fpc,) or indices.min() < 0 or indices.max() >= num_frames:
        raise ValueError(f'Sampler produced invalid indices for {num_frames} frames: {indices.tolist()}')
    return indices, branch


def image_layout(grid_hw, num_registers):
    """CLS, registers, then row-major patches: native DINO forward_features order."""
    rows, cols = np.meshgrid(np.arange(grid_hw[0]), np.arange(grid_hw[1]), indexing='ij')
    patches = np.stack([np.zeros(rows.size, np.int64), rows.ravel(), cols.ravel()], axis=1)
    registers = np.stack([np.zeros(num_registers, np.int64), np.zeros(num_registers, np.int64),
                          np.arange(num_registers)], axis=1)
    positions = np.concatenate([np.zeros((1, 3), np.int64), registers, patches])
    types = np.concatenate([[CLS], np.full(num_registers, REGISTER), np.full(len(patches), PATCH)]).astype(np.int64)
    return positions, types


def dense_layout(grid):
    """Patch-only (T,H,W) grid in native time/row/col flatten order."""
    axes = np.meshgrid(*(np.arange(n) for n in grid), indexing='ij')
    positions = np.stack([axis.ravel() for axis in axes], axis=1).astype(np.int64)
    return positions, np.zeros(len(positions), np.int64)


def view_group_layout(views, grid_hw, num_registers):
    """Per view: camera, registers, row-major patches; the native Omega aggregator token order."""
    positions, types = [], []
    rows, cols = np.meshgrid(np.arange(grid_hw[0]), np.arange(grid_hw[1]), indexing='ij')
    for view in range(views):
        positions += [[[view, 0, 0]], [[view, 0, r] for r in range(num_registers)],
                      np.stack([np.full(rows.size, view), rows.ravel(), cols.ravel()], axis=1)]
        types += [[CAMERA], [REGISTER] * num_registers, [PATCH] * rows.size]
    return (np.concatenate([np.asarray(p, np.int64).reshape(-1, 3) for p in positions]),
            np.concatenate([np.asarray(t, np.int64) for t in types]))


def native_module(track, name):
    """Import experiments/<track>/<name>.py with its sibling-module imports (one track per process)."""
    directory = str(REPO / 'experiments' / track)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(name)


def module_versions(*names):
    versions = {}
    for name in ('numpy', 'torch') + names:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, '__version__', 'unknown'))
    return versions


def json_safe(value):
    return json.loads(json.dumps(value, default=str))


def autocast(device, compute_dtype):
    import torch
    if compute_dtype == 'float32':
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=getattr(torch, compute_dtype))


def numpy_tokens(tensor):
    return tensor.detach().float().cpu().numpy()


class Dinov3:
    layers = (None,)
    max_batch = None

    def __init__(self, args, path_map):
        import torch
        self.native = native = native_module('dinov3', 'extract')
        checkpoint = args.checkpoint or path_map(native.DEFAULT_CHECKPOINT)
        self.device, self.dtype = args.device, getattr(torch, args.compute_dtype)
        self.model, asset = native.load_model(checkpoint, args.device, args.compute_dtype)
        self.transform = native.make_transform()
        grid = native.SIZE // self.model.patch_size
        self.grid = (1, grid, grid)
        self.positions, self.types = image_layout((grid, grid), self.model.n_storage_tokens)
        self.provenance = {
            'model_id': native.MODEL_ID, 'source_revision': native.REVISION,
            'checkpoint_sha256': asset['checkpoint']['sha256'], 'layer': native.LAYER,
            'layer_indexing': 'zero-based final transformer block',
            'tensor_boundary': native.TENSOR_BOUNDARY, 'special_token_boundary': native.SPECIAL_TOKEN_BOUNDARY,
            'native_normalization': {'type': 'learned LayerNorm', 'eps': self.model.norm.eps,
                                     'extra_compressor_normalization': None},
            'preprocessing': json_safe(native.PREPROCESSING),
            'compute_policy': 'native DINO extractor: weights and pixels cast to compute dtype',
            'feature_dim': self.model.embed_dim, 'grid': list(self.grid),
            'token_layout': 'CLS (0,0,0); registers (0,0,ordinal); patches (0,row,col) row-major; native order',
            'versions': module_versions('torchvision', 'PIL')}

    def prepare(self, record):
        if len(record.paths) != 1:
            raise ValueError(f'{record.sample_id}: DINO samples need exactly one image')
        return self.native.load_image(record.paths[0], self.transform)

    def forward(self, payloads):
        import torch
        pixels = torch.stack([pixels for pixels, _ in payloads]).to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            out = self.model.forward_features(pixels)
            tokens = torch.cat([out['x_norm_clstoken'][:, None], out['x_storage_tokens'],
                                out['x_norm_patchtokens']], dim=1)
        tokens = numpy_tokens(tokens)
        if tokens.shape[1:] != (len(self.types), self.model.embed_dim):
            raise RuntimeError(f'Unexpected DINO token shape {tokens.shape}')
        return [{None: (tokens[i], self.positions, self.types, self.grid, payloads[i][1])}
                for i in range(len(payloads))]


class Qwen38:
    layers = (None,)
    max_batch = None
    size = 384

    def __init__(self, args, path_map):
        self.native = native = native_module('qwen38', 'experiment')
        checkpoint = args.checkpoint or path_map(native.CHECKPOINT)
        self.device, self.compute_dtype = args.device, args.compute_dtype
        self.model, provenance = native.load_model(checkpoint, args.device)
        self.transform, transform_config = native.make_transform(self.model, self.size)
        grid = self.size // 16
        self.grid = (1, grid, grid)
        self.positions, self.types = dense_layout(self.grid)
        self.provenance = {
            'model_id': native.MODEL_ID, 'model_revision': native.MODEL_REVISION,
            'source_id': native.SOURCE_ID, 'source_revision': native.SOURCE_REVISION,
            'timm_revision': native.TIMM_REVISION, 'checkpoint_sha256': provenance['checkpoint_sha256'],
            'layer': native.LAYER, 'tensor_boundary': native.TENSOR_BOUNDARY,
            'native_normalization': native.OUTPUT_NORMALIZATION,
            'preprocessing': json_safe({**transform_config, **native.PREPROCESSING_POLICY}),
            'temporal_image_handling': native.TEMPORAL_IMAGE_HANDLING,
            'compute_policy': 'float32 weights; torch.autocast to compute dtype when not float32',
            'feature_dim': 1152, 'grid': list(self.grid),
            'token_layout': 'patches only (0,row,col) row-major; visual tower has no CLS/register tokens',
            'versions': module_versions('torchvision', 'PIL', 'timm', 'safetensors')}

    def prepare(self, record):
        if len(record.paths) != 1:
            raise ValueError(f'{record.sample_id}: Qwen samples need exactly one image')
        pixels, original_size = self.native.load_image(record.paths[0], self.transform)
        return pixels, {'original_size_wh': original_size}

    def forward(self, payloads):
        import torch
        pixels = torch.stack([pixels for pixels, _ in payloads]).to(self.device)
        with torch.inference_mode(), autocast(self.device, self.compute_dtype):
            raw = self.model.forward_features(pixels)
        if tuple(raw.shape) != (len(payloads), self.grid[1], self.grid[2], 1152):
            raise RuntimeError(f'Expected raw NHWC pre-merger tokens, got {tuple(raw.shape)}')
        tokens = numpy_tokens(raw.reshape(len(payloads), -1, 1152))
        return [{None: (tokens[i], self.positions, self.types, self.grid, payloads[i][1])}
                for i in range(len(payloads))]


class Vggt:
    max_batch = 1
    mode = 'balanced'

    def __init__(self, args, path_map):
        assets = native_module('vggt', 'assets')
        try:
            importlib.import_module('vggt_omega')
        except ImportError:
            sys.path.insert(0, str(path_map(assets.SOURCE)))
        self.native = native = native_module('vggt', 'extract')
        self.layers = tuple(native.parse_layers(args.layers))
        checkpoint = args.checkpoint or path_map(assets.CHECKPOINT)
        self.device, self.compute_dtype = args.device, args.compute_dtype
        self.aggregator = native.load_aggregator(checkpoint, self.layers, args.device)
        from vggt_omega.models.aggregator import _RESNET_MEAN, _RESNET_STD
        self.provenance = {
            'model_id': assets.MODEL_ID, 'model_variant': assets.MODEL_VARIANT,
            'model_revision': assets.MODEL_REVISION, 'source_revision': assets.SOURCE_REVISION,
            'checkpoint_sha256': assets.CHECKPOINT_SHA256, 'layers': list(self.layers),
            'tensor_boundary': {str(layer): native.tensor_boundary(layer) for layer in self.layers},
            'inter_frame_attention_type': {str(layer): self.aggregator.inter_frame_attention_types[layer]
                                           for layer in self.layers},
            'native_normalization': {'input_rgb_range': [0, 1], 'internal_mean': list(_RESNET_MEAN),
                                     'internal_std': list(_RESNET_STD), 'extra_feature_normalization': None},
            'preprocessing': native.preprocessing_policy(self.mode),
            'context': native.CONTEXT, 'reference_view_index': 0,
            'reference_convention': native.REFERENCE_CONVENTION,
            'head_required_layers': native.HEAD_REQUIRED_LAYERS, 'benchmark_caveat': native.BENCHMARK_CAVEAT,
            'compute_policy': 'float32 weights; torch.autocast to compute dtype on CUDA when not float32',
            'feature_dim': 2048, 'channel_halves': {'frame': [0, 1024], 'inter_frame': [1024, 2048]},
            'token_layout': 'per ordered view: camera (view,0,0); 16 registers (view,0,ordinal); patches (view,row,col) row-major; native aggregator order; grid (views,rows,cols)',
            'versions': module_versions('torchvision', 'PIL')}

    def prepare(self, record):
        return (self.native.load_images(record.paths, self.mode),
                self.native.transform_metadata(record.paths, self.mode))

    def forward(self, payloads):
        (images, transforms), = payloads
        outputs, patch_start, grid, input_shape = self.native.forward_group(
            self.aggregator, images, self.device, self.compute_dtype)
        views, rows, cols = grid
        positions, types = view_group_layout(views, (rows, cols), patch_start - 1)
        metadata = {'per_view_transforms': transforms, 'model_input_shape': input_shape,
                    'patch_start_index': patch_start}
        result = {}
        for layer in self.layers:
            tokens = outputs[layer]
            if tokens is None or tokens.shape[2] - patch_start != rows * cols:
                raise RuntimeError(f'Layer {layer} missing or inconsistent with grid {grid}')
            result[layer] = (numpy_tokens(tokens[0]).reshape(-1, tokens.shape[-1]), positions, types, grid, metadata)
        return [result]


class Vjepa21:
    layers = (None,)
    max_batch = None

    def __init__(self, args, path_map):
        self.native = native = native_module('vjepa21', 'extract')
        root = path_map(WORKSTATION_ROOT)
        self.device, self.compute_dtype = args.device, args.compute_dtype
        self.model, checkpoint, provenance = native.load_encoder(root, args.device)
        config = (root / 'checkpoints/vjepa21/source' / SSV2_EVAL_CONFIG).read_text()
        for key, value in SSV2_EVAL.items():
            if f'{key}: {value}\n' not in config:
                raise RuntimeError(f'Pinned {SSV2_EVAL_CONFIG} no longer states {key}: {value}')
        from evals.video_classification_frozen.utils import EvalVideoTransform
        from src.datasets.utils.video.functional import get_resize_sizes
        self.get_resize_sizes = get_resize_sizes
        self.views = SSV2_EVAL['num_views_per_segment']
        self.view = self.views // 2
        self.side = SSV2_EVAL['resolution']
        self.transform = EvalVideoTransform(num_views_per_clip=self.views, short_side_size=self.side,
                                            normalize=IMAGENET_NORMALIZATION)
        self.sampler = {'frames_per_clip': args.video_frames, 'frame_step': args.video_frame_step,
                        'num_segments': args.video_segments, 'allow_segment_overlap': True,
                        'segment': args.video_segment}
        self.grid = (args.video_frames // 2, self.side // 16, self.side // 16)
        self.positions, self.types = dense_layout(self.grid)
        import cv2
        self.provenance = {
            'model_id': native.MODEL_ID, 'source_revision': native.REVISION,
            'checkpoint_key': 'ema_encoder', 'checkpoint_sha256': provenance['encoder_sha256'],
            'source_checkpoint_sha256': provenance['source_sha256'], 'layer': native.LAYER,
            'tensor_boundary': native.TENSOR_BOUNDARY, 'native_normalization': 'checkpoint-native final LayerNorm only',
            'flatten_order': native.FLATTEN_ORDER, 'tubelet_size': 2, 'patch_size': 16,
            'temporal_sampling': {
                'implementation': 'src/datasets/video_dataset.py VideoDataset.loadvideo_decord arithmetic, random_clip_sampling=False',
                'eval_config': SSV2_EVAL_CONFIG, 'eval_config_values': SSV2_EVAL, **self.sampler,
                'upstream_eval_difference': 'upstream eval loader keeps random_clip_sampling=True, which only randomizes the window end when frames // num_segments > frames_per_clip * frame_step; this cache fixes end=clip_len; one segment instead of averaging all segments',
                'decoder': 'PyAV full presentation-order decode for frame count; upstream uses decord len(VideoReader)'},
            'spatial_view': {'implementation': 'evals/video_classification_frozen/utils.py EvalVideoTransform',
                             'short_side': self.side, 'resize': 'src video_transforms.Resize bilinear (cv2.INTER_LINEAR)',
                             'num_views_per_segment': self.views, 'view_index': self.view,
                             'view': 'center crop along the long side (start = view_index * ((long - short_side) // (views - 1)))',
                             'scale': 'uint8 / 255', 'mean': IMAGENET_NORMALIZATION[0], 'std': IMAGENET_NORMALIZATION[1]},
            'compute_policy': 'float32 weights; torch.autocast to compute dtype when not float32',
            'feature_dim': 768, 'grid': list(self.grid),
            'token_layout': 'patches only (tubelet,row,col); encoder has no CLS/register tokens',
            'versions': {**module_versions('torchvision', 'PIL', 'av', 'timm', 'safetensors'), 'cv2': cv2.__version__}}

    def prepare(self, record):
        if len(record.paths) != 1:
            raise ValueError(f'{record.sample_id}: video samples need exactly one clip')
        path = record.paths[0]
        count, fps = self.native.count_frames(path)
        indices, branch = video_eval_clip_indices(count, **self.sampler)
        frames, timestamps = self.native.decode_frames(path, indices)
        clip = self.transform(frames)[self.view]
        height, width = frames.shape[1:3]
        resized = self.get_resize_sizes(height, width, self.side)
        offset = self.view * ((max(resized) - self.side) // (self.views - 1))
        return clip, {'decoded_frames': count, 'average_fps': fps, 'decoded_rgb_shape': list(frames.shape[1:]),
                      'temporal_branch': branch, 'selected_frame_indices': indices.tolist(),
                      'selected_timestamps_seconds': timestamps, 'resized_hw': list(resized),
                      'crop_axis': 'height' if height > width else 'width', 'crop_start': offset}

    def forward(self, payloads):
        import torch
        clips = torch.stack([clip for clip, _ in payloads]).to(self.device)
        with torch.inference_mode(), autocast(self.device, self.compute_dtype):
            tokens = self.model(clips)
        expected = (len(payloads), len(self.types), 768)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(f'Unexpected native dense-token shape {tuple(tokens.shape)}, expected {expected}')
        tokens = numpy_tokens(tokens)
        return [{None: (tokens[i], self.positions, self.types, self.grid, payloads[i][1])}
                for i in range(len(payloads))]


ADAPTERS = {'dinov3': Dinov3, 'qwen38': Qwen38, 'vggt': Vggt, 'vjepa21': Vjepa21}


def prefetched_batches(adapter, records, batch_size, workers, depth=2):
    """Yield (records, prepared payloads) while up to `depth` later batches preprocess in threads."""
    batches = iter([records[i:i + batch_size] for i in range(0, len(records), batch_size)])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue = collections.deque()

        def submit():
            batch = next(batches, None)
            if batch is not None:
                queue.append((batch, [pool.submit(adapter.prepare, record) for record in batch]))

        for _ in range(depth):
            submit()
        while queue:
            batch, futures = queue.popleft()
            submit()
            yield batch, [future.result() for future in futures]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--manifest', type=Path, action='append', required=True,
                        help='JSONL with sample_id, split (train/validation, alias val), paths; repeatable')
    parser.add_argument('--root', type=Path, default=Path(WORKSTATION_ROOT),
                        help=f'Shared project root; when it differs, {WORKSTATION_ROOT} paths are mapped to it')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW',
                        help='Extra explicit absolute prefix mapping for manifest/checkpoint paths; repeatable')
    parser.add_argument('--output-root', type=Path, help='Cache root (default <mapped project root>/features/cache)')
    parser.add_argument('--checkpoint', type=Path, help='Override the mapped native checkpoint path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compute-dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4, help='Preprocessing threads')
    parser.add_argument('--threads', type=int, default=8, help='torch intra-op CPU threads')
    parser.add_argument('--shard-tokens', type=int, default=65536)
    parser.add_argument('--resume', action='store_true', help='Continue a matching-identity cache')
    parser.add_argument('--limit-per-split', type=int, help='Smoke subset: first N records of each split')
    parser.add_argument('--log-every', type=int, default=50, help='Progress line every N batches')
    parser.add_argument('--layers', default='heads', help='vggt: heads (4,11,17,23), last, or indices')
    parser.add_argument('--video-frames', type=int, default=SSV2_EVAL['frames_per_clip'])
    parser.add_argument('--video-frame-step', type=int, default=SSV2_EVAL['frame_step'])
    parser.add_argument('--video-segments', type=int, default=SSV2_EVAL['num_segments'])
    parser.add_argument('--video-segment', type=int, default=0, help='Retained temporal segment index')
    args = parser.parse_args(argv)
    positive = ('batch_size', 'workers', 'threads', 'shard_tokens', 'log_every', 'video_frames',
                'video_frame_step', 'video_segments')
    if any(getattr(args, name) < 1 for name in positive) or (args.limit_per_split is not None and args.limit_per_split < 1):
        parser.error('Counts, sizes and video sampler parameters must be positive')
    if args.video_frames % 2 or not 0 <= args.video_segment < args.video_segments:
        parser.error('Video frames must be even (tubelet 2) and --video-segment < --video-segments')
    if args.device.startswith('cpu') and args.compute_dtype != 'float32':
        parser.error('CPU extraction is float32; bfloat16 is for CUDA')
    if args.model == 'vggt' and args.batch_size != 1:
        parser.error('VGGT caches one ordered view group per forward (--batch-size 1)')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from compression.cache import FeatureCacheWriter, cache_identity

    torch.set_num_threads(args.threads)
    path_map = build_path_map(args.root, args.path_map)
    output_root = args.output_root or path_map(WORKSTATION_ROOT) / 'features/cache'
    records, manifests = load_manifests(args.manifest, path_map, args.limit_per_split)
    adapter = ADAPTERS[args.model](args, path_map)
    expected = [{'sample_id': record.sample_id, 'split': record.split, 'layer': layer}
                for record in records for layer in adapter.layers]
    provenance = json_safe({'schema': SCHEMA, 'model': args.model, **adapter.provenance,
                            'compute_dtype': args.compute_dtype, 'device_type': torch.device(args.device).type,
                            'storage_dtype': 'float16', 'token_types': TOKEN_TYPES, 'manifests': manifests,
                            'selection': {'order': 'manifest order', 'limit_per_split': args.limit_per_split,
                                          'split_aliases': SPLIT_ALIASES}})
    identity = cache_identity(provenance, expected)
    output = output_root / args.model / identity
    started, written = time.monotonic(), 0
    with FeatureCacheWriter(output, provenance, expected, shard_tokens=args.shard_tokens,
                            resume=args.resume) as writer:
        missing = {(item['sample_id'], item.get('layer')) for item in writer.missing()}
        pending = [record for record in records
                   if any((record.sample_id, layer) in missing for layer in adapter.layers)]
        print(json.dumps({'cache': str(output), 'identity': identity, 'records': len(records),
                          'expected': len(expected), 'pending_records': len(pending)}), flush=True)
        batches = prefetched_batches(adapter, pending, args.batch_size, args.workers)
        for step, (batch, payloads) in enumerate(batches, 1):
            for record, per_layer in zip(batch, adapter.forward(payloads)):
                for layer, (features, positions, types, grid, metadata) in per_layer.items():
                    if (record.sample_id, layer) not in missing:
                        continue
                    writer.add(sample_id=record.sample_id, split=record.split, features=features,
                               positions=positions, token_types=types, grid=grid, layer=layer,
                               metadata=json_safe({**metadata, 'manifest_record': record.source}))
                    written += 1
            if step % args.log_every == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({'written': written, 'batches': step,
                                  'records_per_second': round(step * args.batch_size / elapsed, 3)}), flush=True)
        path = writer.finish()
    print(json.dumps({'cache': str(path), 'identity': identity, 'written': written,
                      'seconds': round(time.monotonic() - started, 1)}), flush=True)


if __name__ == '__main__':
    main()
