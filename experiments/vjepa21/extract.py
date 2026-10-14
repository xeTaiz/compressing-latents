#!/usr/bin/env python3
"""Extract frozen V-JEPA 2.1 base384 dense tokens, one clip artifact at a time (diagnostic native export).

Training caches use `python -m compression.extract --model vjepa21`, which reuses the encoder loader
and decoder defined here with the pinned SSv2 evaluation sampler and spatial view.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import torch
from safetensors.torch import load_file

from prepare import ROOT, REVISION, MODEL_ID

LAYER = 11
TENSOR_BOUNDARY = 'VisionTransformer.forward(training=False): after norms_block[-1] (checkpoint-native LayerNorm eps=1e-6)'
FLATTEN_ORDER = 'time/tubelet major, row then col; same as Conv3D flatten(2).transpose(1,2)'


def records(args):
    if args.clip:
        yield {'sample_id': args.sample_id or args.clip.stem, 'split': args.split or 'explicit',
               'paths': [str(args.clip.resolve())], 'dataset': 'explicit-user-clip'}
        return
    with args.manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if args.split is not None and record['split'] != args.split:
                continue
            if int(hashlib.sha256(record['sample_id'].encode()).hexdigest(), 16) % args.num_shards == args.shard_index:
                yield record


def count_frames(path):
    """Decode every presentation-order frame once, retaining none; returns (count, average FPS)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        count = sum(1 for _ in container.decode(stream))
    if count == 0:
        raise RuntimeError(f'No decoded frames in {path}')
    return count, fps


def decode_frames(path, indices):
    """Second decode pass retaining only the requested (possibly repeated) frame indices."""
    indices = [int(i) for i in indices]
    wanted = set(indices)
    selected, timestamps = {}, {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                selected[i] = frame.to_ndarray(format='rgb24')
                timestamps[i] = float(frame.time) if frame.time is not None else None
            if i == max(indices):
                break
    if len(selected) != len(wanted):
        raise RuntimeError(f'Decode frame count changed between passes: {path}')
    buffer = [selected[i] for i in indices]
    shape = buffer[0].shape
    if any(frame.shape != shape for frame in buffer):
        raise RuntimeError(f'Variable frame geometry is unsupported: {path}')
    return np.stack(buffer), [timestamps[i] for i in indices]


def sample_clip(path, frames):
    """Legacy diagnostic policy: endpoint-inclusive uniform sampling over the whole decoded clip."""
    count, fps = count_frames(path)
    indices = np.rint(np.linspace(0, count - 1, frames)).astype(np.int64)
    buffer, timestamps = decode_frames(path, indices)
    return list(buffer), {'sampling': 'round(linspace(0, decoded_frames-1, frames)); full-clip uniform endpoints; repeats if short',
                          'decoded_frames': count, 'selected_frame_indices': indices.tolist(),
                          'selected_timestamps_seconds': timestamps,
                          'average_fps': fps, 'decoded_rgb_shape': list(buffer.shape[1:])}


def load_encoder(root, device):
    """Construct the Hub-identical base384 encoder from pinned source and strictly load the EMA weights."""
    source = root / 'checkpoints/vjepa21/source'
    # The shared mount's owner differs on workers; trust only this exact checkout for the read-only check.
    actual = subprocess.check_output(['git', '-c', f'safe.directory={source}', '-C', str(source),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f'Wrong encoder source revision: {actual}')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from app.vjepa_2_1.models.vision_transformer import vit_base
    # Identical to the encoder construction in vjepa2_1_vit_base_384, without
    # creating/downloading a predictor. Keep Hub-native num_frames=64; forward
    # computes actual temporal grid from its input and RoPE supports 16 frames.
    model = vit_base(patch_size=16, img_size=(384, 384), num_frames=64,
                     tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                     uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                     interpolate_rope=True)
    checkpoint = root / 'checkpoints/vjepa21/ema_encoder.safetensors'
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.eval().requires_grad_(False).to(device)
    provenance = json.loads((checkpoint.parent / 'provenance.json').read_text())
    return model, checkpoint, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--clip', type=Path)
    inputs.add_argument('--manifest', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'features/vjepa21')
    p.add_argument('--device', default='cpu')
    p.add_argument('--frames', type=int, choices=(16, 64), default=16)
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16', help='Storage dtype only; encoder computes float32')
    p.add_argument('--split', help='Manifest filter; or explicit clip source split')
    p.add_argument('--sample-id')
    p.add_argument('--max-samples', type=int, default=16, help='Bounded extraction; each invocation/shard writes at most this many clips')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--offset', type=int, default=0, help='Skip this many selected records within shard before extraction')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.max_samples < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.offset < 0 or args.threads < 1:
        p.error('Invalid sample, shard, offset or thread bound')
    torch.set_num_threads(args.threads)
    model, checkpoint, provenance = load_encoder(args.root, args.device)
    from evals.hub.preprocessor import vjepa2_preprocessor
    processor = vjepa2_preprocessor(crop_size=384)
    args.output.mkdir(parents=True, exist_ok=True)
    grid = (args.frames // 2, 24, 24)
    positions = np.stack(np.meshgrid(*(np.arange(n) for n in grid), indexing='ij'), axis=-1).reshape(-1, 3).astype(np.int32)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'av', 'numpy', 'Pillow', 'safetensors')}
    written = 0
    for i, record in enumerate(records(args)):
        if i < args.offset:
            continue
        if len(record['paths']) != 1:
            raise ValueError('A video sample must contain exactly one clip path')
        path = Path(record['paths'][0])
        if not path.is_absolute():
            path = args.manifest.parent / path
        buffer, sampling = sample_clip(path, args.frames)
        clip = processor(buffer)[0].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            result = model(clip)
        expected = (1, int(np.prod(grid)), 768)
        if tuple(result.shape) != expected:
            raise RuntimeError(f'Unexpected native dense-token shape {tuple(result.shape)}, expected {expected}')
        features = result[0].cpu().numpy().astype(args.dtype)
        artifact_id = hashlib.sha256(record['sample_id'].encode()).hexdigest()[:24]
        feature_path = args.output / f'{artifact_id}.npz'
        metadata_path = args.output / f'{artifact_id}.json'
        if feature_path.exists() or metadata_path.exists():
            raise FileExistsError(f'Refusing to overwrite existing extraction: {artifact_id}; use a fresh output directory')
        np.savez(feature_path, features=features, positions=positions,
                 special_tokens=np.empty((0, 768), dtype=args.dtype))
        metadata = {'source': record, 'resolved_clip_path': str(path.resolve()),
                    'model_id': MODEL_ID, 'revision': REVISION, 'checkpoint': str(checkpoint),
                    'checkpoint_key': 'ema_encoder', 'checkpoint_sha256': provenance['encoder_sha256'],
                    'source_checkpoint_sha256': provenance['source_sha256'],
                    'layer': 'block 11 (12th/final transformer block)',
                    'tensor_boundary': TENSOR_BOUNDARY,
                    'extra_normalization': None, 'grid': list(grid), 'features_shape': list(features.shape),
                    'positions_shape': list(positions.shape), 'positions_axes': ['tubelet', 'row', 'col'],
                    'flatten_order': FLATTEN_ORDER,
                    'special_tokens_shape': [0, 768], 'tubelet_size': 2, 'patch_size': 16,
                    'input_shape': list(clip.shape), 'storage_dtype': args.dtype, 'compute_dtype': 'float32',
                    'pilot_deviation': '16 frames instead of Hub default 64' if args.frames == 16 else None,
                    'preprocessing': {'implementation': 'pinned evals.hub.preprocessor.vjepa2_preprocessor(crop_size=384)',
                                      'rgb': True, 'resize_short_side': 438, 'resize_interpolation': 'bilinear',
                                      'center_crop': [384, 384], 'scale': 'uint8 / 255',
                                      'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                      'random_augmentation': False},
                    'temporal_sampling': sampling, 'versions': versions,
                    'manifest': str(args.manifest.resolve()) if args.manifest else None,
                    'shard_index': args.shard_index, 'num_shards': args.num_shards,
                    'artifact': str(feature_path)}
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'sample_id': record['sample_id'], 'features': str(feature_path), 'shape': list(features.shape)}), flush=True)
        written += 1
        if written == args.max_samples:
            break
    if written == 0:
        raise RuntimeError('No clips matched the manifest/split/shard/offset selection')


if __name__ == '__main__':
    main()
