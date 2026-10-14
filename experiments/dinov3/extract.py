"""Stream frozen final-layer DINOv3 patch features into bounded NPZ shards (diagnostic native export).

Training caches use `python -m compression.extract --model dinov3`, which reuses the loader and
preprocessing defined here.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

from prepare import DEFAULT_CHECKPOINT, MODEL_ID, REVISION, WEIGHT_NAME, native_factory

SIZE = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_OUTPUT = Path('/home/engeld/mnt/datawaha/compressing-latents/features/dinov3')
LAYER = 39
TENSOR_BOUNDARY = 'native DinoVisionTransformer.forward_features.x_norm_patchtokens after learned final model.norm LayerNorm'
SPECIAL_TOKEN_BOUNDARY = 'native forward_features.x_norm_clstoken and x_storage_tokens after model.norm in eval mode; local_cls_norm is training-only'
PREPROCESSING = {'exif': 'PIL.ImageOps.exif_transpose', 'color': 'RGB',
                 'resize_hw': [SIZE, SIZE], 'resize': 'torchvision.transforms.v2.Resize',
                 'interpolation': 'bilinear', 'antialias': True, 'aspect_ratio': 'direct square resize',
                 'crop': None, 'pixel_scale': 'uint8/255 -> float32',
                 'mean': MEAN, 'std': STD}


def load_model(checkpoint, device, compute_dtype):
    """Meta-construct the native factory, strictly assign mmap weights, then cast/move once."""
    asset = json.loads((checkpoint / 'asset.json').read_text())
    if not asset.get('available') or asset['model_id'] != MODEL_ID or asset['revision'] != REVISION:
        raise ValueError('The exact approved DINOv3 checkpoint is not available; run prepare.py')
    weight = checkpoint / WEIGHT_NAME
    if weight.stat().st_size != asset['checkpoint']['bytes']:
        raise ValueError('Checkpoint size differs from inspected asset')
    factory = native_factory(checkpoint / f'dinov3-{REVISION}')
    with torch.device('meta'):
        model = factory(pretrained=False)
    state = torch.load(weight, weights_only=True, mmap=True, map_location='cpu')
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.eval().requires_grad_(False).to(device=device, dtype=getattr(torch, compute_dtype))
    if (model.patch_size != 16 or model.embed_dim != 4096 or model.n_blocks != LAYER + 1
            or model.n_storage_tokens != 4):
        raise ValueError('Expected the inspected native DINOv3 ViT-7B/16 architecture')
    return model, asset


def make_transform():
    return v2.Compose([v2.ToImage(), v2.Resize((SIZE, SIZE), antialias=True),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=MEAN, std=STD)])


def load_image(path, transform):
    """Return float32 [3,SIZE,SIZE] pixels plus original/EXIF-oriented sizes."""
    with Image.open(path) as image:
        original_wh = list(image.size)
        image = ImageOps.exif_transpose(image).convert('RGB')
        oriented_wh = list(image.size)
        pixels = transform(image)
    return pixels, {'original_size_wh': original_wh, 'oriented_size_wh': oriented_wh,
                    'preprocessed_size_hw': [SIZE, SIZE]}


def samples(args):
    if args.image:
        for path in args.image:
            path = path.resolve()
            yield {'sample_id': 'explicit/' + hashlib.sha256(str(path).encode()).hexdigest()[:20],
                   'split': 'explicit', 'paths': [str(path)]}
    else:
        with args.manifest.open() as manifest:
            for line in manifest:
                if line.strip():
                    yield json.loads(line)


def write_shard(output, index, dense, cls, registers, records, common, dtype):
    output.mkdir(parents=True, exist_ok=True)
    name = f'shard-{index:06d}'
    npz_path, metadata_path = output / f'{name}.npz', output / f'{name}.json'
    if npz_path.exists() or metadata_path.exists():
        raise FileExistsError(f'Refusing to overwrite an existing extraction: {name}')
    count, grid = len(records), common['token_grid_hw']
    row, col = np.meshgrid(np.arange(grid[0]), np.arange(grid[1]), indexing='ij')
    positions_one = np.stack([np.zeros(row.size, dtype=np.int32), row.ravel(), col.ravel()], axis=1)
    features = np.concatenate(dense, axis=0).astype(dtype, copy=False)
    special_cls = np.stack(cls).astype(dtype, copy=False)
    special_registers = np.stack(registers).astype(dtype, copy=False)
    offsets = np.arange(count + 1, dtype=np.int64) * positions_one.shape[0]
    np.savez(npz_path, features=features, positions=np.tile(positions_one, (count, 1)),
             sample_offsets=offsets, cls=special_cls, registers=special_registers)
    metadata = dict(common, samples=records, sample_offsets=offsets.tolist(),
                    arrays={'features': list(features.shape), 'positions': [int(offsets[-1]), 3],
                            'cls': list(special_cls.shape), 'registers': list(special_registers.shape)},
                    storage_dtype=str(np.dtype(dtype)), feature_file=npz_path.name)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'features': str(npz_path), 'metadata': str(metadata_path), 'samples': count}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path, action='append', help='Repeat for explicit image list')
    source.add_argument('--manifest', type=Path, help='JSONL with sample_id, split, paths:[absolute image]')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--shard-size', type=int, default=16)
    parser.add_argument('--device', default='cpu', help='CPU locally, or cuda on a Worker Harness worker')
    parser.add_argument('--storage-dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--compute-dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    if args.shard_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('shard-size and limit must be positive')
    if args.output.exists() and any(args.output.glob('shard-*')):
        raise FileExistsError('Use a new empty extraction output directory')
    model, asset = load_model(args.checkpoint, args.device, args.compute_dtype)
    compute_dtype = getattr(torch, args.compute_dtype)
    transform = make_transform()
    grid = SIZE // model.patch_size
    common = {'model_id': MODEL_ID, 'revision': REVISION, 'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': asset['checkpoint']['sha256'], 'layer': LAYER,
              'layer_indexing': 'zero-based final transformer block',
              'tensor_boundary': TENSOR_BOUNDARY, 'special_token_boundary': SPECIAL_TOKEN_BOUNDARY,
              'native_normalization': {'type': 'learned LayerNorm', 'eps': model.norm.eps,
                                       'extra_compressor_normalization': None},
              'token_order': 'CLS, register tokens, spatial patches in row-major order',
              'token_grid_hw': [grid, grid], 'dense_dimension': model.embed_dim,
              'num_register_tokens': model.n_storage_tokens,
              'positions_semantics': '[time_or_view=0,row,col] zero-based integer indices; offsets delimit images',
              'preprocessing': PREPROCESSING,
              'compute_dtype': args.compute_dtype, 'device': args.device,
              'versions': {package: importlib.metadata.version(package)
                           for package in ['torch', 'torchvision', 'numpy', 'Pillow']},
              'input_manifest': str(args.manifest.resolve()) if args.manifest else None}
    dense, cls, registers, records = [], [], [], []
    shard_index, consumed, shard_split = 0, 0, None
    for sample in samples(args):
        if args.limit is not None and consumed >= args.limit:
            break
        paths = sample['paths']
        if len(paths) != 1:
            raise ValueError('DINO image manifests must contain exactly one image per sample')
        split = sample['split']
        if records and split != shard_split:
            write_shard(args.output, shard_index, dense, cls, registers, records, common, args.storage_dtype)
            dense, cls, registers, records = [], [], [], []
            shard_index += 1
        shard_split = split
        pixels, sizes = load_image(paths[0], transform)
        pixels = pixels.unsqueeze(0).to(device=args.device, dtype=compute_dtype)
        with torch.inference_mode():
            tokens = model.forward_features(pixels)
        patches = tokens['x_norm_patchtokens'][0].float().cpu().numpy()
        if patches.shape != (grid * grid, model.embed_dim):
            raise ValueError(f'Unexpected output tensor shape {patches.shape}')
        dense.append(patches)
        cls.append(tokens['x_norm_clstoken'][0].float().cpu().numpy())
        registers.append(tokens['x_storage_tokens'][0].float().cpu().numpy())
        records.append(dict(sample, **sizes))
        consumed += 1
        if len(records) == args.shard_size:
            write_shard(args.output, shard_index, dense, cls, registers, records, common, args.storage_dtype)
            dense, cls, registers, records = [], [], [], []
            shard_index += 1
    if records:
        write_shard(args.output, shard_index, dense, cls, registers, records, common, args.storage_dtype)
    if not consumed:
        raise ValueError('Input contains no samples')


if __name__ == '__main__':
    main()
