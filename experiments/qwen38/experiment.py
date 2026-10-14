#!/usr/bin/env python3
"""Prepare exact Qwen3.8 visual weights and stream frozen pre-merger image tokens."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

ROOT = Path('/home/engeld/mnt/datawaha/compressing-latents')
CHECKPOINT = ROOT / 'checkpoints/qwen38'
MODEL_ID = 'timm/qwen3_vit_416m.qwen3_8_27b'
MODEL_REVISION = '290f775dc6c2a7844b065855867ded2aa55a6a7c'
SOURCE_ID = 'Qwen/Qwen3.8-27B'
SOURCE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
TIMM_REVISION = '12c4939be518a6d02d5d3b47b0b2bed49ad32312'
WEIGHT_BYTES = 1660060752
WEIGHT_SHA256 = 'eda956958255e255333af183fb2c78b7501cc3ddbcf42dc6a2f8f0681de50a24'
SHARED_DATA = ROOT / 'data/images/imagenet1k-pilot'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_records(path, split=None):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get('sample_id'):
                raise ValueError(f'{path}:{line_number}: missing sample_id')
            if split is not None and record.get('split') != split:
                continue
            paths = record.get('paths')
            if not isinstance(paths, list) or len(paths) != 1:
                raise ValueError(f'{path}:{line_number}: image-only input requires exactly one path')
            image = Path(paths[0])
            if not image.is_absolute() or not image.is_file():
                raise FileNotFoundError(f'{path}:{line_number}: unavailable absolute image path {image}')
            yield record


def dataset_inventory(manifests):
    inventory = []
    for path in manifests:
        if not path.is_file():
            inventory.append({'manifest': str(path), 'available': False})
            continue
        splits = {}
        for record in manifest_records(path):
            split = record.get('split', 'unspecified')
            splits[split] = splits.get(split, 0) + 1
        inventory.append({'manifest': str(path.resolve()), 'available': True,
                          'sha256': sha256(path), 'counts_by_split': splits})
    return inventory


def prepare(args):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    checkpoint = args.checkpoint_dir
    visual = checkpoint / 'visual'
    visual.mkdir(parents=True, exist_ok=True)
    weights = visual / 'model.safetensors'
    free = shutil.disk_usage(checkpoint).free
    needed = max(0, WEIGHT_BYTES - (weights.stat().st_size if weights.exists() else 0))
    if free < needed + 256 * 1024 * 1024:
        raise RuntimeError(f'Insufficient checkpoint storage: {free} bytes free, {needed} needed plus headroom')
    for filename in ('config.json', 'conversion.json', 'LICENSE', 'README.md'):
        hf_hub_download(MODEL_ID, filename, revision=MODEL_REVISION, local_dir=visual)
    if not weights.is_file() or weights.stat().st_size != WEIGHT_BYTES:
        hf_hub_download(MODEL_ID, 'model.safetensors', revision=MODEL_REVISION, local_dir=visual)
    actual_hash = sha256(weights)
    if actual_hash != WEIGHT_SHA256:
        raise RuntimeError(f'Checkpoint integrity mismatch: {weights}; expected {WEIGHT_SHA256}, got {actual_hash}')
    upstream = checkpoint / 'upstream'
    for filename in ('config.json', 'preprocessor_config.json', 'LICENSE'):
        hf_hub_download(SOURCE_ID, filename, revision=SOURCE_REVISION, local_dir=upstream)
    config = json.loads((visual / 'config.json').read_text())
    conversion = json.loads((visual / 'conversion.json').read_text())
    original = json.loads((upstream / 'config.json').read_text())
    vision = original['vision_config']
    expected = {'depth': 27, 'hidden_size': 1152, 'patch_size': 16,
                'temporal_patch_size': 2, 'spatial_merge_size': 2,
                'out_hidden_size': 5120, 'deepstack_visual_indexes': []}
    if original['model_type'] != 'qwen3_5' or any(vision.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Official source configuration differs from the requested Qwen3.8 visual tower')
    if conversion['source_repo'] != SOURCE_ID or conversion['source_revision'] != SOURCE_REVISION:
        raise RuntimeError('Visual conversion does not identify the exact requested source')
    if config['architecture'] != 'qwen3_vit_416m' or config['model_args'].get('encoder_pool') != '':
        raise RuntimeError('Expected the pre-merger Qwen visual checkpoint, not a merger or classifier target')
    with safe_open(weights, framework='pt', device='cpu') as handle:
        shapes = {key: handle.get_slice(key).get_shape() for key in handle.keys()}
    if shapes.get('encoder.patch_embed.proj.weight') != [1152, 3, 16, 16]:
        raise RuntimeError('Unexpected folded image patch-embedding shape')
    if shapes.get('encoder.blocks.26.attn.qkv.weight') != [3456, 1152]:
        raise RuntimeError('Unexpected final vision transformer block')
    if any('merger' in key or 'language' in key for key in shapes):
        raise RuntimeError('Pre-merger checkpoint unexpectedly includes merger or language tensors')
    provenance = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                  'source_id': SOURCE_ID, 'source_revision': SOURCE_REVISION,
                  'timm_revision': TIMM_REVISION, 'checkpoint_bytes': weights.stat().st_size,
                  'checkpoint_sha256': actual_hash, 'license': 'Apache-2.0',
                  'free_bytes_before_prepare': free, 'conversion': conversion,
                  'vision_config': vision, 'tensor_shapes': shapes}
    save_json(checkpoint / 'provenance.json', provenance)
    manifests = args.manifest or [SHARED_DATA / 'train.jsonl', SHARED_DATA / 'validation.jsonl']
    inventory = dataset_inventory(manifests)
    save_json(checkpoint / 'dataset-inventory.json', inventory)
    print(json.dumps({'checkpoint': str(weights), 'bytes': weights.stat().st_size,
                      'sha256': actual_hash, 'datasets': inventory}, indent=2))


def selected_records(args):
    if args.image:
        if args.split is not None:
            raise ValueError('--split filters manifests only; explicit images use split=explicit')
        for image in args.image:
            image = image.resolve(strict=True)
            yield {'sample_id': 'explicit/' + hashlib.sha256(str(image).encode()).hexdigest()[:20],
                   'split': 'explicit', 'paths': [str(image)]}
    else:
        yield from manifest_records(args.manifest, args.split)


LAYER = 26
TENSOR_BOUNDARY = 'model.forward_features / encoder.blocks[26] residual output; before merger.norm'
OUTPUT_NORMALIZATION = 'none on output tokens; native block pre-norm retained; no encoder final norm'
PREPROCESSING_POLICY = {'decode': 'Pillow RGB; no EXIF transpose', 'rescale': 'uint8 / 255',
                        'resize': 'bicubic fixed square squash, no crop; fixed-resolution experiment policy'}
TEMPORAL_IMAGE_HANDLING = {'temporal_patch_size': 2, 'logical_temporal_groups': 1,
                           'original_image_frames': 'same image repeated twice',
                           'conversion': 'temporal Conv3d kernel summed in float64 into Conv2d by official timm conversion; image-only, not a video encoder'}


def load_model(checkpoint, device):
    """Strictly load the prepared pre-merger visual tower in float32 and move it once."""
    import timm
    from safetensors.torch import load_file
    provenance = json.loads((checkpoint / 'provenance.json').read_text())
    if provenance['model_id'] != MODEL_ID or provenance['model_revision'] != MODEL_REVISION:
        raise RuntimeError('Run prepare for the exact Qwen3.8 vision-only checkpoint first')
    weights = checkpoint / 'visual/model.safetensors'
    if weights.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError('Checkpoint size changed; run prepare to verify integrity')
    config = json.loads((checkpoint / 'visual/config.json').read_text())
    model = timm.create_model(config['architecture'], pretrained=False, num_classes=0,
                              pretrained_cfg=config['pretrained_cfg'], **config['model_args'])
    model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, provenance


def make_transform(model, size):
    import timm
    transform_config = timm.data.resolve_model_data_config(model)
    transform_config.update(input_size=(3, size, size), crop_pct=1.0, crop_mode='squash')
    return timm.data.create_transform(**transform_config, is_training=False), transform_config


def load_image(path, transform):
    """Pillow RGB decode (no EXIF transpose) through the fixed timm squash transform."""
    from PIL import Image
    with Image.open(path) as image:
        original_size = list(image.size)
        pixels = transform(image.convert('RGB'))
    return pixels, original_size


def extract(args):
    import numpy as np
    import torch

    if args.limit < 1 or args.size < 32 or args.size % 32:
        raise ValueError('--limit must be positive and --size must be a positive multiple of 32')
    device = torch.device(args.device)
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    model, provenance = load_model(args.checkpoint_dir, device)
    transform, transform_config = make_transform(model, args.size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    index = args.output_dir / 'index.jsonl'
    if index.exists():
        raise FileExistsError(f'Output already contains an index: {index}; choose a new output directory')
    with index.open('x') as listing, torch.inference_mode():
        for record in selected_records(args):
            image_path = Path(record['paths'][0])
            key = hashlib.sha256((str(record.get('split')) + '\0' + str(record['sample_id'])).encode()).hexdigest()[:24]
            stem = args.output_dir / key
            feature_path = stem.with_suffix('.npz')
            metadata_path = stem.with_suffix('.json')
            if feature_path.exists() or metadata_path.exists():
                raise FileExistsError(f'Refusing to overwrite existing sample outputs: {stem}')
            x, original_size = load_image(image_path, transform)
            x = x.unsqueeze(0).to(device)
            # Crucial: bypass classifier forward()/forward_head(), pooling, and affine-free LayerNorm.
            raw = model.forward_features(x)
            grid = args.size // 16
            if tuple(raw.shape) != (1, grid, grid, 1152):
                raise RuntimeError(f'Expected raw NHWC pre-merger tokens, got {tuple(raw.shape)}')
            features = raw[0].reshape(grid * grid, 1152).to(
                torch.float16 if args.save_dtype == 'float16' else torch.float32).cpu().numpy()
            rows, cols = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
            positions = np.stack((np.zeros_like(rows), rows, cols), axis=-1).reshape(-1, 3).astype(np.int32)
            np.savez(feature_path, features=features, positions=positions,
                     special_tokens=np.empty((0, 1152), dtype=features.dtype))
            metadata = {
                'sample': record, 'source_manifest': str(args.manifest.resolve()) if args.manifest else None,
                'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
                'source_id': SOURCE_ID, 'source_revision': SOURCE_REVISION,
                'checkpoint_sha256': provenance['checkpoint_sha256'], 'timm_revision': TIMM_REVISION,
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'pillow', 'numpy')},
                'layer': LAYER, 'layer_number_one_based': LAYER + 1,
                'tensor_boundary': TENSOR_BOUNDARY,
                'normalization': OUTPUT_NORMALIZATION,
                'preprocessing': {**transform_config, 'original_size_wh': original_size, **PREPROCESSING_POLICY},
                'native_resolution_note': 'Official multimodal processor uses dynamic aspect-preserving smart_resize; this experiment intentionally fixes a square input, following the timm squash policy',
                'temporal_image_handling': TEMPORAL_IMAGE_HANDLING,
                'grid_thw': [1, grid, grid], 'features_shape': list(features.shape),
                'features_dtype': args.save_dtype, 'compute_dtype': 'float32',
                'positions_shape': list(positions.shape), 'positions_columns': ['temporal_group', 'row', 'col'],
                'token_order': 'spatial row-major; differs from upstream spatial-merge-grouped sequence ordering only by permutation',
                'special_tokens': {'count': 0, 'reason': 'visual tower has no CLS/register tokens; text image delimiters are not vision tokens'},
                'post_merger': 'not extracted; native merger normalizes each patch, concatenates 2x2 neighborhoods and projects to 5120 (12x12 tokens at 384)',
            }
            save_json(metadata_path, metadata)
            listing.write(json.dumps({'sample_id': record['sample_id'], 'split': record.get('split'),
                                      'features': str(feature_path.resolve()), 'metadata': str(metadata_path.resolve())}) + '\n')
            listing.flush()
            count += 1
            if count >= args.limit:
                break
    if count == 0:
        raise ValueError('No images matched the input and split selection')
    print(json.dumps({'samples': count, 'index': str(index.resolve()), 'features_per_sample': [grid * grid, 1152]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Download exact visual-only assets, verify SHA256, and inventory shared images')
    prep.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT)
    prep.add_argument('--manifest', type=Path, action='append', help='Inventory existing shared manifest; never downloads images')
    prep.set_defaults(run=prepare)
    ext = commands.add_parser('extract', help='Extract frozen raw final-block image patch tokens, one NPZ per image')
    ext.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT)
    source = ext.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path, action='append', help='Explicit image; repeat for a small image list')
    source.add_argument('--manifest', type=Path, help='Shared JSONL image-path manifest')
    ext.add_argument('--split', help='Exact image-level manifest split filter; never repartitions')
    ext.add_argument('--limit', type=int, default=1)
    ext.add_argument('--size', type=int, default=384)
    ext.add_argument('--device', default='cpu')
    ext.add_argument('--threads', type=int, default=4)
    ext.add_argument('--save-dtype', choices=('float16', 'float32'), default='float16')
    ext.add_argument('--output-dir', type=Path, default=ROOT / 'features/qwen38/smoke384')
    ext.set_defaults(run=extract)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
