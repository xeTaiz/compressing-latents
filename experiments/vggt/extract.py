"""Extract frozen VGGT-Omega frame/inter-frame tokens for ONE ordered view group (diagnostic native export).

Training caches use `python -m compression.extract --model vggt`, which reuses the loader,
preprocessing and forward pass defined here.
"""
import argparse
import contextlib
import json
from pathlib import Path

from assets import (CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA256, FEATURES,
                    MODEL_ID, MODEL_VARIANT, MODEL_REVISION, SOURCE_REVISION, verify_checkpoint)

HEAD_LAYERS = (4, 11, 17, 23)
PATCH_SIZE = 16
IMAGE_RESOLUTION = 512
CONTEXT = 'All views jointly contextualized through global and register attention; never cache as independent image features'
REFERENCE_CONVENTION = 'First ordered view uses first-frame camera/register embeddings; remaining views use other-frame embeddings. No geometric head run here.'
HEAD_REQUIRED_LAYERS = {'dense_depth': list(HEAD_LAYERS), 'camera_and_register_tokens': [23]}
BENCHMARK_CAVEAT = 'Original512 checkpoint may have upstream training-data contamination; not clean foundation-model benchmark evidence.'


def tensor_boundary(index):
    return (f'Aggregator outputs[{index}], concatenated frame then inter-frame residual-stream outputs; '
            'before all head normalization/projection')


def preprocessing_policy(mode):
    return {'implementation': 'vggt_omega.utils.load_fn.load_and_preprocess_images',
            'mode': mode, 'resize': 'PIL bicubic', 'image_resolution': IMAGE_RESOLUTION,
            'aspect_ratio_crop_range_h_over_w': [0.5, 2.0],
            'height_width_rounding': 'native balanced token-budget or max-size shape, NumPy round to multiples of 16',
            'rgba': 'alpha composite onto white, convert RGB',
            'exif_orientation': 'not transposed by official implementation',
            'padding_rgb_value': 1.0}


def parse_layers(spec):
    if spec == 'last':
        layers = [23]
    elif spec == 'heads':
        layers = list(HEAD_LAYERS)
    else:
        layers = sorted(set(int(value) for value in spec.split(',')))
    if not layers or any(index not in range(24) for index in layers):
        raise ValueError('Layer indices must be within 0..23')
    return layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--images', nargs='+', help='Ordered paths from the SAME scene')
    source.add_argument('--manifest', type=Path, help='JSONL scene-view groups')
    parser.add_argument('--sample-id', help='Required with --manifest')
    parser.add_argument('--scene-id', help='Required with explicit --images')
    parser.add_argument('--split', default='unspecified', choices=['train', 'validation', 'unspecified'])
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=FEATURES / 'smoke')
    parser.add_argument('--layers', default='last', help='last, heads, or comma-separated zero-based indices 0..23')
    parser.add_argument('--mode', choices=['balanced', 'max_size'], default='balanced')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['float32', 'bfloat16', 'float16'], default='float32')
    parser.add_argument('--storage-dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.images and not args.scene_id:
        parser.error('--scene-id is required with --images; do not mix scenes')
    if args.manifest and not args.sample_id:
        parser.error('--sample-id is required with --manifest')
    if args.device == 'cpu' and args.precision != 'float32':
        parser.error('CPU extraction uses float32; mixed precision is CUDA-only')
    if args.threads < 1:
        parser.error('--threads must be positive')
    try:
        args.layer_indices = parse_layers(args.layers)
    except ValueError as error:
        parser.error(str(error))
    return args


def load_group(args):
    if args.images:
        group = {'sample_id': args.scene_id + '-explicit', 'scene_id': args.scene_id,
                 'split': args.split, 'paths': args.images, 'dataset': 'explicit-authorized-scene'}
    else:
        group = None
        with args.manifest.open() as stream:
            for line in stream:
                candidate = json.loads(line)
                if candidate['sample_id'] == args.sample_id:
                    if group is not None:
                        raise ValueError('Duplicate sample_id in manifest')
                    group = candidate
        if group is None:
            raise ValueError(f'Unknown sample_id: {args.sample_id}')
    if not group.get('scene_id') or not group.get('paths'):
        raise ValueError('A scene identity and nonempty ordered paths are required')
    group['paths'] = [str(Path(path).resolve(strict=True)) for path in group['paths']]
    return group


def transform_metadata(paths, mode):
    from PIL import Image
    from vggt_omega.utils.load_fn import (_crop_to_supported_aspect_ratio,
                                          _balanced_target_shape, _max_size_target_shape)
    transforms = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            original_mode = image.mode
            cropped_width, cropped_height = _crop_to_supported_aspect_ratio(image).size
        left, top = (width - cropped_width) // 2, (height - cropped_height) // 2
        shape_fn = _balanced_target_shape if mode == 'balanced' else _max_size_target_shape
        resized_height, resized_width = shape_fn(cropped_height / cropped_width, IMAGE_RESOLUTION, PATCH_SIZE)
        transforms.append({'original_size_wh': [width, height], 'original_mode': original_mode,
                           'crop_box_original_xyxy': [left, top, left + cropped_width, top + cropped_height],
                           'cropped_size_wh': [cropped_width, cropped_height],
                           'resize_size_wh': [resized_width, resized_height],
                           'size_before_group_padding_wh': [resized_width, resized_height]})
    max_width = max(item['size_before_group_padding_wh'][0] for item in transforms)
    max_height = max(item['size_before_group_padding_wh'][1] for item in transforms)
    for item in transforms:
        width, height = item['size_before_group_padding_wh']
        dw, dh = max_width - width, max_height - height
        item['group_pad_ltrb_px'] = [dw // 2, dh // 2, dw - dw // 2, dh - dh // 2]
    return transforms


def load_aggregator(checkpoint, layer_indices, device):
    """Verify the exact Omega-512 digest, meta-construct the aggregator and strictly assign its weights."""
    verify_checkpoint(checkpoint)
    import torch
    from vggt_omega.models.aggregator import Aggregator, _RESNET_MEAN, _RESNET_STD
    # Meta initialization avoids allocating/randomly initializing a second 5GB model.
    with torch.device('meta'):
        aggregator = Aggregator(cached_layer_indices=tuple(layer_indices))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = {key.removeprefix('aggregator.'): value
             for key, value in state.items() if key.startswith('aggregator.')}
    aggregator.load_state_dict(state, strict=True, assign=True)
    del state
    # These two upstream buffers are deliberately nonpersistent in the checkpoint.
    aggregator._resnet_mean = torch.tensor(_RESNET_MEAN).view(1, 1, 3, 1, 1)
    aggregator._resnet_std = torch.tensor(_RESNET_STD).view(1, 1, 3, 1, 1)
    return aggregator.eval().requires_grad_(False).to(device)


def load_images(paths, mode):
    """Official Omega preprocessing for one ordered group: float32 [S,3,H,W] in [0,1]."""
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    return load_and_preprocess_images([str(path) for path in paths], mode=mode,
                                      image_resolution=IMAGE_RESOLUTION, patch_size=PATCH_SIZE)


def forward_group(aggregator, images, device, precision):
    """Run one group [S,3,H,W]; returns (per-layer outputs list, patch_start_idx, grid (S,Gh,Gw))."""
    import torch
    images = images.unsqueeze(0).to(device)
    dtype = getattr(torch, precision)
    autocast = (torch.autocast('cuda', dtype=dtype) if torch.device(device).type == 'cuda' and dtype != torch.float32
                else contextlib.nullcontext())
    with torch.inference_mode(), autocast:
        outputs, patch_start_idx = aggregator(images)
    grid = (images.shape[1], images.shape[-2] // PATCH_SIZE, images.shape[-1] // PATCH_SIZE)
    return outputs, patch_start_idx, grid, list(images.shape)


def main():
    args = parse_args()
    group = load_group(args)
    import numpy as np
    import torch
    from vggt_omega.models.aggregator import _RESNET_MEAN, _RESNET_STD

    torch.set_num_threads(args.threads)
    aggregator = load_aggregator(args.checkpoint, args.layer_indices, args.device)
    images = load_images(group['paths'], args.mode)
    outputs, patch_start_idx, (views, grid_height, grid_width), input_shape = forward_group(
        aggregator, images, args.device, args.precision)
    positions = np.stack(np.meshgrid(np.arange(views), np.arange(grid_height), np.arange(grid_width), indexing='ij'), axis=-1).reshape(-1, 3).astype(np.int32)
    transforms = transform_metadata(group['paths'], args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(np, args.storage_dtype)
    for index in args.layer_indices:
        tokens = outputs[index]
        if tokens is None:
            raise RuntimeError(f'Requested layer {index} was not cached')
        if tokens.shape[2] - patch_start_idx != grid_height * grid_width:
            raise RuntimeError('Patch token count disagrees with image transform/grid')
        def array(value):
            return value.detach().to(device='cpu', dtype=getattr(torch, args.storage_dtype)).numpy()
        features = array(tokens[0, :, patch_start_idx:]).reshape(-1, tokens.shape[-1])
        camera = array(tokens[0, :, 0])
        registers = array(tokens[0, :, 1:patch_start_idx])
        stem = args.output / f'layer_{index:02d}'
        np.savez(stem.with_suffix('.npz'), features=features.astype(dtype, copy=False), positions=positions,
                 camera_tokens=camera, register_tokens=registers)
        metadata = {'schema_version': 1, 'source_group': group,
                    'model_id': MODEL_ID, 'model_variant': MODEL_VARIANT, 'model_revision': MODEL_REVISION,
                    'source_revision': SOURCE_REVISION, 'checkpoint_path': str(args.checkpoint.resolve()),
                    'checkpoint_sha256': CHECKPOINT_SHA256, 'checkpoint_bytes': CHECKPOINT_BYTES,
                    'layer_index_zero_based': index,
                    'tensor_boundary': tensor_boundary(index),
                    'inter_frame_attention_type': aggregator.inter_frame_attention_types[index],
                    'normalization': {'input_rgb_range': [0, 1], 'internal_mean': _RESNET_MEAN,
                                      'internal_std': _RESNET_STD,
                                      'patch_embed_boundary': 'Omega DinoVisionTransformer ViT-L/16 x_norm_patchtokens (native LayerNorm) before alternating attention',
                                      'extra_feature_normalization': None},
                    'context': CONTEXT,
                    'reference_view_index': 0,
                    'reference_convention': REFERENCE_CONVENTION,
                    'positions_axes': ['view_index', 'patch_row', 'patch_col'],
                    'flatten_order': 'view-major, row-major, column-major',
                    'patch_size': PATCH_SIZE, 'patch_start_index': patch_start_idx,
                    'token_grid': [views, grid_height, grid_width],
                    'features_shape': list(features.shape), 'positions_shape': list(positions.shape),
                    'camera_tokens_shape': list(camera.shape), 'register_tokens_shape': list(registers.shape),
                    'channel_halves': {'frame': [0, 1024], 'inter_frame': [1024, 2048]},
                    'preprocessing': {**preprocessing_policy(args.mode), 'per_view_transforms': transforms,
                                      'model_input_shape': input_shape},
                    'compute_precision': args.precision, 'storage_dtype': args.storage_dtype,
                    'torch_version': torch.__version__, 'numpy_version': np.__version__,
                    'heads_executed': False,
                    'head_required_layers': HEAD_REQUIRED_LAYERS,
                    'benchmark_caveat': BENCHMARK_CAVEAT,
                    'benchmark_caveat_source': f'https://github.com/facebookresearch/vggt-omega/blob/{SOURCE_REVISION}/reproduction.md'}
        stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'npz': str(stem.with_suffix('.npz')), 'features_shape': list(features.shape), 'token_grid': metadata['token_grid']}))


if __name__ == '__main__':
    main()
