"""Prepare authorized Omega weights, a scene-disjoint RGB view pilot, or the training-scale group manifest (no inference)."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import numpy as np

from assets import (CHECKPOINT, CHECKPOINT_BYTES, CHECKPOINT_SHA256, CHECKPOINT_URL,
                    DATA, MODEL_ID, MODEL_VARIANT, MODEL_REVISION, SOURCE_REVISION,
                    PROVENANCE, verify_checkpoint)


def checkpoint_command(args):
    # Reuse the user-selected local checkpoint: never mutate it or bypass its access gate.
    digest = verify_checkpoint(args.checkpoint)
    import torch
    from vggt_omega.models import VGGTOmega
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    with torch.device('meta'):
        model = VGGTOmega(autocast=False)
    result = model.load_state_dict(state, strict=True, assign=True)
    provenance = {'model_id': MODEL_ID, 'model_variant': MODEL_VARIANT,
                  'model_revision': MODEL_REVISION,
                  'source_repository': 'https://github.com/facebookresearch/vggt-omega',
                  'source_revision': SOURCE_REVISION, 'checkpoint_url': CHECKPOINT_URL,
                  'checkpoint_path': str(args.checkpoint.resolve()),
                  'checkpoint_bytes': CHECKPOINT_BYTES,
                  'official_lfs_sha256': CHECKPOINT_SHA256, 'observed_sha256': digest,
                  'official_digest_source': 'Authenticated HfApi.model_info(files_metadata=True), matching local sha256sum',
                  'license': 'FAIR Noncommercial Research License v1 October 16, 2024',
                  'upstream_access_gate': 'manual; already-authorized local artifact reused',
                  'checkpoint_download_bytes': 0,
                  'format': 'PyTorch tensor state_dict; weights_only=True, mmap=True; no unsafe pickle loading',
                  'full_model_strict_load': str(result), 'state_entries': len(state),
                  'patch_size': model.aggregator.patch_size,
                  'register_count': model.aggregator.register_token.shape[2],
                  'patch_token_start': model.aggregator.patch_token_start,
                  'head_required_layers': {'dense_depth': model.dense_head.intermediate_layer_idx,
                                           'camera_and_register_tokens': [23]},
                  'benchmark_caveat': 'User-selected original 512 checkpoint: official potential training-data contamination warning; do not claim clean foundation-model benchmarking.',
                  'benchmark_caveat_source': f'https://github.com/facebookresearch/vggt-omega/blob/{SOURCE_REVISION}/reproduction.md',
                  'stock_checkpoint_used': False}
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


SPLIT_FILES = {'train': 'splits/nvs_sem_train.txt', 'validation': 'splits/nvs_sem_val.txt'}
# Original DSLR frame partitions: train scenes use their 'train' frames, validation scenes their 'test' frames.
FRAME_PARTITION = {'train': 'train', 'validation': 'test'}
RGB_DIR = 'dslr/resized_undistorted_images'


def split_scene_ids(root):
    split_ids = {split: (root / name).read_text().split() for split, name in SPLIT_FILES.items()}
    if set(split_ids['train']) & set(split_ids['validation']):
        raise ValueError('Source scene splits overlap')
    return split_ids


def seeded_order(scene_ids, seed):
    return sorted(scene_ids, key=lambda scene: hashlib.sha256((seed + ':' + scene).encode()).hexdigest())


def scene_frames(root, scene_id, split):
    """Numerically ordered RGB frames of the scene's original frame partition, or (None, reason)."""
    scene_root = root / 'data' / scene_id
    rgb = scene_root / RGB_DIR
    if not rgb.is_dir():
        return None, 'RGB directory unavailable'
    paths = sorted((path for path in rgb.iterdir() if path.suffix.lower() in ['.jpg', '.jpeg', '.png']),
                   key=lambda path: (int(''.join(char for char in path.stem if char.isdigit()) or '0'), path.name))
    partition = FRAME_PARTITION[split]
    frame_split_path = scene_root / 'dslr/train_test_lists.json'
    if not frame_split_path.is_file():
        return None, 'train_test_lists.json unavailable'
    allowed = {Path(name).name for name in json.loads(frame_split_path.read_text())[partition]}
    eligible = [path for path in paths if path.name in allowed]
    return {'scene_root': scene_root, 'partition': partition, 'frame_split_path': frame_split_path,
            'all_rgb_frame_count': len(paths), 'eligible': eligible}, None


def inventory_command(args):
    root = args.source_root
    source_scenes = {path.name for path in (root / 'data').iterdir() if path.is_dir()}
    split_files = {split: root / name for split, name in SPLIT_FILES.items()}
    split_ids = split_scene_ids(root)
    scenes, missing = [], []
    for split, target in [('train', args.train_scenes), ('validation', args.val_scenes)]:
        count = 0
        ordered = seeded_order(split_ids[split], args.seed)
        for scene_id in ordered:
            if count >= target:
                break
            frames, reason = scene_frames(root, scene_id, split) if scene_id in source_scenes else (None, 'RGB directory unavailable')
            if frames is None:
                missing.append({'scene_id': scene_id, 'split': split, 'reason': reason})
                continue
            partition, eligible = frames['partition'], frames['eligible']
            if len(eligible) < args.views:
                missing.append({'scene_id': scene_id, 'split': split, 'reason': f'Fewer than {args.views} {partition} RGB views'})
                continue
            scenes.append({'scene_id': scene_id, 'split': split,
                           'source_root': str(frames['scene_root']), 'source_split': split,
                           'source_frame_partition': partition,
                           'source_frame_split_file': str(frames['frame_split_path']),
                           'all_rgb_frame_count': frames['all_rgb_frame_count'], 'eligible_frame_count': len(eligible),
                           'paths': [str(path.resolve()) for path in eligible]})
            count += 1
    inventory = {'dataset': 'ScanNet++', 'modality': 'DSLR resized_undistorted_images',
                 'source_root': str(root), 'available_source_scene_directories': len(source_scenes),
                 'listed_split_scenes': {key: len(value) for key, value in split_ids.items()},
                 'present_scene_directories_in_split': {key: len(set(value) & source_scenes) for key, value in split_ids.items()},
                 'rgb_inventory_scope': 'Only selected eligible pilot scenes; unselected scene RGB counts not enumerated',
                 'split_provenance': {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                      for key, path in split_files.items()},
                 'missing_candidates': missing, 'scenes': scenes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({key: value for key, value in inventory.items() if key != 'scenes'}, indent=2))
    print(json.dumps({'selected_scene_counts': {split: sum(scene['split'] == split for scene in scenes) for split in split_ids},
                      'selected_total_rgb_frames': sum(scene['all_rgb_frame_count'] for scene in scenes),
                      'selected_eligible_rgb_frames': sum(scene['eligible_frame_count'] for scene in scenes)}))


def pilot_command(args):
    inventory = json.loads(args.inventory.read_text())
    scenes = inventory['scenes']
    by_split = {'train': [], 'validation': []}
    seen = set()
    for scene in scenes:
        scene_id = scene['scene_id']
        if scene_id in seen:
            raise ValueError(f'Duplicate scene identity / split leakage: {scene_id}')
        seen.add(scene_id)
        split = scene['split']
        if split not in by_split:
            continue
        paths = scene['paths']
        if len(paths) >= args.views:
            by_split[split].append(scene)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = {}
    manifest = []
    staged_bytes = 0
    for split, target in [('train', args.train_scenes), ('validation', args.val_scenes)]:
        candidates = sorted(by_split[split], key=lambda scene: hashlib.sha256((args.seed + ':' + scene['scene_id']).encode()).hexdigest())
        selected[split] = [scene['scene_id'] for scene in candidates[:target]]
        for scene in candidates[:target]:
            paths = scene['paths']
            stride = min(args.stride, max(1, (len(paths) - 1) // max(1, args.views - 1)))
            window = (args.views - 1) * stride + 1
            start = int(hashlib.sha256((args.seed + ':views:' + scene['scene_id']).encode()).hexdigest(), 16) % (len(paths) - window + 1)
            indices = [start + index * stride for index in range(args.views)]
            chosen = [str(Path(paths[index]).resolve(strict=True)) for index in indices]
            # Metadata-only image inspection is preparation, not model inference.
            from PIL import Image
            for path in chosen:
                with Image.open(path) as image:
                    if image.width < 16 or image.height < 16:
                        raise ValueError(f'Invalid tiny RGB view: {path}')
            staged, digests = [], []
            for source_path in chosen:
                source = Path(source_path)
                destination = args.output / 'rgb' / scene['scene_id'] / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(destination.parent).free < source.stat().st_size + 1024 ** 3:
                    raise RuntimeError('Insufficient free space for RGB pilot plus 1GiB reserve')
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        raise ValueError(f'Existing staged RGB differs from its source: {destination}')
                else:
                    shutil.copyfile(source, destination)
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                        raise ValueError(f'Copy integrity mismatch: {destination}')
                staged_bytes += destination.stat().st_size
                staged.append(str(destination.resolve()))
                digests.append(digest)
            manifest.append({'sample_id': scene['scene_id'] + f'-v{args.views}',
                             'scene_id': scene['scene_id'], 'split': split,
                             'dataset': inventory['dataset'], 'source_root': scene.get('source_root'),
                             'paths': staged, 'source_paths': chosen,
                             'source_view_ids': [Path(path).stem for path in chosen], 'rgb_sha256': digests,
                             'source_frame_partition': scene.get('source_frame_partition'),
                             'source_frame_split_file': scene.get('source_frame_split_file'),
                             'source_split': scene.get('source_split', split),
                             'split_provenance': inventory['split_provenance'],
                             'view_sampling': {'seed': args.seed, 'order': 'source temporal numeric order',
                                               'source_frame_count': len(paths), 'start_rank': start,
                                               'stride_ranks': stride, 'selected_ranks': indices},
                             'reference_view_index': 0})
    manifest_path = args.output / f'pilot_{args.views}views.jsonl'
    manifest_path.write_text(''.join(json.dumps(row) + '\n' for row in manifest))
    summary = {'dataset': inventory['dataset'], 'inventory_path': str(args.inventory.resolve()),
               'source_inventory_scenes': len(scenes), 'eligible_scenes': {key: len(value) for key, value in by_split.items()},
               'requested_scenes': {'train': args.train_scenes, 'validation': args.val_scenes},
               'actual_scenes': {key: len(value) for key, value in selected.items()},
               'selected_scene_ids': selected, 'views_per_scene': args.views,
               'manifest': str(manifest_path.resolve()), 'split_provenance': inventory['split_provenance'],
               'seed': args.seed, 'downloaded_dataset_bytes': 0, 'staged_rgb_bytes': staged_bytes,
               'reuse_originals_without_modification': True,
               'shortfall': {key: max(0, target - len(selected[key])) for key, target in [('train', args.train_scenes), ('validation', args.val_scenes)]}}
    (args.output / 'pilot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def group_starts(frame_count, views, groups, max_stride):
    """Per-scene rank stride and evenly spread distinct window starts, or None when ineligible."""
    stride = max_stride if views == 1 else min(max_stride, (frame_count - groups) // (views - 1))
    if stride < 1:
        return None
    window = (views - 1) * stride + 1
    if frame_count - window + 1 < groups:
        return None
    starts = np.rint(np.linspace(0, frame_count - window, groups)).astype(int).tolist()
    if len(set(starts)) != groups:
        raise RuntimeError(f'Non-distinct group starts {starts}')
    return stride, starts


def inspect_view(path):
    """Read once: SHA256, bytes and decodable RGB geometry (metadata only, no inference)."""
    import io
    from PIL import Image
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    if width < 16 or height < 16:
        raise ValueError(f'Invalid tiny RGB view: {path}')
    return hashlib.sha256(data).hexdigest(), len(data)


def write_atomic(path, text):
    partial = path.with_name(path.name + '.partial')
    partial.write_text(text)
    partial.replace(path)


def groups_command(args):
    """Training-scale manifest: several ordered view groups per scene, original RGB referenced in place."""
    root = args.source_root
    split_ids = split_scene_ids(root)
    targets = {'train': (args.train_scenes, args.train_groups), 'validation': (args.val_scenes, args.val_groups)}
    scene_inventory, rows, selected = [], [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split, (scene_target, groups) in targets.items():
            ordered = seeded_order(split_ids[split], args.seed)
            inspected = list(pool.map(lambda scene_id, split=split: scene_frames(root, scene_id, split), ordered))
            selected[split] = []
            for scene_id, (frames, reason) in zip(ordered, inspected):
                entry = {'scene_id': scene_id, 'split': split, 'eligible': False}
                plan = None
                if frames is not None:
                    entry.update(all_rgb_frame_count=frames['all_rgb_frame_count'],
                                 eligible_frame_count=len(frames['eligible']),
                                 source_frame_partition=frames['partition'])
                    plan = group_starts(len(frames['eligible']), args.views, groups, args.stride)
                    reason = None if plan else f'Fewer than {groups} distinct {args.views}-view groups in {frames["partition"]} frames'
                entry['eligible'], entry['reason'] = plan is not None, reason
                if plan is not None and len(selected[split]) < scene_target:
                    entry['selected'] = True
                    selected[split].append(scene_id)
                    stride, starts = plan
                    eligible = frames['eligible']
                    for group_index, start in enumerate(starts):
                        ranks = [start + view * stride for view in range(args.views)]
                        chosen = [eligible[rank] for rank in ranks]
                        rows.append({'sample_id': f'{scene_id}-v{args.views}-g{group_index:02d}',
                                     'scene_id': scene_id, 'split': split, 'dataset': 'ScanNet++',
                                     'source_root': str(root), 'paths': [str(path) for path in chosen],
                                     'source_relative_paths': [str(path.relative_to(root)) for path in chosen],
                                     'source_view_ids': [path.stem for path in chosen],
                                     'source_split': split, 'source_frame_partition': frames['partition'],
                                     'source_frame_split_file': str(frames['frame_split_path'].relative_to(root)),
                                     'view_sampling': {'seed': args.seed,
                                                       'order': 'numeric source filename order within the original frame partition',
                                                       'eligible_frame_count': len(eligible), 'groups_per_scene': groups,
                                                       'group_index': group_index, 'start_rank': start,
                                                       'stride_ranks': stride, 'selected_ranks': ranks,
                                                       'start_policy': 'round(linspace(0, eligible - window, groups)); stride = min(max_stride, (eligible - groups) // (views - 1))'},
                                     'reference_view_index': 0})
                scene_inventory.append(entry)
        if len({tuple(row['paths']) for row in rows}) != len(rows) or len({row['sample_id'] for row in rows}) != len(rows):
            raise RuntimeError('Duplicate ordered view group or sample identity')
        unique = sorted({path for row in rows for path in row['paths']})
        digests = dict(zip(unique, pool.map(lambda path: inspect_view(Path(path)), unique)))
    for row in rows:
        row['rgb_sha256'] = [digests[path][0] for path in row['paths']]
        row['rgb_bytes'] = [digests[path][1] for path in row['paths']]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / f'groups_{args.views}views.jsonl'
    write_atomic(manifest, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    split_provenance = {split: {'path': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()}
                        for split, name in SPLIT_FILES.items()}
    write_atomic(args.output / 'groups_inventory.json',
                 json.dumps({'source_root': str(root), 'seed': args.seed, 'split_provenance': split_provenance,
                             'scenes': scene_inventory}, indent=2) + '\n')
    summary = {'dataset': 'ScanNet++', 'modality': f'DSLR {RGB_DIR}', 'source_root': str(root),
               'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
               'views_per_group': args.views, 'max_stride_ranks': args.stride, 'seed': args.seed,
               'split_provenance': split_provenance, 'frame_partitions': FRAME_PARTITION,
               'listed_split_scenes': {split: len(ids) for split, ids in split_ids.items()},
               'eligible_scenes': {split: sum(e['eligible'] for e in scene_inventory if e['split'] == split) for split in targets},
               'requested': {split: {'scenes': scenes, 'groups_per_scene': groups} for split, (scenes, groups) in targets.items()},
               'selected_scenes': {split: len(ids) for split, ids in selected.items()},
               'groups': {split: sum(row['split'] == split for row in rows) for split in targets},
               'shortfall_scenes': {split: targets[split][0] - len(ids) for split, ids in selected.items()},
               'selected_scene_ids': selected, 'unique_rgb_files': len(unique),
               'referenced_rgb_bytes': sum(size for _, size in digests.values()),
               'staged_rgb_bytes': 0, 'downloaded_dataset_bytes': 0,
               'portability': 'paths are absolute under source_root; source_relative_paths are relative to it; map the root explicitly (compression.extract --path-map OLD=NEW)'}
    write_atomic(args.output / 'groups_summary.json', json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'selected_scene_ids'}, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    checkpoint = sub.add_parser('checkpoint', help='Verify authorized local Omega-512 digest and strict full-model compatibility')
    checkpoint.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    checkpoint.set_defaults(action=checkpoint_command)
    inventory = sub.add_parser('inventory', help='Inspect authorized local ScanNet++ RGB and original scene/frame splits')
    inventory.add_argument('--source-root', type=Path, default=Path('/home/engeld/mnt/ibex_c2324/data/scannetpp'))
    inventory.add_argument('--output', type=Path, default=DATA / 'inventory.json')
    inventory.add_argument('--train-scenes', type=int, default=20)
    inventory.add_argument('--val-scenes', type=int, default=5)
    inventory.add_argument('--views', type=int, default=4)
    inventory.add_argument('--stride', type=int, default=10)
    inventory.add_argument('--seed', default='vggt-pilot-20261014')
    inventory.set_defaults(action=inventory_command)
    pilot = sub.add_parser('pilot', help='Sample one ordered group per source scene from an inspected inventory')
    pilot.add_argument('--inventory', type=Path, default=DATA / 'inventory.json')
    pilot.add_argument('--output', type=Path, default=DATA)
    pilot.add_argument('--views', type=int, default=4)
    pilot.add_argument('--stride', type=int, default=10)
    pilot.add_argument('--train-scenes', type=int, default=20)
    pilot.add_argument('--val-scenes', type=int, default=5)
    pilot.add_argument('--seed', default='vggt-pilot-20261014')
    pilot.set_defaults(action=pilot_command)
    groups = sub.add_parser('groups', help='Training-scale scene-disjoint manifest: many ordered view groups per scene, source RGB in place')
    groups.add_argument('--source-root', type=Path, default=Path('/home/engeld/mnt/ibex_c2324/data/scannetpp'))
    groups.add_argument('--output', type=Path, default=DATA / 'groups')
    groups.add_argument('--views', type=int, default=4)
    groups.add_argument('--stride', type=int, default=10, help='Maximum rank stride between consecutive views')
    groups.add_argument('--train-scenes', type=int, default=200)
    groups.add_argument('--train-groups', type=int, default=16)
    groups.add_argument('--val-scenes', type=int, default=25)
    groups.add_argument('--val-groups', type=int, default=4)
    groups.add_argument('--workers', type=int, default=16, help='I/O threads for scene listing and RGB hashing')
    groups.add_argument('--seed', default='vggt-pilot-20261014')
    groups.set_defaults(action=groups_command)
    args = parser.parse_args()
    if hasattr(args, 'views') and (args.views < 1 or args.stride < 1 or args.train_scenes < 0 or args.val_scenes < 0):
        parser.error('views and stride must be positive; scene counts cannot be negative')
    if hasattr(args, 'train_groups') and (args.train_groups < 1 or args.val_groups < 1 or args.workers < 1):
        parser.error('group counts and workers must be positive')
    args.action(args)


if __name__ == '__main__':
    main()
