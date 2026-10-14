#!/usr/bin/env python3
"""Prepare the official V-JEPA 2.1 encoder, SSv2 corpus, or Kinetics pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = Path('/home/engeld/mnt/datawaha/compressing-latents')
REVISION = '204698b45b3712590f06245fbfba32d3be539812'
MODEL_ID = 'facebookresearch/vjepa2:vjepa2_1_vit_base_384'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt'
CHECKPOINT_BYTES = 1664223428

SSV2_PAGE = 'https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads'
SSV2_BASE = 'https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/'
SSV2_PARTS = [('20bn-something-something-v2-00', 10000000000),
              ('20bn-something-something-v2-01', 9444975475)]
SSV2_INSTRUCTIONS = 'https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf'
SSV2_LICENSE = 'https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf'

CHECKPOINT_SHA256 = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def download(url, path, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if size is None:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-15', 'User-Agent': 'Mozilla/5.0'}), timeout=60) as r:
            size = int(r.headers['Content-Range'].rsplit('/', 1)[-1]) if r.status == 206 else int(r.headers['Content-Length'])
    if path.exists() and path.stat().st_size == size:
        return
    partial = path.with_name(path.name + '.partial')
    needed = max(0, size - (partial.stat().st_size if partial.exists() else 0))
    if shutil.disk_usage(path.parent).free < needed + 2 * 1024**3:
        raise RuntimeError(f'Insufficient space for {url}')
    subprocess.run(['curl', '-fL', '--http1.1', '--retry', '3', '--retry-all-errors',
                    '--silent', '--show-error', '--user-agent', 'Mozilla/5.0',
                    '--continue-at', '-', '--output', str(partial), url], check=True)
    if partial.stat().st_size != size:
        raise RuntimeError(f'Incomplete transfer: {partial}')
    partial.replace(path)


def checkpoint(args):
    import torch
    from safetensors.torch import save_file
    base = args.root / 'checkpoints/vjepa21'
    base.mkdir(parents=True, exist_ok=True)
    source = base / 'source'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/vjepa2.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f'Source is {actual}, expected {REVISION}; do not overwrite existing source')
    raw = base / 'vjepa2_1_vitb_dist_vitG_384.pt'
    download(CHECKPOINT_URL, raw)
    if raw.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError('Official checkpoint size changed; review provenance before loading')
    raw_hash = digest(raw)
    if raw_hash != CHECKPOINT_SHA256:
        raise RuntimeError('Official checkpoint checksum differs from recorded release')
    # weights_only forbids arbitrary checkpoint code execution. The base Hub entry
    # selects ema_encoder, not target_encoder, predictor or the action model.
    state = torch.load(raw, map_location='cpu', weights_only=True)
    encoder = {k.replace('module.', '').replace('backbone.', ''): v.contiguous()
               for k, v in state['ema_encoder'].items()}
    out = base / 'ema_encoder.safetensors'
    save_file(encoder, str(out), metadata={'model_id': MODEL_ID, 'revision': REVISION,
              'checkpoint_key': 'ema_encoder', 'source_sha256': raw_hash})
    save_json(base / 'provenance.json', {'model_id': MODEL_ID, 'revision': REVISION,
              'source_url': CHECKPOINT_URL, 'source_bytes': raw.stat().st_size,
              'source_sha256': raw_hash, 'encoder_path': str(out),
              'encoder_bytes': out.stat().st_size, 'encoder_sha256': digest(out),
              'checkpoint_key': 'ema_encoder', 'tensor_count': len(encoder),
              'hub_url_bug': 'Pinned Hub source points to localhost:8300; use documented Meta download URL instead.'})
    print(out)


def annotations(base, split):
    path = base / f'{split}.csv'
    download(f'https://s3.amazonaws.com/kinetics/400/annotations/{split}.csv', path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    # Official archive names use zero-padded second intervals.
    return {f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}.mp4": r for r in rows}


def kinetics(args):
    base = args.root / 'data/vjepa21'
    base.mkdir(parents=True, exist_ok=True)
    ann = {s: annotations(base, s) for s in ('train', 'val')}
    val_ids = {r['youtube_id'] for r in ann['val'].values()}
    inventory = []
    manifest = []
    seen_sources = set()
    for split, count, max_parts in [('val', args.val, 1), ('train', args.train, args.max_train_parts)]:
        candidates = {}
        for part in range(max_parts):
            archive = base / f'{split}_part_{part}.tar.gz'
            url = f'https://s3.amazonaws.com/kinetics/400/{split}/part_{part}.tar.gz'
            download(url, archive)
            with tarfile.open(archive, 'r:*') as tar:
                for member in tar:
                    name = Path(member.name).name
                    row = ann[split].get(name)
                    if not member.isfile() or row is None:
                        continue
                    if split == 'train' and row['youtube_id'] in val_ids:
                        continue
                    candidates[name] = (archive, member.name, row)
            inventory.append({'split': split, 'url': url, 'path': str(archive),
                              'bytes': archive.stat().st_size, 'sha256': digest(archive)})
            eligible = {v[2]['youtube_id'] for v in candidates.values()} - seen_sources
            if len(eligible) >= count:
                break
        names = sorted(candidates, key=lambda n: hashlib.sha256(('vjepa21-pilot-0:' + n).encode()).hexdigest())
        chosen = []
        for name in names:
            archive, member_name, row = candidates[name]
            if row['youtube_id'] in seen_sources:
                continue
            seen_sources.add(row['youtube_id'])
            chosen.append((name, archive, member_name, row))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise RuntimeError(f'Only {len(chosen)} eligible {split} videos in bounded shards; requested {count}')
        clips = base / 'clips' / split
        clips.mkdir(parents=True, exist_ok=True)
        # Explicit member file copies, never extractall: tar paths cannot escape.
        by_archive = {}
        for item in chosen:
            by_archive.setdefault(item[1], []).append(item)
        for archive, items in by_archive.items():
            with tarfile.open(archive, 'r:*') as tar:
                for name, _, member_name, row in items:
                    dest = clips / name
                    member = tar.getmember(member_name)
                    if not dest.exists() or dest.stat().st_size != member.size:
                        with tar.extractfile(member) as src, dest.open('wb') as out:
                            shutil.copyfileobj(src, out)
                    manifest.append({'sample_id': f'k400:{split}:{dest.stem}', 'split': split,
                                     'paths': [str(dest.resolve())], 'dataset': 'Kinetics-400',
                                     'youtube_id': row['youtube_id'], 'label': row['label'],
                                     'source_start_seconds': int(row['time_start']),
                                     'source_end_seconds': int(row['time_end']),
                                     'source_archive': str(archive), 'bytes': dest.stat().st_size})
    manifest.sort(key=lambda r: (r['split'], r['sample_id']))
    with (base / 'pilot.jsonl').open('w') as f:
        for record in manifest:
            f.write(json.dumps(record) + '\n')
    save_json(base / 'inventory.json', {'dataset': 'Kinetics-400', 'selection_reason':
              'Explicitly requested bounded Kinetics-400 alternative using official CVDF public S3 archives; SSv2 has a separate full-corpus preparation command.',
              'upstream': 'https://github.com/cvdfoundation/kinetics-dataset',
              'rights': 'Underlying video rights/terms remain with original creators; public availability is not blanket redistribution permission.',
              'pilot_selection': 'sha256(vjepa21-pilot-0:<filename>), bounded official parts; one clip per YouTube source; all validation source IDs excluded from training',
              'train_count': args.train, 'val_count': args.val, 'archives': inventory,
              'extracted_bytes': sum(r['bytes'] for r in manifest), 'manifest_sha256': digest(base / 'pilot.jsonl')})
    print(base / 'pilot.jsonl')


def ssv2_subset(base, records=None):
    """Select the requested small cache corpus without changing source splits."""
    if records is None:
        with (base / 'manifest.jsonl').open() as source:
            records = [json.loads(line) for line in source]
    selected = []
    outputs = {}
    seed = 'vjepa21-subset-0:'
    for split, count in [('train', 10000), ('val', 1000)]:
        candidates = [record for record in records if record['split'] == split]
        candidates.sort(key=lambda record: hashlib.sha256((seed + record['sample_id']).encode()).hexdigest())
        if len(candidates) < count:
            raise RuntimeError(f'SSv2 {split} has fewer than the requested {count} subset clips')
        subset = candidates[:count]
        selected.extend(subset)
        out = base / f'subset-{split}.jsonl'
        partial = out.with_suffix('.jsonl.partial')
        with partial.open('w') as target:
            for record in subset:
                target.write(json.dumps(record) + '\n')
        partial.replace(out)
        outputs[split] = {'path': str(out), 'count': count, 'sha256': digest(out)}
    out = base / 'subset.jsonl'
    partial = out.with_suffix('.jsonl.partial')
    with partial.open('w') as target:
        for record in selected:
            target.write(json.dumps(record) + '\n')
    partial.replace(out)
    outputs['manifest'] = {'path': str(out), 'count': len(selected), 'sha256': digest(out)}
    inventory = {'selection': 'First per-split SHA256(seed + sample_id) ranks',
                 'seed': seed, 'train_count': 10000, 'val_count': 1000,
                 'source_manifest_sha256': digest(base / 'manifest.jsonl'),
                 'manifests': outputs}
    save_json(base / 'subset-inventory.json', inventory)
    return inventory


def ssv2(args, selected_ids=None):
    """Index official clips, optionally limiting extraction to the requested cache subset."""
    base = args.root / 'data/vjepa21/ssv2'
    base.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    label_map = json.loads(args.class_labels.read_text()) if args.class_labels else None
    labels_provenance = []
    counts = {}
    source_counts = {}
    for split, labels in [('train', args.train_labels), ('val', args.val_labels)]:
        items = json.loads(labels.read_text())
        source_counts[split] = len(items)
        if selected_ids is not None:
            items = [item for item in items if str(item['id']) in selected_ids]
        counts[split] = len(items)
        if not items:
            raise RuntimeError(f'Empty official SSv2 {split} annotations')
        labels_provenance.append({'split': split, 'path': str(labels.resolve()),
                                  'bytes': labels.stat().st_size, 'sha256': digest(labels)})
        for item in items:
            video_id = str(item['id'])
            if not video_id.isdigit() or video_id in seen:
                raise RuntimeError(f'Invalid or repeated official SSv2 ID: {video_id}')
            seen.add(video_id)
            path = args.video_root / f'{video_id}.webm'
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f'Missing official SSv2 {split} video: {path}')
            template = item['template']
            class_name = template.replace('[', '').replace(']', '')
            records.append({'sample_id': f'ssv2:{video_id}', 'split': split,
                            'source_split': 'validation' if split == 'val' else split,
                            'paths': [str(path.resolve())], 'dataset': 'Something-Something-V2',
                            'video_id': video_id, 'label': template,
                            'class_id': int(label_map[class_name]) if label_map is not None else None,
                            'text': item.get('label'), 'placeholders': item.get('placeholders', []),
                            'source_labels': str(labels.resolve()), 'bytes': path.stat().st_size,
                            'sha256': digest(path)})
    outputs = {}
    for name, subset in [('train', [r for r in records if r['split'] == 'train']),
                         ('val', [r for r in records if r['split'] == 'val']),
                         ('manifest', records)]:
        filename = f'{name}.jsonl' if selected_ids is None else ('subset.jsonl' if name == 'manifest' else f'subset-{name}.jsonl')
        out = base / filename
        partial = out.with_suffix('.jsonl.partial')
        with partial.open('w') as f:
            for record in subset:
                f.write(json.dumps(record) + '\n')
        partial.replace(out)
        outputs[name] = {'path': str(out), 'count': len(subset), 'sha256': digest(out)}
    inventory = {'dataset': 'Something-Something-V2', 'upstream': SSV2_PAGE,
                 'selection': 'All official train and validation annotations' if selected_ids is None else 'Requested feature-cache subset; first per-split SHA256(seed + sample_id) ranks',
                 'source_annotation_counts': source_counts,
                 'train_count': counts['train'], 'val_count': counts['val'],
                 'extracted_bytes': sum(r['bytes'] for r in records),
                 'annotations': labels_provenance, 'manifests': outputs,
                 'class_labels': {'path': str(args.class_labels.resolve()), 'sha256': digest(args.class_labels)}
                 if args.class_labels else None}
    if selected_ids is None:
        inventory['feature_cache_subset'] = ssv2_subset(base, records)
    else:
        inventory['seed'] = 'vjepa21-subset-0:'
        save_json(base / 'subset-inventory.json', inventory)
    save_json(base / 'inventory.json', inventory)
    print(outputs['manifest']['path'], flush=True)
    return inventory


def ssv2_download(args):
    """Use only Qualcomm's public asset URLs, retaining their original packages."""
    base = args.root / 'data/vjepa21/ssv2'
    packages = base / 'packages'
    packages.mkdir(parents=True, exist_ok=True)
    # Official links are named zip1/zip2, but current responses are raw split TGZ
    # bytes. HEAD is forbidden; public GET Range supplies length and provenance.
    assets = [(name, SSV2_BASE + name, size) for name, size in SSV2_PARTS]
    assets += [('20bn-something-something-download-package-labels.zip',
                SSV2_BASE + '20bn-something-something-download-package-labels.zip', None),
               ('download-instructions.pdf', SSV2_INSTRUCTIONS, None),
               ('research-use-license.pdf', SSV2_LICENSE, None)]
    source_records = []
    for name, url, expected in assets:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-15', 'User-Agent': 'Mozilla/5.0'}), timeout=60) as response:
            size = int(response.headers['Content-Range'].rsplit('/', 1)[-1]) if response.status == 206 else int(response.headers['Content-Length'])
            if expected is not None and size != expected:
                raise RuntimeError(f'Official SSv2 release size changed: {name}: {size} != {expected}')
            source_records.append({'name': name, 'url': url, 'path': str(packages / name),
                                   'bytes': size, 'etag': response.headers.get('ETag'),
                                   'last_modified': response.headers.get('Last-Modified')})
    save_json(base / 'download-sources.json', {'upstream': SSV2_PAGE, 'assets': source_records})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, record['url'], Path(record['path']), record['bytes'])
                   for record in source_records]
        for future in futures:
            future.result()
    for record in source_records:
        record['sha256'] = digest(Path(record['path']))
    save_json(base / 'download-sources.json', {'upstream': SSV2_PAGE, 'assets': source_records})
    labels = base / 'labels'
    labels.mkdir(exist_ok=True)
    labels_archive = packages / '20bn-something-something-download-package-labels.zip'
    with zipfile.ZipFile(labels_archive) as archive:
        names = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            if name in names:
                raise RuntimeError(f'Duplicate label package basename: {name}')
            names[name] = member
        for required in ('train.json', 'validation.json', 'labels.json'):
            if required not in names:
                raise RuntimeError(f'Official labels package lacks {required}')
        # Copy individual basenames only; never trust archive paths.
        for name, member in names.items():
            with archive.open(member) as src, (labels / name).open('wb') as dst:
                shutil.copyfileobj(src, dst)
    train_items = json.loads((labels / 'train.json').read_text())
    val_items = json.loads((labels / 'validation.json').read_text())
    if (len(train_items), len(val_items)) != (168913, 24777):
        raise RuntimeError('Official SSv2 split counts changed; review release before extraction')
    wanted = {str(item['id']) for item in train_items + val_items}
    if len(wanted) != len(train_items) + len(val_items):
        raise RuntimeError('Duplicate IDs in official SSv2 split labels')
    # Full official archives and labels are retained. Only the user-requested
    # small feature-cache corpus needs standalone files on the shared mount.
    wanted = set()
    for items, count in [(train_items, 10000), (val_items, 1000)]:
        ranked = sorted(items, key=lambda item: hashlib.sha256(
            ('vjepa21-subset-0:ssv2:' + str(item['id'])).encode()).hexdigest())
        wanted.update(str(item['id']) for item in ranked[:count])
    clips = base / 'videos'
    clips.mkdir(exist_ok=True)
    extracted = set()
    parts = [str(packages / name) for name, _ in SSV2_PARTS]
    # Stream concatenation without allocating another 19.4 GB archive. Copy
    # only regular members belonging to the requested train/validation subset.
    stream = subprocess.Popen(['cat', *parts], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=stream.stdout, mode='r|gz') as archive:
            for member in archive:
                name = Path(member.name).name
                video_id = Path(name).stem
                if not member.isfile() or not name.endswith('.webm') or video_id not in wanted:
                    continue
                if video_id in extracted:
                    raise RuntimeError(f'Duplicate video in official archive: {video_id}')
                extracted.add(video_id)
                dest = clips / f'{video_id}.webm'
                if dest.exists() and dest.stat().st_size == member.size:
                    continue
                partial = dest.with_suffix('.webm.partial')
                with archive.extractfile(member) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                if partial.stat().st_size != member.size:
                    raise RuntimeError(f'Incomplete archive member: {member.name}')
                partial.replace(dest)
        stream.stdout.close()
        if stream.wait() != 0:
            raise RuntimeError('Official archive concatenation failed')
    finally:
        if stream.poll() is None:
            stream.terminate()
            stream.wait()
    missing = wanted - extracted
    if missing:
        raise RuntimeError(f'Official archive lacks {len(missing)} train/validation clips')
    args.video_root = clips
    args.train_labels = labels / 'train.json'
    args.val_labels = labels / 'validation.json'
    args.class_labels = labels / 'labels.json'
    inventory = ssv2(args, selected_ids=wanted)
    inventory.update({'archives': source_records,
                      'rights': 'Qualcomm research-use terms apply; retain bundled notices. No dataset redistribution by this repository.',
                      'published_source_archive_video_count': 220847, 'materialized_splits': ['train', 'validation'],
                      'full_official_archives_available': True, 'materialized_selection': '10000 train / 1000 validation feature-cache subset'})
    save_json(base / 'inventory.json', inventory)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('checkpoint').set_defaults(func=checkpoint)
    k = sub.add_parser('kinetics', help='Explicit Kinetics-400 alternative; bounded official shards')
    k.add_argument('--train', type=int, default=1000)
    k.add_argument('--val', type=int, default=100)
    k.add_argument('--max-train-parts', type=int, default=2)
    k.set_defaults(func=kinetics)
    sub.add_parser('ssv2-download', help='Download complete public official SSv2 train/validation corpus').set_defaults(func=ssv2_download)
    s = sub.add_parser('ssv2', help='Index every official train/validation clip in an existing authorized SSv2 corpus')
    s.add_argument('--video-root', type=Path, required=True)
    s.add_argument('--train-labels', type=Path, required=True)
    s.add_argument('--val-labels', type=Path, required=True)
    s.add_argument('--class-labels', type=Path)
    s.set_defaults(func=ssv2)
    args = p.parse_args()
    for key in ('train', 'val', 'max_train_parts'):
        if hasattr(args, key) and getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
