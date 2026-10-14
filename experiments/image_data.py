"""Materialize a deterministic ImageNet pilot from existing authorized parquet shards."""
import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

DEFAULT_ROOT = Path('/home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot')


def prepare(source, destination, train_count, validation_count, seed):
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Output must not be inside the original dataset')
    destination.mkdir(parents=True, exist_ok=True)
    inventory = {'source': str(source), 'source_dataset': 'ILSVRC/imagenet-1k',
                 'source_license': 'ImageNet Terms of Access; existing authorized local copy only',
                 'seed': seed, 'selection': 'SHA256-order shards; uniform seeded row sample within minimum sufficient shard prefix',
                 'splits': {}}
    for split, requested in [('train', train_count), ('validation', validation_count)]:
        if requested < 1:
            raise ValueError('Pilot counts must be positive')
        shards = sorted((source / 'data').glob(f'{split}-*.parquet'))
        if not shards:
            raise FileNotFoundError(f'No {split} parquet shards in {source / "data"}')
        sizes = {p: pq.ParquetFile(p).metadata.num_rows for p in shards}
        total = sum(sizes.values())
        if total < requested:
            raise ValueError(f'{split}: requested {requested}, source only has {total}; refusing a smaller pilot')
        ordered = sorted(shards, key=lambda p: hashlib.sha256(f'{seed}/{split}/{p.name}'.encode()).digest())
        selected, pool = [], 0
        for shard in ordered:
            selected.append(shard)
            pool += sizes[shard]
            if pool >= requested:
                break
        rng = random.Random(f'{seed}/{split}')
        rows = set(rng.sample(range(pool), requested))
        # Conservative allowance before any image transfer; selected parquet bytes bound decoded byte payload.
        required = sum(p.stat().st_size for p in selected) + 100 * 1024**2
        if shutil.disk_usage(destination).free < required:
            raise OSError(f'Insufficient free space for {split}: need {required} bytes')
        manifest = destination / f'{split}.jsonl'
        temporary = manifest.with_suffix('.jsonl.partial')
        image_bytes, emitted, offset = 0, 0, 0
        with temporary.open('w') as output:
            for shard in selected:
                wanted = {r - offset for r in rows if offset <= r < offset + sizes[shard]}
                offset += sizes[shard]
                row_index = 0
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=['image', 'label']):
                    batch_start = row_index
                    row_index += batch.num_rows
                    local = sorted(r - batch_start for r in wanted if batch_start <= r < row_index)
                    if not local:
                        continue
                    for index in local:
                        record = batch.slice(index, 1).to_pylist()[0]
                        encoded = record['image']['bytes']
                        if not encoded:
                            raise ValueError(f'Missing embedded image bytes in {shard}:{batch_start + index}')
                        with Image.open(io.BytesIO(encoded)) as image:
                            dimensions = list(image.size)
                            image_format = image.format
                            image.verify()
                        extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(image_format, '.image')
                        row = batch_start + index
                        target = destination / split / shard.stem / f'{row:06d}{extension}'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        digest = hashlib.sha256(encoded).hexdigest()
                        if target.exists():
                            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                                raise ValueError(f'Existing project image differs from source: {target}')
                        else:
                            target.write_bytes(encoded)
                        sample = {'sample_id': f'imagenet1k/{split}/{shard.stem}/{row}',
                                  'split': split, 'paths': [str(target)], 'label': int(record['label']),
                                  'original_size_wh': dimensions, 'image_format': image_format,
                                  'sha256': digest, 'source': {'dataset': 'ILSVRC/imagenet-1k',
                                                            'parquet': str(shard), 'row': row}}
                        output.write(json.dumps(sample, sort_keys=True) + '\n')
                        image_bytes += len(encoded)
                        emitted += 1
        if emitted != requested:
            raise RuntimeError(f'{split}: materialized {emitted}, expected {requested}')
        temporary.replace(manifest)
        inventory['splits'][split] = {'source_shards': len(shards), 'source_rows': total,
                                      'source_parquet_bytes': sum(p.stat().st_size for p in shards),
                                      'selected_shards': [str(p) for p in selected],
                                      'pilot_rows': emitted, 'image_bytes': image_bytes,
                                      'manifest': str(manifest),
                                      'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (destination / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(inventory, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/engeld/mnt/datawaha/IN1K'))
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--train-count', type=int, default=10000)
    parser.add_argument('--validation-count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    prepare(args.source, args.output, args.train_count, args.validation_count, args.seed)


if __name__ == '__main__':
    main()
