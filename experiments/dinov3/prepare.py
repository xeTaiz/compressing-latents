"""Prepare the authorized native DINOv3 ViT-7B/16 LVD teacher and official source."""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import torch

MODEL_ID = 'facebookresearch/dinov3:dinov3_vit7b16:LVD1689M'
REVISION = '6876159a11b4df116f30f667f8c9888617df0751'
WEIGHT_NAME = 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth'
WEIGHT_BYTES = 26864407683
DEFAULT_CHECKPOINT = Path('/home/engeld/mnt/datawaha/compressing-latents/checkpoints/dinov3/vit7b16-lvd1689m')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(16 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source(destination):
    root = destination / f'dinov3-{REVISION}'
    archive = destination / f'dinov3-{REVISION}.tar.gz'
    url = f'https://codeload.github.com/facebookresearch/dinov3/tar.gz/{REVISION}'
    if not archive.exists():
        partial = archive.with_suffix('.gz.partial')
        with urllib.request.urlopen(url) as response, partial.open('wb') as output:
            shutil.copyfileobj(response, output, 1024**2)
        partial.replace(archive)
    if not (root / 'hubconf.py').exists():
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar.getmembers():
                target = (destination / member.name).resolve()
                if destination.resolve() not in target.parents or member.issym() or member.islnk():
                    raise ValueError('Unsafe source archive member')
                if not (member.isdir() or member.isfile()):
                    raise ValueError('Unexpected source archive member')
            tar.extractall(destination)
    return root, {'repository': 'https://github.com/facebookresearch/dinov3', 'revision': REVISION,
                  'archive_url': url, 'archive_sha256': sha256(archive), 'path': str(root)}


def native_factory(source):
    sys.path.insert(0, str(source.resolve()))
    # Native model imports initialize Dynamo; do this before the meta-device context.
    from importlib import import_module
    import_module('dinov3.models.vision_transformer')
    from dinov3.hub.backbones import dinov3_vit7b16
    return dinov3_vit7b16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--source-checkpoint', type=Path, help='Already-authorized local original; never downloads gated weights')
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    weight = destination / WEIGHT_NAME
    if args.source_checkpoint:
        source = args.source_checkpoint.resolve()
        if source != weight:
            if not weight.exists():
                if shutil.disk_usage(destination).free < source.stat().st_size + 1024**3:
                    raise OSError('Insufficient checkpoint free space')
                partial = weight.with_suffix('.pth.partial')
                with source.open('rb') as incoming, partial.open('xb') as output:
                    shutil.copyfileobj(incoming, output, 16 * 1024**2)
                if sha256(source) != sha256(partial):
                    raise ValueError('Authorized checkpoint copy integrity mismatch')
                partial.rename(weight)
            elif sha256(source) != sha256(weight):
                raise ValueError('Existing destination differs; refusing overwrite')
    if not weight.exists():
        raise FileNotFoundError(f'Authorized native checkpoint absent: {weight}. Copy the user-authorized KW61627 /code/DRRT/teacher_weights/{WEIGHT_NAME} through Worker Harness, or supply --source-checkpoint.')
    digest = sha256(weight)
    if weight.stat().st_size != WEIGHT_BYTES or not digest.startswith('a955f4ea'):
        raise ValueError('Checkpoint differs from official native ViT-7B/16 LVD filename hash/size')
    source_root, source_metadata = prepare_source(destination)
    # Metadata-only checkpoint inspection and exact native architecture matching, not inference.
    state = torch.load(weight, map_location='meta', mmap=True, weights_only=True)
    factory = native_factory(source_root)
    with torch.device('meta'):
        model = factory(pretrained=False)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Native checkpoint keys do not match the official 7B architecture')
    mismatched = [key for key in expected if tuple(state[key].shape) != tuple(expected[key].shape)]
    if mismatched:
        raise ValueError(f'Native checkpoint shape mismatch: {mismatched[:8]}')
    status = {'available': True, 'model_id': MODEL_ID, 'variant': 'ViT-7B/16',
              'pretraining_dataset': 'LVD-1689M', 'revision': REVISION,
              'checkpoint': {'filename': weight.name, 'bytes': weight.stat().st_size, 'sha256': digest,
                             'format': 'native PyTorch state_dict; weights_only=True'},
              'source': source_metadata, 'license': 'DINOv3 License; user-authorized existing local checkpoint',
              'architecture': {'patch_size': model.patch_size, 'embed_dim': model.embed_dim,
                               'depth': model.n_blocks, 'num_heads': model.num_heads,
                               'register_tokens': model.n_storage_tokens, 'norm_eps': model.norm.eps,
                               'parameters_and_buffers_numel': sum(t.numel() for t in state.values()),
                               'state_dtypes': sorted({str(t.dtype) for t in state.values()}),
                               'inspected_tensors': {key: list(state[key].shape) for key in
                                                     ['cls_token', 'storage_tokens', 'patch_embed.proj.weight',
                                                      'blocks.0.attn.qkv.weight', 'norm.weight']}}}
    original_provenance = destination / 'native-asset.json'
    if original_provenance.exists():
        status['authorized_copy_provenance'] = json.loads(original_provenance.read_text())
    (destination / 'asset.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
