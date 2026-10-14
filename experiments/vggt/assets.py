"""Exact user-selected, authorized VGGT-Omega-1B-512 asset; no stock fallback."""
from pathlib import Path
import hashlib

ROOT = Path('/home/engeld/mnt/datawaha/compressing-latents')
MODEL_ID = 'facebook/VGGT-Omega'
MODEL_VARIANT = 'VGGT-Omega-1B-512'
MODEL_REVISION = '1041e80fc0e911235d3426b0a3d9a81075111a53'
SOURCE_REVISION = '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
CHECKPOINT_BYTES = 4576706117
CHECKPOINT_SHA256 = 'c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934'
CHECKPOINT_URL = f'https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/vggt_omega_1b_512.pt'
CHECKPOINT = Path('/home/engeld/mnt/datawaha/vggt_omega_1b_512.pt')
SOURCE = ROOT / 'checkpoints/vggt/omega_source'
PROVENANCE = ROOT / 'checkpoints/vggt/provenance.json'
DATA = ROOT / 'data/vggt'
FEATURES = ROOT / 'features/vggt'


def verify_checkpoint(path):
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(f'Incomplete or wrong Omega-1B-512 checkpoint: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint does not match authenticated official Omega-512 LFS SHA256: {path}')
    return digest.hexdigest()
