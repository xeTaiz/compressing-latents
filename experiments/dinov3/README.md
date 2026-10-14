# DINOv3 ViT-7B/16 frozen dense features

The primary experiment now uses the user's **existing authorized DINOv3 ViT-7B/16 LVD-1689M teacher** from KW61627 `/code/DRRT/teacher_weights`. This is the native 7B model, not the initially proposed ViT-B/16, DINOv2, or SAT weights. Dense extraction remains frozen at **384 × 384**. Compressor training caches are written by the shared `compression.extract` path below; `extract.py` remains a diagnostic native NPZ export.

## Environment and storage

Run local commands from `/home/engeld/Work/compressing-latents`:

```bash
/usr/bin/python3 -m venv .venvs/dinov3
.venvs/dinov3/bin/python -m pip install -r experiments/dinov3/requirements.txt
```

The isolated environment uses Python 3.10, CPU PyTorch 2.7.1, and pinned required dependencies. Runtime files are local because the rclone mount does not support normal executable/symlink permissions; no global packages are changed. Data/checkpoints/features remain shared.

| Asset | Default location |
| --- | --- |
| Existing ImageNet source (read-only) | `/home/engeld/mnt/datawaha/IN1K/data/*.parquet` |
| Shared image pilot | `/home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot` |
| Authorized native checkpoint/source | `/home/engeld/mnt/datawaha/compressing-latents/checkpoints/dinov3/vit7b16-lvd1689m` |
| Features | `/home/engeld/mnt/datawaha/compressing-latents/features/dinov3` |

The model checkpoint alone is **26,864,407,683 bytes** (about 25 GiB in float32); float32 execution needs that plus activations/runtime memory. Native meta-device construction and `torch.load(weights_only=True, mmap=True)` followed by strict `assign=True` loading avoid a second full randomly initialized model copy. `--compute-dtype bfloat16` halves parameter storage after loading and is useful on a capable GPU; it changes compute precision, not the normalization boundary. CPU float32 is supported but the 7B forward is substantially more expensive than ViT-B. Main owns all smoke scheduling.

## Shared ImageNet pilot

```bash
.venvs/dinov3/bin/python experiments/image_data.py
```

Preparation reads original embedded image bytes from authorized local parquet shards. It downloads no dataset and never changes originals. Materialization succeeded:

| Split | Source shards / rows | Source parquet bytes | Pilot images / image bytes |
| --- | --- | --- | --- |
| train | 294 / 1,281,167 | 146,477,630,673 | 10,000 / 1,158,646,132 |
| validation | 14 / 50,000 | 6,693,093,726 | 1,000 / 134,325,794 |

The project pilot contains 1,292,971,926 image bytes, plus `train.jsonl`, `validation.jsonl`, and `inventory.json`. Manifest hashes: train `bc7f93679912bc661a5948c756a3b121f9f2a15129e9371910ff1c0036f2a8ca`; validation `705557958de8ccb8b9d57ca888605583752eee9c8d77258ededb834e503f125c`.

For seed 20261014, shards are sorted by SHA256 of `seed/split/name`, the minimum sufficient shard prefix is chosen, and rows are uniformly selected within it using a seeded RNG. Train selects 00147, 00019, 00222; validation selects 00013. This avoids reading all encoded images and is **not a globally uniform or class-balanced ImageNet pilot**. Native train/validation membership and split-qualified IDs are preserved; Imagenette is never substituted.

Manifest rows contain `sample_id`, `split`, `paths:[absolute_image_path]`, numeric `label`, `original_size_wh`, original format/SHA256, and `source:{dataset,parquet,row}`. Original encoded bytes are Pillow-checked. Qwen consumes the same images/manifests unchanged. Re-running preparation verifies existing files against the source and resumes missing images. Source/count/output/seed overrides are explicit; insufficient source rows fail rather than silently reducing counts. The original [ImageNet Terms of Access](https://huggingface.co/datasets/ILSVRC/imagenet-1k) remain applicable.

## Authorized checkpoint and official source preparation

The user-authorized original is:

```text
KW61627:/code/DRRT/teacher_weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

It is copied through Worker Harness to the shared worker-visible `/data/shared/datawaha/compressing-latents/checkpoints/dinov3/vit7b16-lvd1689m/`, with the original untouched and destination integrity checked. Workstation-visible paths map to `/home/engeld/mnt/datawaha/compressing-latents/...`. `native-asset.json` records authorized copy provenance and full SHA256.

The completed authorized copy is **26,864,407,683 bytes**, and both original/copy hashes matched SHA256 `a955f4ea3bec4fcd666bf363630da4386383069b482c8a927e17a3e1154965b7`. The pinned official source archive SHA256 is `94da41da596633a73619078bb3bbde64ff3be95c55e8658638fa57c542cc4fe7`.

```bash
.venvs/dinov3/bin/python experiments/dinov3/prepare.py
```

This validates the already-copied checkpoint, downloads **only public official source** from [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) at immutable revision `6876159a11b4df116f30f667f8c9888617df0751`, and records source archive/checkpoint hashes in `asset.json`. It safely inspects tensor metadata with `weights_only=True` and compares every key/shape against the native official `dinov3_vit7b16` constructor on the meta device; this is asset inspection, not a forward pass. An absent checkpoint raises an exact prerequisite error rather than downloading another model. `--source-checkpoint /authorized/local/file.pth` can safely copy an already-authorized local original; it never obtains unofficial mirrors or bypasses gates. The [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license) remains applicable.

No guessed Hugging Face configuration or conversion is used. Native architecture: 40 blocks, width 4096, 32 heads, 16-pixel patches, four storage/register tokens, SwiGLU64 with FFN ratio 3, bias-free QKV, LayerNorm epsilon 1e-5, axial RoPE with fp32 positional compute. The native constructor, checkpoint shape match, and exact source revision—not a hand-written approximation—define model loading.

Safe metadata-only inspection on KW61627 observed 610 state entries and 6,716,043,296 parameter/buffer elements, all float32. Actual tensor shapes include CLS `[1,1,4096]`, storage `[1,4,4096]`, patch convolution `[4096,3,16,16]`, first/final-block QKV `[12288,4096]`, and final norm `[4096]`. Its existing `/code/DRRT/.venv/bin/python` successfully performed this inspection with Torch `2.11.0+cu128` and CUDA available; no inference or GPU allocation was performed.

## Extraction and smoke commands

One-image local CPU smoke (Main executes):

```bash
OMP_NUM_THREADS=8 .venvs/dinov3/bin/python experiments/dinov3/extract.py \
  --image /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation/validation-00013-of-00014/000000.jpg \
  --shard-size 1 --device cpu --compute-dtype float32 \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/dinov3/smoke-explicit
```

One-image manifest smoke:

```bash
OMP_NUM_THREADS=8 .venvs/dinov3/bin/python experiments/dinov3/extract.py \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --limit 1 --shard-size 1 --device cpu \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/dinov3/smoke-validation
```

Repeat `--image` for a small explicit list. For streaming extraction, use `--manifest .../train.jsonl --limit 32 --shard-size 16 --output .../features/dinov3/train-32`; omit `--limit` only when intentionally processing the entire pilot. Memory remains bounded to one feature shard; source splits cannot mix within a shard and existing output shards are never overwritten.

A Worker Harness GPU runtime must have CUDA PyTorch plus torchvision, NumPy, and Pillow. The local requirements intentionally install CPU wheels; do not use them to overwrite an existing worker CUDA environment. Main verified KW60995 `/code/lc64-venv/bin/python` has Torch 2.7.1+cu126 and CUDA; Main must check/provision its remaining native imports and schedule the smoke. Example worker invocation after the scripts are transferred through Worker Harness:

```bash
/code/lc64-venv/bin/python /path/to/experiments/dinov3/extract.py \
  --checkpoint /data/shared/datawaha/compressing-latents/checkpoints/dinov3/vit7b16-lvd1689m \
  --image /data/shared/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation/validation-00013-of-00014/000000.jpg \
  --shard-size 1 --device cuda --compute-dtype bfloat16 \
  --output /data/shared/datawaha/compressing-latents/features/dinov3/smoke-worker
```

The code accepts explicit remote checkpoint/image/output roots. Default JSONL paths refer to the workstation, so use the worker-visible explicit image for remote smoke. These scripts reserve or launch no GPUs.

### Exact transforms, tensor boundary, and outputs

Decode, EXIF transpose, RGB; official torchvision v2 `ToImage`, bilinear direct-square `Resize((384,384), antialias=True)`, uint8-to-float32 `/255`, ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)`. No crop, padding, or random augmentation; aspect ratio is intentionally warped by the [official square-resize transform family](https://github.com/facebookresearch/dinov3#image-transforms). Original/oriented/processed dimensions are saved.

Frozen `eval()` execution under `torch.inference_mode()` selects **native `forward_features()['x_norm_patchtokens']` after final block 39 and learned `model.norm` LayerNorm**. CLS/storage tokens use native `x_norm_clstoken` / `x_storage_tokens`, also after `model.norm` in eval; the separate local-CLS norm is training-only. No L2 normalization, centering, scaling, whitening, or compressor-specific transform is added. Native norm epsilon, preprocessing, precision, source IDs/splits, source revision/checkpoint hash, and dependency versions are recorded in JSON.

Observed one-image numeric NPZ arrays from Main's validation smoke:

- `features`: `[576,4096]`, float16 storage by default (`--storage-dtype float32` supported).
- `positions`: `[576,3]` integer `(time-or-view=0,row,col)`, row-major 24 × 24 grid indices.
- `sample_offsets`: `[0,576]`; concatenated multi-image features are delimited by offsets.
- `cls`: `[1,4096]`; `registers`: `[1,4,4096]`, separate from spatial features.

Each `shard-000000.npz` has adjacent `shard-000000.json`. Load arrays with `allow_pickle=False`; metadata is JSON, never unsafe pickle. Main ran the manifest smoke with `OMP_NUM_THREADS=16`, CPU float32 compute, and float16 storage in 679.43 seconds. Patch, CLS, and register arrays were finite; exact 24×24 grid ordering, sample offsets, validation split, final layer index, and checkpoint digest metadata were verified. Outputs are under `features/dinov3/smoke-validation`; choose a fresh directory to repeat extraction. No compressor training or GPU job was launched.

## Training feature cache (`compression.extract`)

`python -m compression.extract --model dinov3` writes the compressor-training cache through `compression.cache.FeatureCacheWriter`. It reuses this directory's `load_model` (meta construction, `weights_only=True` mmap load, strict `assign=True`, one cast/move), `make_transform` and `load_image`, so the transform and tensor boundary above are unchanged. The model is loaded once per job; `--workers` threads decode/preprocess ahead of the GPU and `--batch-size` images share one `forward_features` call. `--compute-dtype bfloat16` keeps the native extractor policy of casting weights and pixels; CPU runs are float32 only. Storage is float16 and the writer rejects non-finite or float16-overflowing values.

Each record holds **581 tokens × 4096** in native order: CLS (type 1, position `(0,0,0)`), four registers (type 2, `(0,0,ordinal)`), then 576 patches (type 0, `(0,row,col)` row-major) on grid `(1,24,24)`. EXIF transpose stays part of the approved DINO boundary (Qwen does not transpose; this difference is recorded in both caches' provenance rather than harmonized).

The cache directory is `<root>/features/cache/dinov3/<identity>`. The identity hashes the checkpoint SHA256, source revision, layer/boundary, preprocessing, compute dtype, device type, float16 storage, package versions, manifest **content** SHA256 values and the selected sample IDs/splits. Manifest path mapping does not change it. `--resume` reopens only a cache with the same identity and recomputes only records that are not verifiably committed; it never trusts existing filenames. The full pilot (10,000 train + 1,000 validation) is about 52 GB of float16 features.

Local CPU float32 smoke, two records per split (from the repository root):

```bash
.venvs/dinov3/bin/python -m compression.extract --model dinov3 \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/train.jsonl \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --limit-per-split 2 --batch-size 2 --device cpu --threads 16
```

Full worker cache (KW60995; staged repository as working directory; small `requirements-extraction-worker.txt` overlay on `PYTHONPATH`, which DINO itself does not need). `--root` maps the manifests' workstation project prefix to the worker mount:

```bash
PYTHONPATH=/dev/shm/compressing-latents-deps /code/lc64-venv/bin/python -m compression.extract --model dinov3 \
  --root /data/shared/datawaha/compressing-latents \
  --manifest /data/shared/datawaha/compressing-latents/data/images/imagenet1k-pilot/train.jsonl \
  --manifest /data/shared/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --device cuda --compute-dtype bfloat16 --batch-size 16 --workers 8 --resume
```
