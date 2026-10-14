# Qwen3.8-27B — frozen image vision tower only

This experiment prepares **the exact Qwen3.8-27B visual backbone**, not Qwen3.5 weights and not the 27B language model. Compressor-training feature caches are written by the shared `compression.extract` path below; `experiment.py extract` remains a diagnostic per-image export.

## Exact assets and provenance

- Official standalone conversion: [`timm/qwen3_vit_416m.qwen3_8_27b`](https://huggingface.co/timm/qwen3_vit_416m.qwen3_8_27b), revision `290f775dc6c2a7844b065855867ded2aa55a6a7c`.
- Official conversion records source [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, original visual-containing shard `model-00001-of-00018.safetensors`. That large mixed shard is **not downloaded**.
- Only the standalone pre-merger visual checkpoint is downloaded: `1,660,060,752` bytes, SHA256 `eda956958255e255333af183fb2c78b7501cc3ddbcf42dc6a2f8f0681de50a24`. It contains 415,006,704 float32 parameters, no trained classifier head, no language weights, and no spatial merger weights.
- This public checkpoint and its source are ungated and Apache-2.0. Their original license files and the official `conversion.json` are retained beside the weights. No access gate is bypassed.
- timm source is pinned to official commit [`12c4939be518a6d02d5d3b47b0b2bed49ad32312`](https://github.com/huggingface/pytorch-image-models/blob/12c4939be518a6d02d5d3b47b0b2bed49ad32312/timm/models/qwen3_vit.py), installed as `1.0.31.dev0`. The published conversion was made with timm 1.0.29 / torch 2.9.1+cu130; this local runtime uses torch 2.9.1+cpu, not that CUDA runtime.
- Official model documentation: <https://huggingface.co/docs/timm/main/models/qwen3-vit>.

The upstream architecture being named `qwen3_5` is intentional: the Qwen3.8 source config itself uses this model type. Preparation checks the exact source identity and visual config: depth 27, hidden width 1152, patch 16, temporal patch 2, spatial merge 2, output width 5120, no DeepStack visual indices. It verifies the full weight SHA256 and inspects safetensors shapes without running inference.

## Storage and environment

Run commands from the repository root. Datasets/checkpoints/features are on the shared mount; the Python runtime is deliberately local because the `fuse.rclone` mount cannot execute its mode-0644 Python copies or create ordinary venv symlinks.

```text
.venvs/qwen38/                                          local isolated Python 3.10 CPU runtime
/home/engeld/mnt/datawaha/compressing-latents/
  checkpoints/qwen38/visual/                            model.safetensors, config, conversion, license, card
  checkpoints/qwen38/upstream/                          source config, processor config, license only
  checkpoints/qwen38/provenance.json                    exact revisions, checksum, tensor shapes
  checkpoints/qwen38/dataset-inventory.json             actual shared manifest availability/counts
  data/images/imagenet1k-pilot/{train,validation}.jsonl   shared DINO-owned data; read-only to Qwen
  features/qwen38/                                      per-image feature outputs, only on explicit request
```

Preparation checks available checkpoint storage before transfers. Hugging Face downloads use the official client with resumable cache behavior; it does not silently replace the target with another model. Standard authenticated clients may use existing credentials, but this public checkpoint requires none. No image download or image extraction is performed by Qwen.

```bash
# Fresh runtime plus exact checkpoint and current shared-data inventory:
bash experiments/qwen38/prepare.sh

# Existing runtime: prepare assets/inventory only, no model instantiation or inference:
.venvs/qwen38/bin/python experiments/qwen38/experiment.py prepare
```

`environment-lock.txt` pins the resolved local runtime dependencies and official timm source revision. `prepare.sh` installs the two CPU wheels from the official PyTorch CPU index first, then installs that lock. There is no global Python mutation. The prepared local environment uses CPU extraction. A main-agent-managed Worker Harness environment with compatible CUDA-enabled torch/torchvision can use the same extractor with `--device cuda`; GPU jobs must be scheduled through Worker Harness.

## Dataset contract

The DINO experiment alone owns preparation of the shared deterministic ImageNet pilot (target 10,000 train images / 1,000 validation images), reusing authorized local IN1K data. Its JSONL rows contain `sample_id`, image-level `split`, and `paths: [absolute_image_path]`, plus label and source metadata. This extractor preserves every source row in the output metadata and never repartitions it. Missing manifests are explicitly recorded as unavailable by `prepare`; do not equate checkpoint availability with dataset availability. ImageNet's original dataset access/license restrictions still apply; the pilot is not a redistributed public download.

The prepared shared pilot was inventoried by Qwen without modifying it: exactly **10,000 train / 1,000 validation** rows, all referenced image files present. Manifest SHA256: train `bc7f93679912bc661a5948c756a3b121f9f2a15129e9371910ff1c0036f2a8ca`; validation `705557958de8ccb8b9d57ca888605583752eee9c8d77258ededb834e503f125c`. The first validation image is `validation/validation-00013-of-00014/000000.jpg` under the shared pilot root.

Inventory arbitrary already-existing compatible manifests with repeated `prepare --manifest PATH`. Extraction accepts exactly one image per manifest row; video clips/multiview samples are rejected rather than misrepresented as temporal inputs. Explicit image lists are supported independently of the pilot.

## Native tensor boundary and preprocessing

The primary target is **the final transformer block residual output, before `merger.norm`**, from `model.forward_features(x)` with `encoder_pool=''`. It is NHWC internally, flattened into spatial row-major `[N,1152]`. Native pre-norm inside each block remains; there is **no final encoder LayerNorm**, no pooling, output L2 normalization, standardization, classifier normalization, or compressor-specific transform. The classifier wrapper's `forward()` would produce pooled/normalized image embeddings and is deliberately never used.

At the default fixed 384×384 input:

| Quantity | Shape / semantics |
|---|---|
| RGB model input | `[1,3,384,384]`, float32 |
| Logical original temporal grid | `[1,24,24]` |
| Pre-merger output | `[576,1152]`, after block index 26 (27th block) |
| Positions | `[576,3]` int32: `(temporal_group=0,row,col)` |
| Special tokens | `[0,1152]`; vision tower has no CLS/register token |
| Native post-merger, **not extracted** | `[144,5120]`: per-patch norm, 2×2 concatenation, two-layer native projector |

The official conversion sums the original Conv3d's two temporal slices in float64 into a Conv2d kernel. The upstream still-image path repeats the **same image twice**, making this folded image patch embedding mathematically equivalent to that still-image input path. It is not a two-frame video model, and the output must not be assigned two different timestamps. Axial 2D RoPE and learned absolute-position-grid interpolation are preserved. Compared with upstream, the timm raw sequence has ordinary spatial row-major ordering rather than the upstream spatial-merge-grouped permutation.

Images are decoded with Pillow and converted to RGB (no EXIF transpose), bicubically squashed to the selected square size without crop, rescaled by 1/255, then normalized with checkpoint-native mean/std `(0.5,0.5,0.5)`. This follows the published timm conversion's `crop_mode='squash', crop_pct=1.0`, with **an explicit experimental size override from its default 768 to 384**. The original multimodal processor dynamically resizes while preserving aspect ratio and constraining pixel count; this fixed-square experiment is not falsely presented as identical to its dynamic resizing policy. The normalization, patch embedding, positional interpolation and chosen raw feature boundary remain native. Any `--size` must be divisible by 32.

## Main-agent inference smoke (not run during preparation)

Use a fresh output directory, because existing indices/sample files are never overwritten:

```bash
.venvs/qwen38/bin/python experiments/qwen38/experiment.py extract \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --split validation --limit 1 --size 384 --device cpu --threads 4 \
  --output-dir /home/engeld/mnt/datawaha/compressing-latents/features/qwen38/smoke384
```

Or use the following already-existing imagenette validation image for an independent explicit-input smoke (repeat `--image` for a small list). This reads the original file in place; it does **not** replace the shared ImageNet pilot or download a second dataset:

```bash
.venvs/qwen38/bin/python experiments/qwen38/experiment.py extract \
  --image /home/engeld/mnt/datawaha/imagenette2/val/n01440764/ILSVRC2012_val_00009111.JPEG --limit 1 \
  --size 384 --save-dtype float32 --device cpu \
  --output-dir /home/engeld/mnt/datawaha/compressing-latents/features/qwen38/explicit384
```

Both `--checkpoint-dir` and `--output-dir` can be absolute overrides. On a Worker Harness machine where the shared mount is `/data/shared/datawaha`, use `--checkpoint-dir /data/shared/datawaha/compressing-latents/checkpoints/qwen38`, explicit remote `--image` paths and a remote `--output-dir`; local manifests intentionally retain workstation absolute paths. The published local lock installs CPU torch; a main-agent-managed GPU environment must instead supply compatible CUDA-enabled torch/torchvision wheels.

Observed output from the one-image validation smoke (`--threads 8`, CPU float32 compute, 60.51 seconds):

- One hash-named `.npz` with numeric arrays `features [576,1152]`, `positions [576,3]`, `special_tokens [0,1152]`; default stored features float16, float32 compute. Use `--save-dtype float32` to avoid output down-casting. Load with `np.load(path, allow_pickle=False)`.
- Matching `.json` with source row/ID/split, original dimensions, full preprocessing, exact model/source/timm revisions, checkpoint hash, layer/tensor/normalization boundary, temporal-image interpretation, token grid/order and shapes.
- Streaming `index.jsonl` linking the sample ID and split to those files; one image is kept in flight at a time. Default `--limit 1` prevents accidental full-dataset caching.
- Terminal JSON reporting `samples: 1` and `features_per_sample: [576,1152]`.

Main verified finite float16 features, exact 24×24 position ordering, empty special tokens, and validation metadata in `features/qwen38/smoke384`. No compressor training or GPU job was launched. To repeat extraction, choose a fresh output directory.

## Training feature cache (`compression.extract`)

`python -m compression.extract --model qwen38` writes the compressor-training cache through `compression.cache.FeatureCacheWriter`. It reuses `experiment.py`'s `load_model` (strict float32 safetensors load), `make_transform` (timm squash at the fixed **384** size) and `load_image` (Pillow RGB, no EXIF transpose), so the boundary above is unchanged: block 26 residual output before `merger.norm`. The model is loaded once; `--workers` threads preprocess ahead and `--batch-size` images share one `forward_features` call. `--compute-dtype bfloat16` uses `torch.autocast` over float32 weights on CUDA; CPU is float32 only. Storage is float16.

Each record holds **576 patch tokens × 1152** (type 0, `(0,row,col)` row-major, grid `(1,24,24)`); the tower has no CLS/register tokens. Unlike DINO, no EXIF transpose is applied; both policies are recorded in provenance and intentionally left at their approved boundaries.

The cache directory is `<root>/features/cache/qwen38/<identity>`; the identity hashes checkpoint SHA256, model/source/timm revisions, boundary, preprocessing, compute dtype, device type, storage dtype, package versions, manifest content SHA256 and selected sample IDs. `--resume` continues only that identity and skips only verifiably committed records. The full pilot is about 15 GB of float16 features.

Local CPU float32 smoke:

```bash
.venvs/qwen38/bin/python -m compression.extract --model qwen38 \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/train.jsonl \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --limit-per-split 4 --batch-size 4 --device cpu --threads 8
```

Full worker cache (KW60995; staged repository as working directory; `requirements-extraction-worker.txt` overlay supplies the pinned timm commit and safetensors):

```bash
PYTHONPATH=/dev/shm/compressing-latents-deps /code/lc64-venv/bin/python -m compression.extract --model qwen38 \
  --root /data/shared/datawaha/compressing-latents \
  --manifest /data/shared/datawaha/compressing-latents/data/images/imagenet1k-pilot/train.jsonl \
  --manifest /data/shared/datawaha/compressing-latents/data/images/imagenet1k-pilot/validation.jsonl \
  --device cuda --compute-dtype bfloat16 --batch-size 64 --workers 8 --resume
```
