# VGGT-Omega-1B-512 contextual multi-view features

This experiment uses the **existing, user-selected `vggt_omega_1b_512.pt`**, with its matching official **VGGT-Omega** architecture. It is not stock VGGT-1B. Data are the staged ScanNet++ DSLR scene-disjoint pilot plus a training-scale group manifest that references original RGB in place. Compressor-training caches are written by the shared `compression.extract` path; `extract.py` remains a one-group diagnostic export. No prediction head is executed.

## Exact checkpoint and runtime

- Reused original checkpoint, unchanged: `/home/engeld/mnt/datawaha/vggt_omega_1b_512.pt`.
- Official model: [`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega), variant **VGGT-Omega-1B-512**, revision `1041e80fc0e911235d3426b0a3d9a81075111a53`.
- Observed size: **4,576,706,117 bytes**. Observed SHA256: `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- **Authenticated official Hugging Face LFS metadata matches this exact size and digest.** Standard `HfApi.model_info(files_metadata=True)` used existing credentials; credential contents were not inspected. Public unauthenticated listings mask the digest. No checkpoint transfer, gate bypass, or unofficial mirror was needed.
- Official source: [`facebookresearch/vggt-omega`](https://github.com/facebookresearch/vggt-omega), pinned `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`, stored in project `checkpoints/vggt/omega_source/`.
- Provenance and architecture inspection: project `checkpoints/vggt/provenance.json`.
- Local executable CPU environment: repository `.venvs/vggt/`. The rclone mount cannot host normal executable venvs. Project data/checkpoints/features remain on datawaha; the original checkpoint is reused by path rather than duplicated.

Checkpoint inspection used **`torch.load(weights_only=True, mmap=True, map_location='cpu')`**. All **1,411 state entries** strictly matched the complete official `VGGTOmega(autocast=False)` architecture, including camera/dense heads, with no missing/unexpected keys or incompatible shapes. This was metadata/weight compatibility inspection, **not inference**. The extractor also verifies the checkpoint digest and strictly loads the aggregator subset. It does not permissively reinterpret Omega as stock VGGT.

Architecture distinction is material: Omega has ViT-L/16 `DinoVisionTransformer` embedding, **16 aggregator register tokens**, patch-token start **17**, and `inter_frame_blocks` mixing global/register attention. Stock VGGT's patch14/four-register/global-block configuration is not compatible. Native Omega patch embedding has four internal storage tokens; these are distinct from its sixteen aggregator registers. Prior stock experiment code/package has been replaced, not retained as a fallback.

### License and benchmark caveat

Omega weights/source use the **FAIR Noncommercial Research License**, including acceptable-use restrictions; see official [`LICENSE`](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/LICENSE). Upstream checkpoint access is manually gated; this preparation reuses the already-authorized local artifact.

The official [README](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/README.md) and [reproduction notice](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/reproduction.md) warn of **potential training-data contamination in the original 512 checkpoint** and recommend the 416 reproduction checkpoint for foundation-model benchmarking. We retain the exact user-selected 512 checkpoint, without substituting 416. This pilot can support scene-held-out compressor feature-reconstruction experiments; do **not** claim it demonstrates uncontaminated foundation-model or downstream benchmark performance.

## Prepared ScanNet++ pilot

Project data root: `/home/engeld/mnt/datawaha/compressing-latents/data/vggt/`.

Original RGB was traced from existing `scannet_relations/*/meta.json` to `/home/engeld/mnt/ibex_c2324/data/scannetpp/data/<scene_id>/dslr/resized_undistorted_images/`. Originals remain untouched. Only selected RGB files were staged in project `rgb/<scene_id>/`, with SHA256 checks against originals. Existing `scannetv2.tar.gz` was not unpacked, and `clip-centering` reconstructions were not substituted for photographs. Dataset is explicitly **ScanNet++**, not ScanNet v2.

Actual inventory: **1,006 source scene directories**. All **856 train / 50 validation** IDs from source `nvs_sem_train.txt`/`nvs_sem_val.txt` have present scene directories. RGB enumeration was limited to the selected 25 scenes: **23,906 RGB files**, **19,702 eligible source-partition frames**. The staged pilot is **20 training + 5 validation scenes, four views each: 100 RGB images, 33,139,506 bytes**, zero dataset download bytes and no scene shortfall. These counts do not claim that every unselected scene has inspected RGB.

Files:

- `inventory.json`: selected scene frame inventories, full scene-directory presence counts, split-list hashes, excluded candidates.
- `pilot_4views.jsonl`: actual staged absolute image paths, ordered source paths/IDs/digests, scene/view identities and split provenance.
- `pilot_summary.json`: complete scene list, actual counts, bytes, seed and shortfall.

Validation scenes: `acd95847c5`, `3864514494`, `5942004064`, `45b0dac5e3`, `cc5237fd77`. Training IDs are in the summary. Scene lists are checked disjoint. Within training scenes use original DSLR `train` filenames; within validation scenes use original DSLR `test` filenames. SHA256 ordering with seed `vggt-pilot-20261014` selects scenes and one deterministic ordered four-view window per scene, rank stride10 reduced when required. Ordering follows numeric capture filenames, not verified timestamps or guaranteed camera overlap. Manifests preserve sampling ranks/stride and source frame partitions. Never split one contextual group's views across experiment splits.

ScanNet++ data remains subject to its [approved-account access and Terms of Use](https://scannetpp.mlsg.cit.tum.de/scannetpp/). No dataset credential or license gate was bypassed.

## Training-scale group manifest

```bash
.venvs/vggt/bin/python experiments/vggt/prepare.py groups --workers 32
```

`groups` performs metadata-only preparation (listing, frame-split parsing, SHA256 and Pillow header check of every referenced view; no inference, copies or downloads). Scenes come from the official `nvs_sem_train.txt` / `nvs_sem_val.txt` lists in the same SHA256(`vggt-pilot-20261014:` + scene) order as the pilot, so the 20 + 5 pilot scenes are included. Each scene keeps its original DSLR frame partition (training scenes `train` frames, validation scenes `test` frames) in numeric filename order. Per scene, the rank stride is `min(10, (eligible - groups) // (views - 1))` and group starts are `round(linspace(0, eligible - window, groups))`, giving distinct ordered four-view groups spread over the scene; the first view is the reference. A scene is ineligible rather than padded when it cannot provide the requested distinct groups.

Materialized output under `data/vggt/groups/` (2026-10-14):

- `groups_4views.jsonl` — **3,200 train groups (200 scenes × 16) + 100 validation groups (25 scenes × 4)**, SHA256 `2bb969772e06407e2114ec95435fa88191e18a6a0f0c5ab21ae47c895dd3db3d`. Rows carry `sample_id` (`<scene>-v4-gNN`), `scene_id`, `split`, absolute `paths` under `source_root`, `source_relative_paths`, view IDs, per-view SHA256/bytes, frame partition/split file and the exact sampling ranks.
- `groups_inventory.json` — eligibility, frame counts and partition for every listed scene.
- `groups_summary.json` — counts, selected scene IDs, split-list hashes and shortfall.

Actual counts: all **856 train / 50 validation** listed scenes are eligible; **zero shortfall**. The groups reference **12,928 distinct RGB files, 4,301,981,618 bytes**, read in place from `/home/engeld/mnt/ibex_c2324/data/scannetpp` (no staging). Training groups all use stride 10; validation scenes have few `test` frames (minimum 9), so their strides range from 1 to 10. Groups within a scene may share individual views but no two groups are identical; no scene appears in both splits. The source root is not under the project root, so other machines must map it explicitly (`--path-map /home/engeld/mnt/ibex_c2324/data/scannetpp=<visible ScanNet++ root>`).

## Training feature cache (`compression.extract`)

`python -m compression.extract --model vggt` reuses `extract.py`'s `load_aggregator` (digest verification, meta construction, strict mmap `assign=True` load of the aggregator), official `load_images` (balanced mode) and `forward_group`. The aggregator is loaded once per job; each forward is one ordered group (`--batch-size 1`), with `--workers` threads preprocessing the next groups. `--compute-dtype bfloat16` uses CUDA autocast; CPU is float32. `--layers heads` (default) caches layers **4, 11, 17 and 23** from the same forward, one cache record per (group, layer); train layer 23 separately by selecting it in the reader.

Per view, tokens follow native aggregator order: camera (type 3, `(view,0,0)`), 16 registers (type 2, `(view,0,ordinal)`), then patches (type 0, `(view,row,col)` row-major), on grid `(views, rows, cols)`. For the 1752×1168 DSLR views this is 4 × (17 + 26 × 39) = **4,124 tokens × 2048** per layer, about 16.9 MB float16; the full 3,300-group four-layer cache is about **223 GB**. Per-view crop/resize/pad transforms and model input shape are recorded with each record.

Cache directory: `<root>/features/cache/vggt/<identity>`, with the identity covering checkpoint SHA256, model/source revisions, layers and boundary, preprocessing mode, compute dtype, device type, storage dtype, package versions, manifest content SHA256 and selected groups. `--resume` skips only verifiably committed (group, layer) records.

Local CPU float32 smoke (one group per split, all four head layers):

```bash
.venvs/vggt/bin/python -m compression.extract --model vggt \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vggt/groups/groups_4views.jsonl \
  --limit-per-split 1 --device cpu --threads 16
```

Full worker cache (KW60995; staged repository as working directory). `vggt_omega` is imported from the pinned `checkpoints/vggt/omega_source` checkout when it is not installed. The checkpoint lives one level above the project root, hence the extra datawaha mapping:

```bash
PYTHONPATH=/dev/shm/compressing-latents-deps /code/lc64-venv/bin/python -m compression.extract --model vggt \
  --root /data/shared/datawaha/compressing-latents \
  --path-map /home/engeld/mnt/datawaha=/data/shared/datawaha \
  --path-map /home/engeld/mnt/ibex_c2324/data/scannetpp=<worker-visible ScanNet++ root> \
  --manifest /data/shared/datawaha/compressing-latents/data/vggt/groups/groups_4views.jsonl \
  --device cuda --compute-dtype bfloat16 --workers 4 --resume
```

## Reproduce preparation

Run from repository root:

```bash
bash experiments/vggt/setup.sh
.venvs/vggt/bin/python experiments/vggt/prepare.py checkpoint
.venvs/vggt/bin/python experiments/vggt/prepare.py inventory
.venvs/vggt/bin/python experiments/vggt/prepare.py pilot
```

`checkpoint` **only verifies/reuses the existing Omega-512 checkpoint** and performs strict full-model weight compatibility inspection; it does not download or execute inference. Its `--checkpoint` override accepts another mounted path to the exact same verified asset. `inventory --source-root` and `pilot --inventory --output` override data locations. Pilot staging checks free space and never overwrites a differing file.

`requirements.lock` records resolved dependencies, including CPU Torch `2.6.0+cpu`, torchvision `0.21.0+cpu`, NumPy `1.26.4`, Pillow `11.3.0`. Official `vggt-omega` source installs separately at the pinned commit without dependency resolution. Setup refuses an existing source checkout at a different revision. No global Python mutation or GPU reservation occurs.

## Exact executable extraction smoke for Main

```bash
.venvs/vggt/bin/python experiments/vggt/extract.py \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vggt/pilot_4views.jsonl \
  --sample-id acd95847c5-v4 --layers last --mode balanced \
  --device cpu --precision float32 --threads 8 \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/vggt/omega-acd95847c5-v4
```

Equivalent explicit ordered paths, portable by overriding checkpoint/output/image roots:

```bash
.venvs/vggt/bin/python experiments/vggt/extract.py \
  --images \
  /home/engeld/mnt/datawaha/compressing-latents/data/vggt/rgb/acd95847c5/DSC00070.JPG \
  /home/engeld/mnt/datawaha/compressing-latents/data/vggt/rgb/acd95847c5/DSC00078.JPG \
  /home/engeld/mnt/datawaha/compressing-latents/data/vggt/rgb/acd95847c5/DSC00085.JPG \
  /home/engeld/mnt/datawaha/compressing-latents/data/vggt/rgb/acd95847c5/DSC00092.JPG \
  --scene-id acd95847c5 --split validation \
  --checkpoint /home/engeld/mnt/datawaha/vggt_omega_1b_512.pt \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/vggt/omega-acd95847c5-explicit \
  --layers last --mode balanced --device cpu
```

Paths remain ordered; the first is the reference view. Source photos were inspected as RGB **1752×1168**. Official Omega `balanced` preprocessing yields **`[1,4,3,416,624]`**. Main's actual CPU smoke observed patch grid **`[4,26,39]`**, dense **`[4056,2048]`**, positions **`[4056,3]`**, camera **`[4,2048]`**, and registers **`[4,16,2048]`**. For these same photos, `--mode max_size` is expected to yield `[1,4,3,336,512]`, grid `[4,21,32]`, and dense `[2688,2048]`; that mode was not exercised.

The local environment is CPU-only. The extractor accepts CUDA mixed precision in an appropriately provisioned CUDA runtime; GPU work must be parent-launched through Worker Harness. No inference was run by this preparation agent.

## Native feature and normalization boundary

Primary tensor is zero-based aggregator layer **23**, before head normalization/projection, shape **`[1,S,17+Gh*Gw,2048]`**. Channels concatenate **frame residual stream `[0:1024]` then inter-frame residual stream `[1024:2048]`**. Layer23 uses global inter-frame attention. Register-only inter-frame blocks are **2,6,9,14,20**; other blocks use global attention. No extra LayerNorm, whitening, pooling, unit normalization or compressor-specific transform is added. Features are jointly contextualized across the entire ordered view set.

Official `vggt_omega.utils.load_fn.load_and_preprocess_images` reads RGB (RGBA alpha-composited onto white), center-crops extreme height/width ratios into `[0.5,2.0]`, then bicubic-resizes using `image_resolution=512`, `patch_size=16`. Default **balanced** mode chooses rounded patch-grid dimensions with token budget near `(512/16)^2`; **max_size** constrains the longest side to512. Different transformed view sizes are symmetrically white-padded to a common size. No EXIF transpose occurs. JSON records exact native crop boxes, resize dimensions and group padding using the pinned implementation's shape helpers.

RGB is `[0,1]` at loader output. **Inside Aggregator**, native normalization is mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`. Patch embedding returns native **`x_norm_patchtokens`** before alternating attention. Saved concatenated outputs remain **before all dense/camera head normalization**. First ordered view uses first-frame camera/register embeddings; later views use shared other-frame variants. No poses or world-space coordinates are inferred here.

Per extracted layer, one small `.npz`/`.json` pair contains:

- `features [S*Gh*Gw,2048]`, default float16 storage (`--storage-dtype float32` optional).
- `positions [N,3]` int32 `[view_index,patch_row,patch_col]`, view-major then row/column-major. Token addresses, not metric3D positions.
- `camera_tokens [S,2048]` and **`register_tokens [S,16,2048]`**, separate from dense patches.
- JSON: ordered scene/view provenance, exact checkpoint/source revisions/digest, transforms, grid/shapes, native normalization/boundary, attention type, reference convention and compute/storage precision.

`--layers heads` exports verified Omega **DenseHead layers4,11,17,23**, preserving patches plus camera/register tokens at each. **Omega CameraHead consumes layer23 camera AND register tokens**, before its token normalization and head-local cross-frame trunk; saving only the camera token is not sufficient. One final patch layer is not claimed sufficient for dense depth. No point/DPT stock head is claimed or executed. Custom zero-based indices are accepted, e.g. `--layers 4,23`.

## Handoff evidence

Observed: official authenticated checkpoint digest match; safe full-state strict compatibility; pinned Omega package installed; original RGB reused and pilot copies integrity-checked. Main then ran real extraction with `--layers heads --mode balanced --device cpu --precision float32 --threads 16` in 40.87 seconds. Each of layers 4, 11, 17, and 23 had finite patch/camera/register arrays, exact view-row-column position order, and validation metadata in `features/vggt/omega-acd95847c5-v4`. Camera/dense heads, compressor training, and GPU execution were not run. Use a fresh output directory when repeating extraction.
