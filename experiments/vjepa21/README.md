# V-JEPA 2.1 ViT-B/16 frozen dense video features

This experiment prepares an exact encoder and video data. Compressor-training feature caches are written by the shared `compression.extract` path below with the pinned SSv2 evaluation sampler; `extract.py` remains a diagnostic per-clip export. Predictor and action/world-model inference are not implemented.

## Storage and environment

- Data: `/home/engeld/mnt/datawaha/compressing-latents/data/vjepa21`
- Checkpoints and pinned upstream source: `/home/engeld/mnt/datawaha/compressing-latents/checkpoints/vjepa21`
- Feature output: `/home/engeld/mnt/datawaha/compressing-latents/features/vjepa21`
- Runtime: `/home/engeld/Work/compressing-latents/.venvs/vjepa21` (Python 3.10; isolated CPU PyTorch wheels).

The shared mount does not support executable modes/symlinks needed by normal Python environments. Runtime dependencies therefore live locally; datasets/checkpoints remain on the shared mount. An unsuccessful environment created only for this experiment on the mount was removed. No pre-existing unrelated data was changed. `--root` changes the project mount root; `--output` changes the feature output location.

Recreate the pinned environment from the repository root:

```bash
/usr/bin/python3 -m venv .venvs/vjepa21
.venvs/vjepa21/bin/python -m pip install pip==25.3
.venvs/vjepa21/bin/python -m pip install -r experiments/vjepa21/requirements.txt
```

The installed package versions, including transitive dependencies, are pinned in `requirements.txt`. This environment is CPU-only; GPU execution requires a separately provisioned CUDA-enabled environment on a Worker Harness worker, not this CPU wheel environment.

Native evaluation transforms import OpenCV through `src.datasets.utils.video.functional`; `opencv-python-headless==4.12.0.88` is included explicitly. Static inspection of the complete evaluation-preprocessor import chain found no other missing mandatory third-party dependency. Its optional `skimage` import is inside unused `RandomRotation`, not the deterministic resize/center-crop evaluation pipeline.

## Exact checkpoint and provenance

[V-JEPA 2.1 was released on 2026-03-16](https://github.com/facebookresearch/vjepa2). The exact entry is **`vjepa2_1_vit_base_384`**, not a V-JEPA 2 model or a predictor/action-model checkpoint. Source is pinned to Git commit **`204698b45b3712590f06245fbfba32d3be539812`**. Its [Hub implementation](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py) selects **`ema_encoder`** for the distilled base model.

The [official README download](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt) is the release bundle (1,664,223,428 bytes), SHA256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`. Preparation verifies these recorded bytes/checksum, loads it with PyTorch `weights_only=True`, and exports **only its EMA encoder** to `ema_encoder.safetensors` (347,347,544 bytes; 158 tensors), SHA256 `f903a9a4a1daecc27ce8e7f32e3c3000c43a617f3405ba707afd6d19d96fb178`. The release bundle is retained as provenance, but extraction reads only the encoder-only safetensors file.

The pinned upstream Hub source has a testing URL `http://localhost:8300` instead of Meta's download base. We neither call that URL nor alter upstream source: preparation uses the verified official README URL and extraction constructs the identical native encoder without constructing a predictor, then loads encoder weights strictly.

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/prepare.py checkpoint
```

This prepares `source/`, `ema_encoder.safetensors`, and `provenance.json`. Existing source at another revision is rejected rather than overwritten. Repository code is MIT-licensed; retain upstream notices and review the official release terms for intended use.

## Dataset: complete Something-Something V2

The preferred [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) is available from [Qualcomm's official downloads page](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads). The static reader omits SPA download links; inspecting the rendered page revealed public Qualcomm asset GET routes. They require no account, personal information, signature, or explicit agreement click-through. `HEAD` returns 403, so preparation uses public range GET for metadata. The links displayed as `Something-something_zip1/zip2` currently return two **raw split TGZ parts**, not ZIP containers: **10,000,000,000 + 9,444,975,475 = 19,444,975,475 bytes**. The [current official instructions](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/20bn-something-something_download_instructions_-_091622-v2.pdf) describe 240-pixel-high VP9 WebM videos.

Preparation downloads these official parts, the official labels ZIP, the instructions and the [official research-use license](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf) into `data/vjepa21/ssv2/packages`, retaining original packages and notices. That license states that downloading, installing or using the data constitutes acceptance and limits use to internal non-profit research; public availability is not permission to redistribute data or use it commercially. Preparation does not perform a separate website agreement click-through or account registration, and requires or stores no credentials. Dataset assets and derived feature caches must remain outside the public code repository.

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/prepare.py ssv2-download
```

Transfers resume from `.partial` files and enforce expected byte lengths; HTTP/1.1 avoids the observed HTTP/2 stream interruptions. Both complete official archives and all official labels are retained, covering **train (168,913)**, **validation (24,777)** and the original test partition. The two parts are streamed together for safe extraction without a duplicate concatenated archive. Existing Kinetics assets and all unrelated data remain untouched.

For feature caching, the user subsequently requested a **small subset**, not full-corpus features: preparation materializes only **10,000 train / 1,000 validation** standalone videos and manifests, selecting the lowest SHA256 ranks of `'vjepa21-subset-0:' + sample_id` within each unchanged official split. Full official archives and labels remain available; there is no synthetic fallback, resplitting, test-set inclusion or silent narrowing of the selected subset. Missing or duplicate IDs fail.

Outputs under `data/vjepa21/ssv2`:

- `videos/<official-id>.webm`: the selected 11,000 actual official train/validation videos.
- `labels/`: all official label-package files, including the 174-class mapping; the separate research-use license is retained in `packages/`.
- `subset-train.jsonl`, `subset-val.jsonl`, `subset.jsonl`: deterministic 10,000/1,000 feature-cache subset, preserving official IDs, templates, class IDs, text and placeholders, absolute video paths, source split, bytes and per-video SHA256.
- `subset-inventory.json`: selection seed, materialized counts, complete-source annotation counts and subset-manifest SHA256s.
- `inventory.json`: actual split counts and extracted bytes, source URLs/ETags/last-modified dates, package/annotation/manifest SHA256s and class-mapping provenance.
- `download-sources.json`: resumable download inventory, including hashes once transfers complete.

Hugging Face alternatives were inspected rather than assumed: [`HuggingFaceM4/something_something_v2`](https://huggingface.co/datasets/HuggingFaceM4/something_something_v2) and its `giu-alb` fork contain manual-download loaders, not videos; `emirgocen/Something-Something-v2` contains train/validation annotations only. [`morpheushoc/something-something-v2`](https://huggingface.co/datasets/morpheushoc/something-something-v2) contains 20 video parts and Qualcomm's restrictive research-use license, not an independent redistribution grant. Preparation uses the public **official Qualcomm** route, not a mirror to bypass access gates.

### Existing Kinetics-400 pilot

The previously prepared **Kinetics-400 pilot is retained**, using [CVDF's official distribution instructions](https://github.com/cvdfoundation/kinetics-dataset) and public `https://s3.amazonaws.com/kinetics/400/` videos/annotations. Underlying video rights and original-platform terms still apply; no videos are redistributed by this repository.

Prepare exactly **1,000 train / 100 validation clips**:

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/prepare.py kinetics
```

This downloads the official train/val annotation CSVs and bounded tar shards (one validation shard and at most two training shards), never the entire corpus. It hashes candidate filenames using `sha256('vjepa21-pilot-0:' + filename)` and selects deterministically. At most one clip is retained per YouTube source; **all official validation source IDs** are excluded from training, so source-video leakage is prevented even outside the selected validation subset. Split membership always comes from official annotation CSVs. Selected member bytes are copied safely, never using `extractall` or extracting over another project. Resumable transfers check free disk space and expected length. Failed bounded availability raises an error rather than silently shrinking the pilot or substituting clips.

Preparation completed with **1,100 actual clips** (1,000 train / 100 validation), using only official `train/part_0.tar.gz` (1,631,315,939 bytes) and `val/part_0.tar.gz` (1,546,665,596 bytes). Archive transfer total is **3,177,981,535 bytes**; annotation CSVs add **11,340,200 bytes**; selected extracted clips occupy **1,801,295,730 bytes**. No second training shard was needed. Full archive SHA256s are in `inventory.json`; manifest SHA256 is `9b899c647722e03fe50ac582588870dfb731e4d831adea4481ed71ab62b808b8`.

The explicit validation example below is a real official clip (`making tea`): 320×240 RGB video, 25 FPS, **250 frames successfully decoded** during asset inspection. This is decoder inspection only, not encoder inference; the whole pilot has not been exhaustively decoded.

Outputs:

- `data/vjepa21/pilot.jsonl`: actual absolute clip paths, clip IDs, source split, dataset, YouTube ID, class, source timestamps, archive path and bytes.
- `data/vjepa21/inventory.json`: selection/provenance, official shard URLs, SHA256s, downloaded bytes, extracted bytes and counts.
- `data/vjepa21/clips/{train,val}/*.mp4`: only selected actual official videos.

To index an existing authorized SSv2 corpus without copying or modifying originals:

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/prepare.py ssv2 \
  --video-root /absolute/path/to/authorized/ssv2/videos \
  --train-labels /absolute/path/to/train.json \
  --val-labels /absolute/path/to/validation.json \
  --class-labels /absolute/path/to/labels.json
```

The existing-corpus importer indexes **every** supplied official train/validation annotation into `train.jsonl`, `val.jsonl` and `manifest.jsonl`, and additionally creates the deterministic subset manifests. There are no `--train`/`--val` pilot limits: every referenced video must exist. The normal `ssv2-download` workflow does **not** create full-corpus manifests claiming nonexistent standalone video paths. For temporal feature-cache preparation, the pinned [`configs/eval_2_1/vitb-384/ssv2.yaml`](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitb-384/ssv2.yaml) specifies **16 frames per clip, frame step 4, resolution 384, two temporal segments and three spatial views**. Recommend matching its 16-frame/step-4 temporal policy with upstream short-video padding, explicitly recording whichever deterministic segment/view is retained for caching. Hub model construction with `num_frames=64` does not mean the base SSv2 evaluation uses 64 frames.

## Tensor and native preprocessing boundary

The native Hub architecture uses **64 RGB frames at 384×384**, tubelet size 2, spatial patch size 16, width 768, producing **18,432×768** dense tokens on a **32×24×24** grid. For the bounded pilot, extraction defaults to **16 frames at 384×384**. Main's real CPU smoke confirmed **4,608×768** features on an **8×24×24** grid. This is an explicit temporal-length deviation, not a different checkpoint. The upstream video tokenizer and RoPE attention calculate temporal dimensions from the input. `--frames 64` restores the native default; that length was not exercised.

The encoder is instantiated with native `num_frames=64`, no register/CLS tokens, exactly matching the Hub encoder construction. Frozen output is the **12th/final block (index 11), after checkpoint-native `norms_block[-1]` LayerNorm, epsilon 1e-6**, as returned by `VisionTransformer.forward(training=False)`. No additional L2 normalization, standardization, pooling or compressor-specific transform is applied.

The pinned upstream `vjepa2_preprocessor(crop_size=384)` is called directly: bilinear resize of the short side to **438** (`int(384*256/224)`), center crop 384×384, RGB uint8 divided by 255, then channel mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`. There is no random augmentation or horizontal flip.

Temporal sampling in the diagnostic `extract.py` export is separately explicit: decode all presentation-order frames to count them without retaining the video; select `round(linspace(0, frame_count-1, 16))`; decode a second pass retaining only selected RGB frames. Short clips repeat monotonically indexed frames. This full-clip uniform policy is recorded, not falsely described as native 64-frame training sampling. Actual source frame indices, timestamps, average FPS and decoded geometry are saved. Tubelet `t` covers selected input slots `2*t` and `2*t+1`. Outputs made with this policy are legacy smoke artifacts, not training-cache inputs; the cache below uses the native SSv2 evaluation sampler and view instead.

Each clip yields an uncompressed `.npz` containing:

- `features[N,768]`: float16 by default, optionally float32; encoder computes float32.
- `positions[N,3]`: int32 `(tubelet,row,col)` coordinates in native time-major, row-major, column-major flatten order.
- `special_tokens[0,768]`: empty because this encoder has none.

The separate `.json` records source IDs/split/path, exact revision and checkpoint hashes, preprocessing, temporal sampling, boundary/layer, input/token shapes, grid, dtype and package versions. Load NPZ using `numpy.load(path, allow_pickle=False)`; metadata never uses pickle. Output filenames are the first 24 hex digits of SHA256(sample_id). Existing artifacts cause an error rather than silently overwriting a run.

## Smoke commands for Main (not executed during preparation)

From the repository root, CPU smoke on a real pilot validation clip:

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/extract.py \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vjepa21/pilot.jsonl \
  --split val --frames 16 --max-samples 1 --device cpu --threads 4 \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/vjepa21/smoke-manifest
```

For any explicit authorized clip, no dataset manifest is required:

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/extract.py \
  --clip /home/engeld/mnt/datawaha/compressing-latents/data/vjepa21/clips/val/--7qK_w-g3Y_000185_000195.mp4 \
  --sample-id k400:val:--7qK_w-g3Y_000185_000195 --split val --frames 16 --max-samples 1 \
  --device cpu --output /home/engeld/mnt/datawaha/compressing-latents/features/vjepa21/smoke-explicit
```

Observed smoke output: one printed JSON record, one NPZ with finite float16 features `[4608,768]`, exact time-row-column positions `[4608,3]`, special tokens `[0,768]`, and validation metadata. Main ran the manifest command with `--threads 8` in 7.66 seconds after adding the native preprocessor's required OpenCV dependency. The 16 sampled frame indices matched uniform endpoint-inclusive selection from 250 decoded frames. Outputs are in `features/vjepa21/smoke-manifest`; use a fresh directory to repeat extraction. With float16, feature data is 7,077,888 bytes before NPZ headers/positions.

Bounded shard extraction example:

```bash
.venvs/vjepa21/bin/python experiments/vjepa21/extract.py \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vjepa21/pilot.jsonl \
  --split train --num-shards 10 --shard-index 0 --max-samples 16 --offset 0 \
  --output /home/engeld/mnt/datawaha/compressing-latents/features/vjepa21/train-shard0-batch0
```

Shards assign each sample by SHA256(sample_id) modulo shard count, independent of manifest ordering. `--offset` skips records within that selected shard; use a fresh output directory for each bounded batch. The default maximum is 16 clips per invocation; no invocation automatically caches an entire dataset.

Worker Harness paths can use `--root /data/shared/datawaha/compressing-latents` and `--clip /data/shared/datawaha/compressing-latents/data/vjepa21/clips/val/<actual-file>.mp4` with a worker-provisioned runtime. Manifest absolute workstation paths are not silently rewritten. Main alone schedules GPU jobs through Worker Harness if desired. Preparation launched no inference smoke, tests, build, formatter, GPU jobs or compressor training.

## Training feature cache (`compression.extract`)

`python -m compression.extract --model vjepa21` writes the compressor-training cache through `compression.cache.FeatureCacheWriter`. It reuses `extract.py`'s `load_encoder` (pinned-source revision check, Hub-identical `vit_base` construction, strict EMA-encoder load), `count_frames` and `decode_frames`. The encoder is loaded once; `--workers` threads decode and preprocess clips ahead and `--batch-size` clips share one forward. `--compute-dtype bfloat16` uses `torch.autocast` over float32 weights on CUDA; CPU is float32 only. Tensor boundary is unchanged: block 11 after `norms_block[-1]`, **4,608 tokens × 768** per clip on grid `(8,24,24)`, all patch tokens (type 0, `(tubelet,row,col)`), no CLS/register tokens. The 10,000 + 1,000 SSv2 subset is about 78 GB of float16 features.

**Temporal sampler.** The pinned [`configs/eval_2_1/vitb-384/ssv2.yaml`](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/configs/eval_2_1/vitb-384/ssv2.yaml) states `frames_per_clip: 16`, `frame_step: 4`, `num_segments: 2`, `num_views_per_segment: 3`, `resolution: 384`; extraction re-checks these lines in the pinned checkout. `compression.extract.video_eval_clip_indices` implements the arithmetic of pinned `src/datasets/video_dataset.py` `VideoDataset.loadvideo_decord` for the evaluation loader's `allow_clip_overlap=True`, with `clip_len = 64` and `partition_len = frames // 2`:

- `partition_len > 64`: `linspace(0, 64, 16)` clipped to `[0, 63]`, truncated to int, plus `segment * partition_len`. Upstream's loader leaves `random_clip_sampling=True`, which only randomizes this window's end; the cache uses the upstream `random_clip_sampling=False` form (window ends at `clip_len`).
- otherwise: `sample_len = min(64, frames) - 1`; `linspace(0, sample_len, sample_len // 4)` padded with `sample_len` to 16 entries, clipped to `[0, sample_len - 1]`, truncated, plus `segment * ((frames - 64) // 1 if frames > 64 else 0)`. For clips of at most 64 frames both segments coincide, and the tail repeats the last valid frame (e.g. 30 frames → `0,4,9,14,19,24` then `28` ×10).

One segment is cached (`--video-segment 0` by default) instead of averaging both segments. Clips with fewer than two frames are rejected because upstream would index frame -1. Frame counts come from a full PyAV presentation-order decode (upstream uses decord `len(VideoReader)`). Each record stores decoded frame count, average FPS, branch, selected indices and timestamps.

**Spatial view.** The pinned `EvalVideoTransform(num_views_per_clip=3, short_side_size=384)` is called directly: bilinear (OpenCV) short-side resize to 384, three crops along the long side at `i * ((long - 384) // 2)`, `/255`, ImageNet mean/std. The cache keeps only the middle (`i = 1`, center) view; each record stores the resized size, crop axis and crop start. This replaces the diagnostic export's Hub preprocessor (short side 438, center crop).

The cache directory is `<root>/features/cache/vjepa21/<identity>`; the identity covers encoder/source checkpoint SHA256, source revision, boundary, sampler and view parameters, compute dtype, device type, storage dtype, package versions (including OpenCV and PyAV), manifest content SHA256 and selected sample IDs. `--resume` skips only verifiably committed records. The manifests are the SSv2 subset files described in the dataset section (`split` `val` is stored as `validation`).

Local CPU float32 smoke:

```bash
.venvs/vjepa21/bin/python -m compression.extract --model vjepa21 \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vjepa21/ssv2/subset-train.jsonl \
  --manifest /home/engeld/mnt/datawaha/compressing-latents/data/vjepa21/ssv2/subset-val.jsonl \
  --limit-per-split 2 --batch-size 2 --device cpu --threads 8
```

Full worker cache (KW60995; staged repository as working directory; `requirements-extraction-worker.txt` overlay supplies PyAV, timm and safetensors; the worker's existing OpenCV is used):

```bash
PYTHONPATH=/dev/shm/compressing-latents-deps /code/lc64-venv/bin/python -m compression.extract --model vjepa21 \
  --root /data/shared/datawaha/compressing-latents \
  --manifest /data/shared/datawaha/compressing-latents/data/vjepa21/ssv2/subset-train.jsonl \
  --manifest /data/shared/datawaha/compressing-latents/data/vjepa21/ssv2/subset-val.jsonl \
  --device cuda --compute-dtype bfloat16 --batch-size 16 --workers 16 --resume
```
