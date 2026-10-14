#!/usr/bin/env bash
# Run with bash experiments/vggt/setup.sh; never place an executable venv on fuse.rclone.
set -euo pipefail
REPO=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
ENV="$REPO/.venvs/vggt"
SOURCE=/home/engeld/mnt/datawaha/compressing-latents/checkpoints/vggt/omega_source
REV=48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
python3 -m venv "$ENV"
"$ENV/bin/python" -m pip install pip==25.3
"$ENV/bin/python" -m pip install -r "$REPO/experiments/vggt/requirements.lock" --extra-index-url https://download.pytorch.org/whl/cpu
if [ ! -d "$SOURCE/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/vggt-omega.git "$SOURCE"
    git -C "$SOURCE" checkout "$REV"
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$REV" ]; then
    echo "Existing source differs from the pinned revision; refusing to modify it" >&2
    exit 1
fi
"$ENV/bin/python" -m pip install --no-deps --no-build-isolation "$SOURCE"
