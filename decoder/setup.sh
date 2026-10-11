#!/usr/bin/env bash
# Fetch the decoder weights into ./weights/. Run once, from the decoder/ folder.
#
#   weights/
#     sd-turbo/{model_index.json, vae/, unet/}   <- public, downloaded automatically
#     AEIC_SE_ft2.pkl                             <- project checkpoint (see below)
#     halfDecoder.ckpt                            <- project checkpoint (see below)
#
# sd-turbo is public on Hugging Face and is fetched here (vae+unet subfolders only,
# ~5 GB instead of the full 13 GB). The two AEIC checkpoints are project artifacts:
# either get them straight from the package author, or set the Google Drive file IDs
# below and this script will download them with gdown.

set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
WEIGHTS="$PWD/weights"
mkdir -p "$WEIGHTS"

# --- OPTIONAL: fill these in to auto-download the two project checkpoints ---------
AEIC_CKPT_GDRIVE_ID=""      # Google Drive *file* id for AEIC_SE_ft2.pkl
HALFDECODER_GDRIVE_ID=""    # Google Drive *file* id for halfDecoder.ckpt
# (The checkpoints live in Drive folder 1vioCW4EIxQiuLkWHKj7xbMi7WAVcWqJI; grab the
#  per-file id from the share link, or just copy the files in by hand.)
# ---------------------------------------------------------------------------------

PY="${PYTHON:-python}"

echo "==> [1/3] sd-turbo (vae + unet) from Hugging Face -> weights/sd-turbo"
if [ -f "$WEIGHTS/sd-turbo/model_index.json" ]; then
  echo "    already present, skipping."
else
  "$PY" - "$WEIGHTS" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1] + "/sd-turbo"
snapshot_download(
    repo_id="stabilityai/sd-turbo",
    local_dir=dest,
    allow_patterns=["model_index.json", "vae/*", "unet/*"],
)
print("    sd-turbo ->", dest)
PY
fi

echo "==> [2/3] AEIC_SE_ft2.pkl -> weights/AEIC_SE_ft2.pkl"
if [ -f "$WEIGHTS/AEIC_SE_ft2.pkl" ]; then
  echo "    already present, skipping."
elif [ -n "$AEIC_CKPT_GDRIVE_ID" ]; then
  "$PY" -m gdown "$AEIC_CKPT_GDRIVE_ID" -O "$WEIGHTS/AEIC_SE_ft2.pkl"
else
  echo "    NOT FOUND. Copy AEIC_SE_ft2.pkl (~457 MB) into weights/, or set"
  echo "    AEIC_CKPT_GDRIVE_ID at the top of this script and re-run."
fi

echo "==> [3/3] halfDecoder.ckpt -> weights/halfDecoder.ckpt"
if [ -f "$WEIGHTS/halfDecoder.ckpt" ]; then
  echo "    already present, skipping."
elif [ -n "$HALFDECODER_GDRIVE_ID" ]; then
  "$PY" -m gdown "$HALFDECODER_GDRIVE_ID" -O "$WEIGHTS/halfDecoder.ckpt"
else
  echo "    NOT FOUND. Copy halfDecoder.ckpt (~377 MB) into weights/, or set"
  echo "    HALFDECODER_GDRIVE_ID at the top of this script and re-run."
fi

echo
echo "Done. Verify weights/ contains: sd-turbo/  AEIC_SE_ft2.pkl  halfDecoder.ckpt"
ls -la "$WEIGHTS"
