#!/usr/bin/env bash
# Run a project script inside the CUDA container that already carries our stack.
#
# This VM's system Python has no pip and no ensurepip, so instead of building a venv
# we reuse one of the pre-pulled Apheris Foundry images. The ESMFold2 image happens to
# ship exactly what we need:
#
#   Python 3.12.16, torch 2.15.0+cu132, transformers 5.22.2, tokenizers 0.23.3,
#   numpy 2.5.4, pandas 3.1.1, tqdm 4.70.1, scipy, scikit-learn, matplotlib, safetensors
#
# Nothing is installed and nothing is downloaded: the image is already on disk.
#
# Usage:
#   env/run.sh scripts/finetune.py --epochs 12 --max-hours 1.5
#   env/run.sh -c 'import torch; print(torch.cuda.get_device_name(0))'
#   IMAGE=... env/run.sh ...          # override the image
#   DOCKER_FLAGS='-it' env/run.sh ... # extra docker flags (e.g. interactive)
set -euo pipefail

IMAGE="${IMAGE:-quay.io/apheris/oslo-hackathon:apheris-esmfold2-v1.4.0}"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# HF_HOME/TRANSFORMERS_OFFLINE keep transformers from reaching for the network: every
# weight we load is a local path.
exec docker run --rm \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${PROJECT}:/work" \
  -w /work \
  -e PYTHONPATH=/work \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/work/.cache/huggingface \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TQDM_MININTERVAL="${TQDM_MININTERVAL:-2}" \
  ${DOCKER_FLAGS:-} \
  --entrypoint python3 \
  "${IMAGE}" \
  "$@"
