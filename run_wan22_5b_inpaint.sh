#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES=4 python examples/wan22_5b_video_inpaint.py \
  --wan22_path /mnt/DataPart/jianghongda/related_work/Wan2.2 \
  --model_path /mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-TI2V-5B \
  --video gs_render.mp4 \
  --mask mask.mp4 \
  --first_frame image.jpg \
  --prompt prompt.txt \
  --output output_lp1.mp4 \
  --height 832 \
  --width 480 \
  --num_frames 81 \
  --steps 20 \
  --lanpaint_steps 1 \
  --compare \
  --compare_official \
  --lanpaint_cfg 5.0 \
  --cfg 5.0 \
  --seed 43
