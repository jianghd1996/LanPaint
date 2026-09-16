#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Short, matched LP0/LP1/LP2 test using the parameters embedded in LanPaint's
# demonstrated Wan2.2-TI2V-5B ComfyUI workflow. Run this before the 81-frame job.
CUDA_VISIBLE_DEVICES=4 python examples/wan22_5b_video_inpaint.py \
  --wan22_path /mnt/DataPart/jianghongda/related_work/Wan2.2 \
  --model_path /mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-TI2V-5B \
  --video gs_render.mp4 \
  --mask mask.mp4 \
  --first_frame image.jpg \
  --prompt prompt.txt \
  --output sanity_lp1.mp4 \
  --height 832 \
  --width 480 \
  --num_frames 21 \
  --steps 20 \
  --shift 8.0 \
  --lanpaint_steps 1 \
  --compare \
  --cfg 5.0 \
  --lanpaint_cfg 5.0 \
  --lanpaint_lambda 16.0 \
  --lanpaint_step_size 0.3 \
  --lanpaint_beta 1.0 \
  --lanpaint_friction 1.0 \
  --lanpaint_early_stop 0 \
  --seed 43
