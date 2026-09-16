#!/usr/bin/env python3
"""Standalone Wan2.2-TI2V-5B + LanPaint video inpainting.

The script reuses the official Wan-Video/Wan2.2 model definitions and does not
depend on VideoX-Fun or ComfyUI. Input masks default to the user's convention:
black pixels regenerate, white pixels keep.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


DEFAULT_MODEL_PATH = "/mnt/DataPart/jianghongda/VideoX-Fun/models/Diffusion_Transformer/Wan2.2-TI2V-5B"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--wan22_path", default=None, help="Official Wan-Video/Wan2.2 source checkout")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--first_frame", default=None)
    parser.add_argument("--prompt", required=True, help="Prompt text or path to a UTF-8 text file")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lanpaint_steps", type=int, choices=range(0, 21), default=1)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--lanpaint_cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; input FPS is used when omitted")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--white_is_regenerate", action="store_true", help="Invert the default black=regenerate convention")
    parser.add_argument("--lanpaint_lambda", type=float, default=5.0)
    parser.add_argument("--lanpaint_step_size", type=float, default=0.15)
    parser.add_argument("--lanpaint_beta", type=float, default=1.0)
    parser.add_argument("--lanpaint_friction", type=float, default=15.0)
    parser.add_argument("--lanpaint_early_stop", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--compare", action="store_true", help="Generate matched LanPaint 0/1/2 results in one run")
    parser.add_argument("--no_exact_composite", action="store_true", help="Do not restore keep pixels after decoding")
    return parser.parse_args()


def find_wan22_source(model_path: Path, explicit: str | None) -> Path:
    candidates = [] if explicit is None else [Path(explicit)]
    candidates.extend([
        Path("/mnt/DataPart/jianghongda/related_work/Wan2.2"),
        model_path,
        *model_path.parents,
    ])
    for candidate in candidates:
        if (candidate / "wan/textimage2video.py").is_file() and (candidate / "wan/configs/wan_ti2v_5B.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the official Wan2.2 source tree. Clone https://github.com/Wan-Video/Wan2.2 and pass --wan22_path."
    )


def read_prompt(value: str) -> str:
    path = Path(value)
    return path.read_text(encoding="utf-8").strip() if path.is_file() else value.strip()


def save_video(tensor: torch.Tensor, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor[0].permute(1, 2, 3, 0).detach().float().cpu().clamp(0, 1).numpy()
    imageio.mimsave(path, (frames * 255.0).round().astype(np.uint8), fps=fps, macro_block_size=1)


def load_pipeline(model_path: Path, device: torch.device, dtype: torch.dtype):
    from wan.configs import WAN_CONFIGS
    from LanPaint.wan22_official import Wan22OfficialLanPaint

    return Wan22OfficialLanPaint(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(model_path),
        device_id=device.index or 0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=dtype != WAN_CONFIGS["ti2v-5B"].param_dtype,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    wan22_source = find_wan22_source(model_path, args.wan22_path)
    sys.path.insert(0, str(wan22_source))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from LanPaint.wan22_runtime import read_video_pair

    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16 for Wan2.2-TI2V-5B")
    if args.num_frames < 1:
        raise ValueError("--num_frames must be positive")

    video, regenerate_mask, input_info = read_video_pair(
        args.video,
        args.mask,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        black_is_regenerate=not args.white_is_regenerate,
        threshold=args.mask_threshold,
    )
    if args.first_frame:
        first = Image.open(args.first_frame).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
        first_tensor = torch.from_numpy(np.asarray(first).copy()).permute(2, 0, 1).float() / 255.0
        video[0, :, 0] = first_tensor
        regenerate_mask[0, :, 0] = 0

    prompt = read_prompt(args.prompt)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pipeline = load_pipeline(model_path, device, dtype)
    output = Path(args.output).resolve()
    fps = args.fps or input_info.fps
    variants = [0, 1, 2] if args.compare else [args.lanpaint_steps]
    base_stem = re.sub(r"_lp\d+$", "", output.stem)
    regen = (regenerate_mask >= 127.5).to(video.dtype)
    mask_rgb = regen.repeat(1, 3, 1, 1, 1)
    all_results = []

    for variant in variants:
        variant_started = time.time()
        generated, diagnostics = pipeline.inpaint(
            prompt=prompt,
            video=video,
            regenerate_mask=regenerate_mask,
            sampling_steps=args.steps,
            guide_scale=args.cfg,
            lanpaint_steps=variant,
            lanpaint_cfg=args.lanpaint_cfg,
            lanpaint_lambda=args.lanpaint_lambda,
            lanpaint_step_size=args.lanpaint_step_size,
            lanpaint_beta=args.lanpaint_beta,
            lanpaint_friction=args.lanpaint_friction,
            lanpaint_early_stop=args.lanpaint_early_stop,
            shift=args.shift,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )
        raw_result = generated.unsqueeze(0).add(1.0).div(2.0).clamp(0, 1).cpu()
        result = raw_result
        if not args.no_exact_composite:
            result = raw_result * regen + video * (1.0 - regen)
        variant_output = output if len(variants) == 1 else output.with_name(f"{base_stem}_lp{variant}{output.suffix}")
        save_video(result, variant_output, fps)
        comparison = torch.cat([video, mask_rgb, raw_result, result], dim=-1)
        comparison_path = variant_output.with_name(f"{variant_output.stem}_comparison{variant_output.suffix}")
        save_video(comparison, comparison_path, fps)
        metadata = vars(args) | {
            "actual_lanpaint_steps": variant,
            "prompt_text": prompt,
            "resolved_model_path": str(model_path),
            "resolved_wan22_path": str(wan22_source),
            "input_fps": input_info.fps,
            "input_frames": input_info.frames,
            "mask_semantics": "black=regenerate, white=keep" if not args.white_is_regenerate else "white=regenerate, black=keep",
            "elapsed_seconds": time.time() - variant_started,
            "comparison_layout": "input | regenerate mask | raw decode | exact composite",
            "comparison_output": str(comparison_path),
            "diagnostics": diagnostics,
        }
        variant_output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        all_results.append(result)
        print(f"Saved LP{variant}: {variant_output}")

    if len(all_results) > 1:
        grid = torch.cat([video, mask_rgb, *all_results], dim=-1)
        grid_path = output.with_name(f"{base_stem}_all_comparisons{output.suffix}")
        save_video(grid, grid_path, fps)
        print(f"Saved all comparisons: {grid_path}")
    print(f"Total elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
