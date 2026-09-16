#!/usr/bin/env python3
"""Standalone Wan2.2-TI2V-5B + LanPaint video inpainting.

The script reuses the model definitions shipped by VideoX-Fun but does not
start or depend on ComfyUI. Input masks default to the user's convention:
black pixels regenerate, white pixels keep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
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
    parser.add_argument("--videox_fun_path", default=None, help="VideoX-Fun checkout; auto-detected from model_path")
    parser.add_argument("--config", default=None, help="Wan 5B YAML; defaults to config/wan2.2/wan_civitai_5b.yaml")
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
    parser.add_argument("--lanpaint_cfg", type=float, default=10000.0)
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
    parser.add_argument("--no_exact_composite", action="store_true", help="Do not restore keep pixels after decoding")
    return parser.parse_args()


def find_videox_fun(model_path: Path, explicit: str | None) -> Path:
    candidates = [] if explicit is None else [Path(explicit)]
    candidates.extend([model_path, *model_path.parents])
    for candidate in candidates:
        if (candidate / "videox_fun").is_dir() and (candidate / "config" / "wan2.2").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find the VideoX-Fun source tree. Pass --videox_fun_path /mnt/DataPart/jianghongda/VideoX-Fun"
    )


def read_prompt(value: str) -> str:
    path = Path(value)
    return path.read_text(encoding="utf-8").strip() if path.is_file() else value.strip()


def save_video(tensor: torch.Tensor, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor[0].permute(1, 2, 3, 0).detach().float().cpu().clamp(0, 1).numpy()
    imageio.mimsave(path, (frames * 255.0).round().astype(np.uint8), fps=fps, macro_block_size=1)


def load_pipeline(model_path: Path, config_path: Path, device: torch.device, dtype: torch.dtype):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer
    from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel
    from videox_fun.utils.utils import filter_kwargs
    from LanPaint.wan22_pipeline import Wan22LanPaintPipeline

    config = OmegaConf.load(config_path)
    additional = OmegaConf.to_container(config["transformer_additional_kwargs"])
    transformer_subpath = config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        model_path / transformer_subpath,
        transformer_additional_kwargs=additional,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    vae_cls = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
        config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")
    ]
    vae = vae_cls.from_pretrained(
        model_path / config["vae_kwargs"].get("vae_subpath", "vae"),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        model_path / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    pipeline = Wan22LanPaintPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    pipeline.to(device=device)
    return pipeline


def main() -> None:
    args = parse_args()
    started = time.time()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    videox_fun = find_videox_fun(model_path, args.videox_fun_path)
    sys.path.insert(0, str(videox_fun))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from LanPaint.wan22_runtime import read_video_pair

    config_path = Path(args.config).resolve() if args.config else videox_fun / "config/wan2.2/wan_civitai_5b.yaml"
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
    pipeline = load_pipeline(model_path, config_path, device, dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        video=video,
        mask_video=regenerate_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
        output_type="pil",
        lanpaint_steps=args.lanpaint_steps,
        lanpaint_cfg=args.lanpaint_cfg,
        lanpaint_lambda=args.lanpaint_lambda,
        lanpaint_step_size=args.lanpaint_step_size,
        lanpaint_beta=args.lanpaint_beta,
        lanpaint_friction=args.lanpaint_friction,
        lanpaint_early_stop=args.lanpaint_early_stop,
    ).videos

    regen = (regenerate_mask >= 127.5).to(result.dtype)
    if not args.no_exact_composite:
        result = result * regen + video.to(result.dtype) * (1.0 - regen)
    output = Path(args.output).resolve()
    fps = args.fps or input_info.fps
    save_video(result, output, fps)

    mask_rgb = regen.repeat(1, 3, 1, 1, 1)
    comparison = torch.cat([video, mask_rgb, result], dim=-1)
    comparison_path = output.with_name(f"{output.stem}_comparison{output.suffix}")
    save_video(comparison, comparison_path, fps)
    metadata = vars(args) | {
        "prompt_text": prompt,
        "resolved_model_path": str(model_path),
        "resolved_videox_fun_path": str(videox_fun),
        "resolved_config": str(config_path),
        "input_fps": input_info.fps,
        "input_frames": input_info.frames,
        "mask_semantics": "black=regenerate, white=keep" if not args.white_is_regenerate else "white=regenerate, black=keep",
        "elapsed_seconds": time.time() - started,
        "comparison_output": str(comparison_path),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
