"""Small runtime helpers for the standalone Wan2.2 video-inpainting example.

This module intentionally has no ComfyUI dependency. The actual Wan model is
loaded lazily from the official Wan-Video/Wan2.2 checkout, while these helpers
cover input validation and LanPaint's model adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .lanpaint import LanPaint


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    if info.fps <= 0 or info.frames <= 0 or info.width <= 0 or info.height <= 0:
        raise ValueError(f"Invalid video metadata for {path}: {info}")
    return info


def read_video_pair(
    video_path: str | Path,
    mask_path: str | Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    black_is_regenerate: bool = True,
    threshold: int = 127,
) -> tuple[torch.Tensor, torch.Tensor, VideoInfo]:
    """Read and align an RGB video and a binary mask video.

    Returns ``video`` in ``[1, 3, T, H, W]`` / [0, 1] and ``regen_mask`` in
    ``[1, 1, T, H, W]`` / {0, 255}: 255 regenerates and 0 keeps.
    """

    video_info = probe_video(video_path)
    mask_info = probe_video(mask_path)
    if abs(video_info.fps - mask_info.fps) > 1e-3:
        raise ValueError(f"FPS mismatch: video={video_info.fps:g}, mask={mask_info.fps:g}")
    if video_info.frames != mask_info.frames:
        raise ValueError(f"Frame-count mismatch: video={video_info.frames}, mask={mask_info.frames}")
    if (video_info.width, video_info.height) != (mask_info.width, mask_info.height):
        raise ValueError(
            "Resolution mismatch: "
            f"video={video_info.width}x{video_info.height}, mask={mask_info.width}x{mask_info.height}"
        )
    if num_frames > video_info.frames:
        raise ValueError(f"Requested {num_frames} frames, but input only has {video_info.frames}")

    video_cap = cv2.VideoCapture(str(video_path))
    mask_cap = cv2.VideoCapture(str(mask_path))
    rgb_frames: list[np.ndarray] = []
    binary_masks: list[np.ndarray] = []
    for index in range(num_frames):
        ok_video, bgr = video_cap.read()
        ok_mask, mask_bgr = mask_cap.read()
        if not ok_video or not ok_mask:
            raise ValueError(f"Failed to decode aligned frame {index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        regenerate = gray < threshold if black_is_regenerate else gray >= threshold
        rgb_frames.append(rgb)
        binary_masks.append(regenerate.astype(np.uint8) * 255)
    video_cap.release()
    mask_cap.release()

    video = torch.from_numpy(np.stack(rgb_frames)).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    regen_mask = torch.from_numpy(np.stack(binary_masks)).unsqueeze(0).unsqueeze(0).float()
    return video, regen_mask, video_info


def latent_regenerate_mask(pixel_mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Convert a pixel regenerate mask to a strict binary latent mask."""

    mask = (pixel_mask >= 127.5).to(device=latent.device, dtype=latent.dtype)
    return F.interpolate(mask, size=latent.shape[-3:], mode="nearest")


class _FlowSampling:
    @staticmethod
    def noise_scaling(sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return sigma * noise + (1.0 - sigma) * latent


class FlowX0Adapter:
    """Expose Wan flow predictions through the interface used by ``LanPaint``."""

    def __init__(self, predict_x0: Callable[[torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]]):
        self.predict_x0 = predict_x0
        self.inner_model = SimpleNamespace(model_sampling=_FlowSampling())

    def __call__(self, x, sigma, model_options=None, seed=None):
        del model_options, seed
        value = float(torch.as_tensor(sigma).mean().item())
        return self.predict_x0(x, value)


def lanpaint_flow_step(
    *,
    latents: torch.Tensor,
    source_latents: torch.Tensor,
    original_noise: torch.Tensor,
    regenerate_mask: torch.Tensor,
    flow_sigma: torch.Tensor,
    predict_x0: Callable[[torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    steps: int,
    friction: float,
    content_lambda: float,
    beta: float,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run LanPaint thinking and return updated latents plus flow velocity.

    LanPaint names its internal mask ``latent_mask``; it is the *known/keep*
    region, hence the inversion of the CLI/API regenerate mask below.
    """

    sigma = torch.as_tensor(flow_sigma, device=latents.device, dtype=latents.dtype).reshape(1)
    sigma = sigma.clamp(1e-5, 1.0 - 1e-5)
    abt = (1.0 - sigma).square() / ((1.0 - sigma).square() + sigma.square())
    ve_sigma = sigma / (1.0 - sigma)
    known_mask = 1.0 - regenerate_mask
    known_mask = known_mask.expand_as(latents)

    adapter = FlowX0Adapter(predict_x0)
    painter = LanPaint(
        Model=adapter,
        NSteps=steps,
        Friction=friction,
        Lambda=content_lambda,
        Beta=beta,
        StepSize=step_size,
        IS_FLOW=True,
        MinStepFrac=1.0,
    )
    working = latents.clone()
    denoised = painter(
        working,
        source_latents,
        original_noise,
        sigma,
        known_mask,
        (ve_sigma, abt, sigma),
        {},
        seed,
        n_steps=steps,
    )
    velocity = (working - denoised) / sigma.view(1, 1, 1, 1, 1)
    return working, velocity
