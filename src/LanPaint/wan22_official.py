"""LanPaint video inpainting on the official Wan2.2 TI2V-5B runtime."""

from __future__ import annotations

import math
import random
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from tqdm import tqdm

from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .wan22_runtime import lanpaint_flow_step


class Wan22OfficialLanPaint(WanTI2V):
    """Official 48-channel Wan2.2-TI2V-5B with LanPaint inner sampling."""

    @torch.no_grad()
    def inpaint(
        self,
        *,
        prompt: str,
        video: torch.Tensor,
        regenerate_mask: torch.Tensor,
        sampling_steps: int,
        guide_scale: float,
        lanpaint_steps: int,
        lanpaint_cfg: float,
        lanpaint_lambda: float,
        lanpaint_step_size: float,
        lanpaint_beta: float,
        lanpaint_friction: float,
        lanpaint_early_stop: int,
        shift: float,
        negative_prompt: str,
        seed: int,
    ) -> torch.Tensor:
        if video.shape[0] != 1:
            raise ValueError("The validation baseline currently supports batch size 1")
        _, _, frame_num, height, width = video.shape
        if (frame_num - 1) % self.vae_stride[0] != 0:
            raise ValueError(f"num_frames must be 4n+1 for the official 5B VAE, got {frame_num}")

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        source_video = video[0].to(self.device, dtype=torch.float32).mul(2.0).sub(1.0)
        source_latent = self.vae.encode([source_video])[0].unsqueeze(0)
        source_latent = source_latent.to(device=self.device, dtype=self.param_dtype)
        noise = torch.randn(source_latent.shape, generator=generator, device=self.device, dtype=torch.float32)
        noise = noise.to(self.param_dtype)

        regen = (regenerate_mask >= 127.5).to(device=self.device, dtype=self.param_dtype)
        regen = F.interpolate(regen, size=source_latent.shape[-3:], mode="nearest")
        # Match LanPaint's conservative video-mask treatment: a painted pixel
        # affects neighboring temporal latent slices after 4x VAE compression.
        regen = F.max_pool3d(regen, kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0))
        latent = source_latent * (1.0 - regen) + noise * regen

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([negative_prompt or self.sample_neg_prompt], self.device)
        else:
            context = [x.to(self.device) for x in self.text_encoder([prompt], torch.device("cpu"))]
            context_null = [x.to(self.device) for x in self.text_encoder(
                [negative_prompt or self.sample_neg_prompt], torch.device("cpu")
            )]

        seq_len = math.ceil(
            source_latent.shape[2] * source_latent.shape[3] * source_latent.shape[4]
            / (self.patch_size[1] * self.patch_size[2])
            / self.sp_size
        ) * self.sp_size
        arg_cond = {"context": [context[0]], "seq_len": seq_len}
        arg_uncond = {"context": context_null, "seq_len": seq_len}

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = scheduler.timesteps

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        self.model.to(self.device)
        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for index, timestep in enumerate(tqdm(timesteps, desc="Wan2.2-5B LanPaint")):
                token_timestep = (regen[0, 0, :, ::2, ::2] * timestep).flatten()
                token_timestep = torch.cat([
                    token_timestep,
                    token_timestep.new_ones(seq_len - token_timestep.numel()) * timestep,
                ]).unsqueeze(0)

                def predict_velocity(sample: torch.Tensor, cfg: float) -> torch.Tensor:
                    cond = self.model([sample[0]], t=token_timestep, **arg_cond)[0]
                    uncond = self.model([sample[0]], t=token_timestep, **arg_uncond)[0]
                    return (uncond + cfg * (cond - uncond)).unsqueeze(0)

                use_lanpaint = lanpaint_steps > 0 and index < len(timesteps) - lanpaint_early_stop
                flow_sigma = (timestep / self.num_train_timesteps).to(latent.dtype).clamp(1e-5, 1.0 - 1e-5)
                if use_lanpaint:
                    def predict_x0(sample: torch.Tensor, _sigma: float):
                        normal = predict_velocity(sample, guide_scale)
                        big = predict_velocity(sample, lanpaint_cfg)
                        return sample - flow_sigma * normal, sample - flow_sigma * big

                    latent, velocity = lanpaint_flow_step(
                        latents=latent,
                        source_latents=source_latent,
                        original_noise=noise,
                        regenerate_mask=regen,
                        flow_sigma=flow_sigma,
                        predict_x0=predict_x0,
                        seed=seed,
                        steps=lanpaint_steps,
                        friction=lanpaint_friction,
                        content_lambda=lanpaint_lambda,
                        beta=lanpaint_beta,
                        step_size=lanpaint_step_size,
                    )
                else:
                    velocity = predict_velocity(latent, guide_scale)

                latent = scheduler.step(
                    velocity,
                    timestep,
                    latent,
                    return_dict=False,
                    generator=generator,
                )[0]
                # Official TI2V-style latent replacement. Masked locations
                # evolve; known locations remain tied to the encoded video.
                latent = source_latent * (1.0 - regen) + latent * regen

        return self.vae.decode([latent[0]])[0]
