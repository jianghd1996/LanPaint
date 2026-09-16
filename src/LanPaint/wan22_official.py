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
from wan.utils.fm_solvers import get_sampling_sigmas

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
    ) -> tuple[torch.Tensor, dict]:
        if video.shape[0] != 1:
            raise ValueError("The validation baseline currently supports batch size 1")
        _, _, frame_num, height, width = video.shape
        if (frame_num - 1) % self.vae_stride[0] != 0:
            raise ValueError(f"num_frames must be 4n+1 for the official 5B VAE, got {frame_num}")

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
        # ComfyUI's LanPaint KSampler starts from the scheduler-scaled noise.
        # Known content is injected by LanPaint at the current noise level;
        # it is not mixed into the initial latent as clean x0.
        latent = noise.clone()

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

        sigmas = torch.as_tensor(
            get_sampling_sigmas(sampling_steps, shift),
            device=self.device,
            dtype=torch.float32,
        )
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        diagnostics = {
            "sampler": "flow_euler",
            "mask_latent_coverage": float(regen.float().mean().item()),
            "source_latent": self._tensor_stats(source_latent),
            "initial_noise": self._tensor_stats(noise),
            "steps": [],
        }

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        self.model.to(self.device)
        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for index in tqdm(range(sampling_steps), desc="Wan2.2-5B LanPaint"):
                flow_sigma = sigmas[index].to(latent.dtype)
                next_sigma = sigmas[index + 1].to(latent.dtype)
                timestep = flow_sigma.float() * self.num_train_timesteps
                # LanPaint's WAN22 override deliberately disables WAN22's
                # per-token TI2V timestep conditioning. Every token sees the
                # same outer diffusion time; the inpaint mask lives solely in
                # the LanPaint conditional sampler.
                token_timestep = timestep.expand(1, seq_len)

                def predict_velocity(sample: torch.Tensor, cfg: float) -> torch.Tensor:
                    cond = self.model([sample[0]], t=token_timestep, **arg_cond)[0]
                    uncond = self.model([sample[0]], t=token_timestep, **arg_uncond)[0]
                    return (uncond + cfg * (cond - uncond)).unsqueeze(0)

                sigma_safe = flow_sigma.clamp(1e-5, 1.0 - 1e-5)
                abt = (1.0 - sigma_safe).square() / ((1.0 - sigma_safe).square() + sigma_safe.square())
                remaining_noise = float((1.0 - abt).item())
                effective_steps = max(0, round(lanpaint_steps * remaining_noise))
                if index >= sampling_steps - lanpaint_early_stop:
                    effective_steps = 0
                use_lanpaint = effective_steps > 0
                if use_lanpaint:
                    def predict_x0(sample: torch.Tensor, _sigma: float):
                        normal = predict_velocity(sample, guide_scale)
                        big = predict_velocity(sample, lanpaint_cfg)
                        return sample - sigma_safe * normal, sample - sigma_safe * big

                    latent, velocity = lanpaint_flow_step(
                        latents=latent,
                        source_latents=source_latent,
                        original_noise=noise,
                        regenerate_mask=regen,
                        flow_sigma=sigma_safe,
                        predict_x0=predict_x0,
                        seed=seed,
                        steps=effective_steps,
                        friction=lanpaint_friction,
                        content_lambda=lanpaint_lambda,
                        beta=lanpaint_beta,
                        step_size=lanpaint_step_size,
                    )
                else:
                    # Baseline equivalent of ComfyUI's masked KSampler input:
                    # inject the known latent at the *current* flow noise
                    # level, never as a clean latent at non-zero sigma.
                    known_noisy = sigma_safe * noise + (1.0 - sigma_safe) * source_latent
                    latent = latent * regen + known_noisy * (1.0 - regen)
                    velocity = predict_velocity(latent, guide_scale)

                # FlowMatch Euler: dx/dsigma is the model's flow prediction.
                # This is the outer sampler used by the official LanPaint Wan
                # workflow and has no incompatible multistep history.
                latent = latent + (next_sigma - flow_sigma) * velocity
                step_stats = self._tensor_stats(latent)
                step_stats.update({
                    "index": index,
                    "sigma": float(flow_sigma.float().item()),
                    "effective_lanpaint_steps": effective_steps,
                })
                diagnostics["steps"].append(step_stats)

        diagnostics["final_latent"] = self._tensor_stats(latent)
        return self.vae.decode([latent[0]])[0], diagnostics

    @staticmethod
    def _tensor_stats(value: torch.Tensor) -> dict:
        finite = torch.isfinite(value)
        safe = value.float()[finite]
        if safe.numel() == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nan": int(torch.isnan(value).sum().item()),
                "inf": int(torch.isinf(value).sum().item()),
            }
        return {
            "min": float(safe.min().item()),
            "max": float(safe.max().item()),
            "mean": float(safe.mean().item()),
            "std": float(safe.std().item()),
            "nan": int(torch.isnan(value).sum().item()),
            "inf": int(torch.isinf(value).sum().item()),
        }
