"""Local SD-compatible API server backed by a real Stable Diffusion model via diffusers.

It exposes AUTOMATIC1111-compatible endpoints used by app/clients.py:
- POST /sdapi/v1/txt2img
- GET  /sdapi/v1/progress
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Default model repo
MODEL_ID = os.getenv("SD_MODEL_ID", "stabilityai/sdxl-turbo")
SDXL_BASE_MODEL = os.getenv("SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
SDXL_LIGHTNING_STEPS = int(os.getenv("SDXL_LIGHTNING_STEPS", "4"))
SDXL_LIGHTNING_CKPT = os.getenv("SDXL_LIGHTNING_CKPT", "")
DEVICE = os.getenv("SD_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32
IS_SDXL_LIGHTNING = MODEL_ID.lower() == "bytedance/sdxl-lightning"
IS_SDXL_TURBO = MODEL_ID.lower() == "stabilityai/sdxl-turbo"


class Txt2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    steps: int = 25
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    batch_size: int = 1
    n_iter: int = 1
    sampler_name: str | None = None


app = FastAPI(title="Local Stable Diffusion API")


def _to_b64_png(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Load a true diffusion model once on startup.
def _build_pipeline():
    if IS_SDXL_TURBO:
        if DTYPE == torch.float16:
            pipe_local = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                variant="fp16",
            )
        else:
            pipe_local = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
            )
        pipe_local.scheduler = EulerDiscreteScheduler.from_config(
            pipe_local.scheduler.config,
            timestep_spacing="trailing",
        )
    elif IS_SDXL_LIGHTNING:
        if SDXL_LIGHTNING_CKPT:
            ckpt_name = SDXL_LIGHTNING_CKPT
        elif SDXL_LIGHTNING_STEPS == 1:
            ckpt_name = "sdxl_lightning_1step_unet_x0.safetensors"
        elif SDXL_LIGHTNING_STEPS in (2, 4, 8):
            ckpt_name = f"sdxl_lightning_{SDXL_LIGHTNING_STEPS}step_unet.safetensors"
        else:
            raise ValueError("SDXL_LIGHTNING_STEPS must be one of: 1, 2, 4, 8")

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.load_config(SDXL_BASE_MODEL, subfolder="unet")
        )
        ckpt_path = hf_hub_download(repo_id=MODEL_ID, filename=ckpt_name)
        unet.load_state_dict(load_file(ckpt_path, device=DEVICE))

        pipe_local = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE_MODEL,
            unet=unet,
            torch_dtype=DTYPE,
            variant="fp16" if DTYPE == torch.float16 else None,
        )

        if SDXL_LIGHTNING_STEPS == 1:
            pipe_local.scheduler = EulerDiscreteScheduler.from_config(
                pipe_local.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="sample",
            )
        else:
            pipe_local.scheduler = EulerDiscreteScheduler.from_config(
                pipe_local.scheduler.config,
                timestep_spacing="trailing",
            )
    else:
        pipe_local = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            safety_checker=None,
        )

    pipe_local = pipe_local.to(DEVICE)
    # These options reduce peak memory usage on constrained devices (especially Apple Silicon).
    pipe_local.enable_attention_slicing()
    if hasattr(pipe_local, "enable_vae_slicing"):
        pipe_local.enable_vae_slicing()
    pipe_local.set_progress_bar_config(disable=True)
    return pipe_local


pipe = _build_pipeline()


@app.post("/sdapi/v1/txt2img")
def txt2img(body: Txt2ImgRequest) -> dict[str, Any]:
    count = max(1, min(4, body.batch_size))
    width = max(256, min(1024, body.width))
    height = max(256, min(1024, body.height))

    # SD models require dimensions divisible by 8.
    width -= width % 8
    height -= height % 8

    if IS_SDXL_TURBO:
        out = pipe(
            prompt=[body.prompt] * count,
            negative_prompt=[body.negative_prompt or ""] * count,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=height,
            width=width,
        )
    elif IS_SDXL_LIGHTNING:
        out = pipe(
            prompt=[body.prompt] * count,
            negative_prompt=[body.negative_prompt or ""] * count,
            num_inference_steps=SDXL_LIGHTNING_STEPS,
            guidance_scale=0.0,
            height=height,
            width=width,
        )
    else:
        out = pipe(
            prompt=[body.prompt] * count,
            negative_prompt=[body.negative_prompt or ""] * count,
            num_inference_steps=max(10, min(50, body.steps)),
            guidance_scale=max(1.0, min(12.0, body.cfg_scale)),
            height=height,
            width=width,
        )

    images = [_to_b64_png(img) for img in out.images]

    return {
        "images": images,
        "parameters": body.model_dump(),
        "info": f"real-sd-model:{MODEL_ID}",
    }


@app.get("/sdapi/v1/progress")
def progress() -> dict[str, Any]:
    return {"progress": 1.0, "state": {"job": "idle"}}
