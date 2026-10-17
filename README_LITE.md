# Text-to-Image Lite Package

This lite package is intended for assignment sharing and review.

## What Is Included

1. FastAPI service code for prompt enhancement and image generation.
2. API route, client, and schema modules under `app/`.
3. Local Stable Diffusion service script under `scripts/`.
4. Tests under `tests/`.
5. Assignment documentation under `docs/`.

## What Is Not Included

1. Local virtual environments (`.venv*`).
2. Python cache folders (`__pycache__`, `.pytest_cache`).
3. Large generated build artifacts.

## How To Run (Quick)

```bash
cd text-to-image
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start services:

```bash
# API gateway
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

```bash
# Stable Diffusion local backend
SD_MODEL_ID=stabilityai/sdxl-turbo uvicorn scripts.sd_local_server:app --host 0.0.0.0 --port 7860
```

```bash
# llama.cpp server (from llama.cpp folder)
./build/bin/llama-server --model ../models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8081 --ctx-size 2048
```

## Verification

```bash
pytest tests -v
```

## Notes

1. First startup can be slow because model assets may download.
2. Keep `size` at `512x512` and `n=1` for faster local runs.
