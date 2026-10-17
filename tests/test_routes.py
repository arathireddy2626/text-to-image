"""
Tests for the Text-to-Image service.

Uses pytest + httpx AsyncClient with mocked upstream clients so no running
llama.cpp or Stable Diffusion server is required.
"""
import base64
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


# ── helpers ──────────────────────────────────────────────────────────────────

FAKE_ENHANCED = "a cyberpunk cat sitting in heavy neon rain, cinematic, 8k, bokeh"
FAKE_B64 = base64.b64encode(b"fake-png-bytes").decode()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_interaction_page(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Text-to-Image Interaction Page" in r.text


# ── /v1/prompts/enhance ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_enhance_success(client):
    with patch("app.routes.enhance_prompt", new=AsyncMock(return_value=FAKE_ENHANCED)):
        r = await client.post(
            "/v1/prompts/enhance",
            json={"prompt": "a cyberpunk cat in the rain"},
        )
    assert r.status_code == 200
    assert r.json() == {"enhanced_prompt": FAKE_ENHANCED}


@pytest.mark.anyio
async def test_enhance_missing_prompt(client):
    r = await client.post("/v1/prompts/enhance", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "Invalid request"


@pytest.mark.anyio
async def test_enhance_empty_prompt(client):
    r = await client.post("/v1/prompts/enhance", json={"prompt": ""})
    assert r.status_code == 400


@pytest.mark.anyio
async def test_enhance_llama_unavailable(client):
    from app.clients import ServiceUnavailableError

    with patch(
        "app.routes.enhance_prompt",
        new=AsyncMock(side_effect=ServiceUnavailableError("llama.cpp", "connection refused")),
    ):
        r = await client.post(
            "/v1/prompts/enhance",
            json={"prompt": "a cat"},
        )
    assert r.status_code == 502
    assert "unavailable" in r.json()["error"].lower()


@pytest.mark.anyio
async def test_enhance_unexpected_error(client):
    with patch(
        "app.routes.enhance_prompt",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        r = await client.post(
            "/v1/prompts/enhance",
            json={"prompt": "a cat"},
        )
    assert r.status_code == 500


# ── /v1/images/generations ────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_generate_no_enhancer(client):
    with patch(
        "app.routes.generate_images",
        new=AsyncMock(return_value=[FAKE_B64]),
    ):
        r = await client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cyberpunk cat in the rain",
                "use_enhancer": False,
                "n": 1,
                "size": "512x512",
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["prompt_used"] == "a cyberpunk cat in the rain"
    assert len(body["data"]) == 1
    assert body["data"][0]["b64_json"] == FAKE_B64


@pytest.mark.anyio
async def test_generate_with_enhancer(client):
    with (
        patch("app.routes.enhance_prompt", new=AsyncMock(return_value=FAKE_ENHANCED)),
        patch("app.routes.generate_images", new=AsyncMock(return_value=[FAKE_B64])),
    ):
        r = await client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cyberpunk cat in the rain",
                "use_enhancer": True,
                "n": 1,
                "size": "512x512",
            },
        )
    assert r.status_code == 200
    body = r.json()
    # prompt_used must be the enhanced version
    assert body["prompt_used"] == FAKE_ENHANCED
    assert len(body["data"]) == 1


@pytest.mark.anyio
async def test_generate_invalid_size(client):
    r = await client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "size": "badsize"},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_generate_invalid_n(client):
    r = await client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "n": 0},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_generate_sd_unavailable(client):
    from app.clients import ServiceUnavailableError

    with patch(
        "app.routes.generate_images",
        new=AsyncMock(side_effect=ServiceUnavailableError("Stable Diffusion", "timeout")),
    ):
        r = await client.post(
            "/v1/images/generations",
            json={"prompt": "a cat", "use_enhancer": False},
        )
    assert r.status_code == 502


@pytest.mark.anyio
async def test_generate_multiple_images(client):
    fake_images = [FAKE_B64, FAKE_B64, FAKE_B64]
    with patch(
        "app.routes.generate_images",
        new=AsyncMock(return_value=fake_images),
    ):
        r = await client.post(
            "/v1/images/generations",
            json={"prompt": "a cat", "n": 3, "size": "512x512"},
        )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


@pytest.mark.anyio
async def test_generate_missing_prompt(client):
    r = await client.post("/v1/images/generations", json={})
    assert r.status_code == 400


# ── input validation edge cases ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_invalid_json_body(client):
    r = await client.post(
        "/v1/prompts/enhance",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
