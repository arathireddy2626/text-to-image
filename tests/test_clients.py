"""
Unit tests for the upstream HTTP clients (llama.cpp + Stable Diffusion).
All network calls are intercepted with respx so no live servers are needed.
"""
import base64

import pytest
import respx
from httpx import Response

from app.clients import ServiceUnavailableError, enhance_prompt, generate_images


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ── enhance_prompt ────────────────────────────────────────────────────────────

@pytest.mark.anyio
@respx.mock
async def test_enhance_prompt_success():
    respx.post("http://localhost:8081/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "  enhanced text  "}}
                ]
            },
        )
    )
    result = await enhance_prompt("a cat")
    assert result == "enhanced text"  # strip() applied


@pytest.mark.anyio
@respx.mock
async def test_enhance_prompt_server_error():
    respx.post("http://localhost:8081/v1/chat/completions").mock(
        return_value=Response(500, text="Internal error")
    )
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await enhance_prompt("a cat")
    assert "llama.cpp" in str(exc_info.value)


@pytest.mark.anyio
@respx.mock
async def test_enhance_prompt_bad_shape():
    respx.post("http://localhost:8081/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": []})
    )
    with pytest.raises(ServiceUnavailableError):
        await enhance_prompt("a cat")


# ── generate_images ───────────────────────────────────────────────────────────

FAKE_B64 = base64.b64encode(b"fake-png").decode()


@pytest.mark.anyio
@respx.mock
async def test_generate_images_success():
    respx.post("http://localhost:7860/sdapi/v1/txt2img").mock(
        return_value=Response(200, json={"images": [FAKE_B64]})
    )
    result = await generate_images("a cat", n=1, size="512x512")
    assert result == [FAKE_B64]


@pytest.mark.anyio
@respx.mock
async def test_generate_images_multiple():
    respx.post("http://localhost:7860/sdapi/v1/txt2img").mock(
        return_value=Response(200, json={"images": [FAKE_B64, FAKE_B64, FAKE_B64]})
    )
    result = await generate_images("a cat", n=3, size="512x512")
    assert len(result) == 3


@pytest.mark.anyio
@respx.mock
async def test_generate_images_server_error():
    respx.post("http://localhost:7860/sdapi/v1/txt2img").mock(
        return_value=Response(503, text="model loading")
    )
    with pytest.raises(ServiceUnavailableError) as exc_info:
        await generate_images("a cat", n=1, size="512x512")
    assert "Stable Diffusion" in str(exc_info.value)


@pytest.mark.anyio
@respx.mock
async def test_generate_images_bad_response_shape():
    respx.post("http://localhost:7860/sdapi/v1/txt2img").mock(
        return_value=Response(200, json={"unexpected_key": []})
    )
    with pytest.raises(ServiceUnavailableError):
        await generate_images("a cat", n=1, size="512x512")
