# Prompt-Assisted Text-to-Image Generation Service

An asynchronous, fully local REST API that bridges Large Language Models with Diffusion models to generate high-quality images from natural language.

This service acts as an orchestration layer, exposing a robust FastAPI interface that delegates prompt engineering to a local `llama.cpp` instance (running as an OpenAI-compatible server) before passing the optimized prompts to a local Stable Diffusion instance for image synthesis. Designed with production-grade engineering principles, it features strict payload validation, non-blocking I/O, and deterministic error handling.

## Key Features & Technical Highlights

* **Intelligent Prompt Routing:** Dynamically intercepts user prompts and optionally routes them through a local LLM for semantic enhancement prior to image generation, drastically improving the quality of diffusion outputs.
* **Asynchronous Architecture:** Built on **FastAPI** and **httpx**, ensuring non-blocking concurrent requests between the gateway, the LLM backend, and the image generation service.
* **Strict Schema Validation:** Utilizes **Pydantic** (`v2.10.4`) to enforce rigid request/response contracts, ensuring predictable downstream model execution.
* **Graceful Error Handling:** Implements standardized HTTP status codes mapping to specific failure domains (400 for contract violations, 502 for upstream AI model timeouts/unavailability, and 500 for unhandled state errors).
* **Fully Local Execution:** Architected to run entirely on-premise without external API dependencies, ensuring complete data privacy and zero recurring inference costs.

## System Architecture

1. **Client:** Submits a JSON payload via standard HTTP POST.
2. **FastAPI Application:** Validates the payload using Pydantic models.
3. **Enhancer Pipeline (Optional):** If `use_enhancer=true`, an async `httpx` call is dispatched to the local `llama.cpp` server to expand the prompt with descriptive visual keywords.
4. **Generation Pipeline:** The final prompt (original or enhanced) is dispatched to the local Stable Diffusion engine.
5. **Response:** Returns a standard JSON payload containing the resulting image data/metadata.

---

## Prerequisites & Installation

### 1. Dependencies

The application requires Python 3.10+ and relies on the following core dependencies:

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
httpx==0.27.2
pydantic==2.10.4

```

### 2. Setup Instructions

Clone the repository and prepare your virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt

```

### 3. Running Upstream Services

Before starting the API, ensure your local models are running on their designated ports:

* **llama.cpp Server:** Start your local OpenAI-compatible LLM server.
* **Stable Diffusion Engine:** Start your local image generation server.

### 4. Start the API Service

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8080 --reload

```

---

## API Reference & Usage

### 1. Enhance Prompt (`POST /v1/prompts/enhance`)

Accepts a raw prompt and returns a highly detailed, descriptive version optimized for diffusion models.

**Request:**

```bash
curl -X POST http://localhost:8080/v1/prompts/enhance \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a futuristic city at night"}'

```

**Successful Response (200 OK):**

```json
{
  "enhanced_prompt": "a sprawling futuristic cyberpunk metropolis at midnight, illuminated by vibrant neon signs in cyan and magenta, towering skyscrapers piercing through thick atmospheric fog, flying vehicles leaving light trails, highly detailed, 8k resolution, cinematic lighting, octane render"
}

```

### 2. Generate Image with Prompt Assist (`POST /v1/images/generations`)

Accepts a prompt and generation parameters. If `use_enhancer` is flagged, it automatically handles the LLM enhancement step before synthesizing the image.

**Request:**

```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a golden retriever wearing a spacesuit on Mars",
    "use_enhancer": true,
    "n": 1,
    "size": "512x512"
  }'

```

**Successful Response (200 OK):**

```json
{
  "created": 1700000000,
  "original_prompt": "a golden retriever wearing a spacesuit on Mars",
  "enhanced_prompt": "a majestic golden retriever dog wearing a highly detailed, futuristic white astronaut spacesuit with a reflective gold visor, standing on the dusty red terrain of Mars, distant rugged mountains, starry outer space background, photorealistic, 4k, cinematic composition",
  "data": [
    {
      "url": "http://localhost:8080/static/generated_images/retriever_mars_01.png"
    }
  ]
}

```

### 3. Standard Image Generation (`POST /v1/images/generations`)

Bypass the LLM enhancer to send raw, exact prompts directly to the Stable Diffusion backend for strict control.

**Request:**

```bash
curl -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "minimalist logo design of a coffee cup, flat vector, black and white",
    "use_enhancer": false,
    "n": 1,
    "size": "512x512"
  }'

```

---

## Error Handling Matrix

The API uses standardized HTTP codes to clearly communicate the source of a failure, making it highly reliable for downstream integrations.

| HTTP Code | Domain | Description |
| --- | --- | --- |
| **400** | Client Error | **Invalid JSON / Schema Violation:** The request body is malformed, missing required fields (e.g., `prompt`), or contains invalid parameter types caught by Pydantic. |
| **502** | Upstream Error | **Bad Gateway:** The local `llama.cpp` server or Stable Diffusion instance is offline, unreachable, or timed out during inference. |
| **500** | Server Error | **Internal Server Error:** An unexpected application state or unhandled exception occurred within the FastAPI orchestration layer. |
