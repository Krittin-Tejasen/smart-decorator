import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI()


def build_room_prompt(room_type: str, theme: str) -> str:
    return f"""
Redesign this uploaded {room_type} as a realistic {theme} interior design.
Preserve the original room layout, walls, floor, windows, doors, and visible fixtures.
Do not change the room structure or camera angle.
Add realistic furniture and decor that fit the room scale.
Keep enough walking space and avoid blocking doors, windows, outlets, and switches.
Return a photorealistic decorated room image only.
""".strip()


def build_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def first_output_url(value: Any) -> str | None:
    if isinstance(value, str) and (
        value.startswith("http://")
        or value.startswith("https://")
        or value.startswith("data:image")
    ):
        return value

    if isinstance(value, list):
        for item in value:
            output = first_output_url(item)
            if output:
                return output

    if isinstance(value, dict):
        for item in value.values():
            output = first_output_url(item)
            if output:
                return output

    return None


async def fetch_output_as_data_url(client: httpx.AsyncClient, output_url: str, token: str) -> str:
    if output_url.startswith("data:image"):
        return output_url

    response = await client.get(
        output_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    response.raise_for_status()

    mime_type = response.headers.get("content-type", "image/png").split(";")[0]
    return build_data_url(response.content, mime_type)


async def generate_with_gemini(prompt: str, image_bytes: bytes, mime_type: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the backend.",
        )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
    except httpx.HTTPStatusError as exc:
        print(exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini generation failed: {exc.response.text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini generation failed: {exc}") from exc

    candidates = response_data.get("candidates", [])

    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                return f"data:image/png;base64,{inline_data['data']}"

    print(response_data)
    raise HTTPException(status_code=502, detail="Gemini did not return an image.")


async def generate_with_replicate(prompt: str, image_bytes: bytes, mime_type: str) -> str:
    token = os.getenv("REPLICATE_API_TOKEN")
    model = os.getenv("REPLICATE_MODEL", "google/nano-banana-2")
    image_input_field = os.getenv("REPLICATE_IMAGE_INPUT_FIELD", "image_input")
    image_input_is_array = os.getenv("REPLICATE_IMAGE_INPUT_IS_ARRAY", "true").lower() == "true"
    aspect_ratio = os.getenv("REPLICATE_ASPECT_RATIO", "match_input_image")
    output_format = os.getenv("REPLICATE_OUTPUT_FORMAT", "png")

    if not token:
        raise HTTPException(
            status_code=500,
            detail="REPLICATE_API_TOKEN is not configured on the backend.",
        )

    image_data_url = build_data_url(image_bytes, mime_type)
    image_value: str | list[str] = [image_data_url] if image_input_is_array else image_data_url
    prediction_input: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
        image_input_field: image_value,
    }

    try:
        async with httpx.AsyncClient(timeout=240) as client:
            response = await client.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Prefer": "wait=60",
                },
                json={"input": prediction_input},
            )
            response.raise_for_status()
            prediction = response.json()

            for _ in range(36):
                status = prediction.get("status")

                if status == "succeeded":
                    output_url = first_output_url(prediction.get("output"))
                    if not output_url:
                        print(prediction)
                        raise HTTPException(
                            status_code=502,
                            detail="Replicate succeeded but did not return an image.",
                        )
                    return await fetch_output_as_data_url(client, output_url, token)

                if status in {"failed", "canceled"}:
                    print(prediction)
                    raise HTTPException(
                        status_code=502,
                        detail=f"Replicate generation {status}: {prediction.get('error')}",
                    )

                get_url = prediction.get("urls", {}).get("get")
                if not get_url:
                    print(prediction)
                    raise HTTPException(
                        status_code=502,
                        detail="Replicate prediction did not include a polling URL.",
                    )

                await asyncio.sleep(5)
                poll_response = await client.get(
                    get_url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                poll_response.raise_for_status()
                prediction = poll_response.json()

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        print(exc.response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Replicate generation failed: {exc.response.text}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Replicate generation failed: {exc}") from exc

    raise HTTPException(status_code=504, detail="Replicate generation timed out.")


async def generate_with_mock(room_type: str, theme: str, source_image: Image.Image) -> str:
    """Echo the uploaded photo with a watermark instead of calling a paid AI provider.

    Lets the rest of the pipeline (upload, processing, results, history) be built and
    tested without Gemini/Replicate credentials configured.
    """
    await asyncio.sleep(1)

    preview = source_image.convert("RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    banner_height = max(40, preview.height // 12)
    draw.rectangle([(0, 0), (preview.width, banner_height)], fill=(0, 0, 0, 160))
    draw.text(
        (16, banner_height // 4),
        f"MOCK PREVIEW - {theme} {room_type} (no AI call made)",
        fill=(255, 255, 255, 255),
    )

    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return build_data_url(buffer.getvalue(), "image/png")


def mock_products() -> list[dict[str, Any]]:
    return [
        {
            "id": "1",
            "name": "Modern Sofa",
            "imageUrl": "sofa",
            "price": 12990,
        },
        {
            "id": "2",
            "name": "Minimal Lamp",
            "imageUrl": "lamp",
            "price": 2490,
        },
    ]


@app.get("/")
def root():
    return {
        "message": "Smart Decorator API Running",
    }


@app.post("/generate-room")
async def generate_room(
    room_type: str = Form(...),
    theme: str = Form(...),
    image: UploadFile = File(...),
):
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    try:
        source_image = Image.open(BytesIO(image_bytes))
        source_image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    prompt = build_room_prompt(room_type, theme)
    provider = os.getenv("AI_IMAGE_PROVIDER", "gemini").lower()

    if provider == "mock":
        generated_image = await generate_with_mock(room_type, theme, source_image)
    elif provider == "replicate":
        generated_image = await generate_with_replicate(prompt, image_bytes, mime_type)
    elif provider == "gemini":
        generated_image = await generate_with_gemini(prompt, image_bytes, mime_type)
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported AI_IMAGE_PROVIDER: {provider}",
        )

    return {
        "generated_image": generated_image,
        "products": mock_products(),
    }
