"""
generation.py
──────────────
POST /generate-room
  Accepts a room photo + room_type/style/color (multipart), returns a
  redesigned room image. A Critic Agent checks the result against the
  requested room/style/color and triggers one automatic retry if it
  doesn't match (skipped for the mock provider). Optionally runs
  furniture segmentation on the result when segment=true.

Image providers (chosen via AI_IMAGE_PROVIDER env var):
  • gemini      — Gemini image generation (default)
  • replicate   — Replicate (e.g. google/nano-banana-2)
  • mock        — echoes the uploaded photo with a watermark; no API call,
                  used when Gemini/Replicate credentials aren't set up yet
"""

import asyncio
import base64
import os
from io import BytesIO
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw

from segmentation import run_segmentation

router = APIRouter(tags=["generation"])

# ── Style/color descriptions fed to the prompt-composition agent ──────────
# Keep these in sync with the style/color ids the app sends
# (lib/features/home/presentation/home_screen.dart). Room type, style, and
# color together form a fixed, enumerable set (5 x 3 x 3 = 45 combinations),
# so composed prompts are cached below instead of re-generated every request.

_STYLE_DESCRIPTIONS: dict[str, str] = {
    "japandi": (
        "Japandi (a warm, minimalist blend of Japanese and Scandinavian "
        "design): light oak and pale wood furniture, clean simple lines, "
        "natural textures, uncluttered layout"
    ),
    "industrial_loft": (
        "Industrial / Loft: exposed concrete and brick, black steel frames, "
        "dark stained wood, utilitarian raw-material finishes"
    ),
    "modern_luxury": (
        "Modern Luxury: polished marble and glass surfaces, metallic gold "
        "or chrome accents, glossy finishes, upscale contemporary furniture"
    ),
}

_COLOR_DESCRIPTIONS: dict[str, str] = {
    "warm_oat_cream": "warm oat and cream tones on walls and soft furnishings",
    "muted_sage_green": "a muted, earthy sage green on accent walls and cushions",
    "soft_terracotta": "a soft, faded terracotta on accent walls and textiles",
    "raw_concrete_matte_black": "raw concrete grey paired with matte black fixtures",
    "rusty_brick_leather": "rusty brick red paired with dark leather upholstery",
    "dark_navy_blue": "a dark navy blue on accent walls and upholstery",
    "ivory_champagne_gold": "ivory white paired with champagne gold accents",
    "emerald_green_brass": "deep emerald green paired with brass accents",
    "midnight_blue_silver": "midnight blue paired with silver/chrome accents",
}

_COMPOSER_SYSTEM_PROMPT = """
You are an interior-design prompt writer for an AI image generator.
Given a room type, a design style, and an accent color description, write
ONE vivid English paragraph (60-100 words) instructing the generator how to
redecorate the room.

Rules:
- Use natural descriptive language only. Never use attention-weighting
  syntax such as (word:1.2) - the target model does not support it.
- Mention concrete furniture, materials, and lighting appropriate to the
  style and color.
- Do not mention changing the room layout, walls, windows, doors, or
  camera angle - those must be preserved as-is.
- Output only the paragraph, no headings or extra commentary.
""".strip()

# room_type/style/color id combo -> composed prompt. Cleared only on process
# restart; the combos are a fixed, small set so this stays warm in practice.
_PROMPT_CACHE: dict[tuple[str, str, str], str] = {}

# Hard structural constraints that must reach the image model on every
# single request, regardless of whether the prompt-composition agent
# succeeded or the static fallback kicked in. The composer only writes the
# style/furniture description - it is explicitly told not to talk about
# layout changes, but that alone doesn't put "keep the doors/windows/outlets
# visible" text in front of the image model, which is what actually keeps it
# from painting over them. This clause is appended after whichever prompt
# path ran, so it's never at the mercy of what the composer happened to say.
_STRUCTURE_PRESERVATION_CLAUSE = """
Preserve the original room layout, walls, floor, windows, doors, and all visible fixtures exactly as shown in the photo. Do not change the room structure, camera angle, or perspective. Every door, window, electrical outlet, and light switch visible in the original photo must remain fully visible and unobstructed in the result - do not place furniture, decor, or any object in front of, over, or blocking them. Leave enough open walking space. Return a photorealistic decorated room image only.
""".strip()


def build_static_fallback_prompt(room_type: str, style: str, color: str) -> str:
    """Used if the prompt-composition agent call fails for any reason, so a
    hiccup in that extra LLM call never breaks the /generate-room endpoint.
    """
    style_desc = _STYLE_DESCRIPTIONS.get(style, style)
    color_desc = _COLOR_DESCRIPTIONS.get(color, color)
    return (
        f"Redesign this uploaded {room_type} as a realistic interior in the "
        f"following style: {style_desc}. Use {color_desc}.\n\n"
        f"{_STRUCTURE_PRESERVATION_CLAUSE}"
    )


async def compose_design_prompt(room_type: str, style: str, color: str) -> str:
    """Prompt-composition agent: turns the short room/style/color ids into a
    single descriptive, natural-language paragraph via a cheap text-only
    Gemini call. Results are cached per id combo (see _PROMPT_CACHE) since
    the option set is fixed and small - repeat requests for the same
    combination cost nothing after the first.
    """
    cache_key = (room_type, style, color)
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the backend.")

    model = os.getenv("GEMINI_PROMPT_MODEL", "gemini-3.5-flash-lite")
    style_desc = _STYLE_DESCRIPTIONS.get(style, style)
    color_desc = _COLOR_DESCRIPTIONS.get(color, color)

    payload = {
        "systemInstruction": {"parts": [{"text": _COMPOSER_SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"Room type: {room_type}\n"
                            f"Style: {style_desc}\n"
                            f"Color: {color_desc}"
                        ),
                    },
                ],
            },
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
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

    text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    _PROMPT_CACHE[cache_key] = text
    return text


def build_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, encoded = data_url.split(",", 1)
    mime_type = header.split(":", 1)[1].split(";", 1)[0] if ":" in header else "image/png"
    return base64.b64decode(encoded), mime_type


_CRITIC_SYSTEM_PROMPT = """
You are a strict quality checker for an AI interior-design image generator.
You will be shown two photos: the ORIGINAL room photo the user uploaded, and
the REDECORATED result the AI produced from it, along with the room type and
style/color it was supposed to become.

Check all of the following against the REDECORATED photo:
1. It plausibly shows the same room type as intended.
2. The decor plausibly matches the requested style/color intent.
3. Every door, window, and visible electrical outlet/light switch that
   appears in the ORIGINAL photo is still visible and NOT obstructed,
   covered, or removed in the REDECORATED photo - new furniture or decor
   must not block them.
4. The room's overall layout, walls, and camera angle still look like the
   same room, not a different one.

Ignore minor imperfections - only fail on a clear, obvious violation (e.g. a
window that disappeared, a door blocked by a sofa, an outlet painted over,
the wrong room type, or a style/color that is clearly not what was asked
for).

Reply with exactly one line: either "PASS" or "FAIL: <short reason>".
""".strip()


async def critique_generated_image(
    original_bytes: bytes,
    original_mime: str,
    generated_bytes: bytes,
    generated_mime: str,
    room_type: str,
    style: str,
    color: str,
) -> bool:
    """Critic Agent: asks Gemini Vision to compare the generated image against
    the original photo, checking both style/room-type intent and that doors,
    windows, and outlets/switches from the original weren't covered or
    removed. Returns True (pass) whenever the check can't be completed (no
    API key, network error, unexpected response) - a broken critic should
    never block a user's generation, only a *confirmed* mismatch should.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return True

    model = os.getenv("GEMINI_CRITIC_MODEL", "gemini-3.6-flash")
    style_desc = _STYLE_DESCRIPTIONS.get(style, style)
    color_desc = _COLOR_DESCRIPTIONS.get(color, color)

    payload = {
        "systemInstruction": {"parts": [{"text": _CRITIC_SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"Intended room type: {room_type}\n"
                            f"Intended style: {style_desc}\n"
                            f"Intended color: {color_desc}\n\n"
                            "Here is the ORIGINAL photo:"
                        ),
                    },
                    {
                        "inline_data": {
                            "mime_type": original_mime,
                            "data": base64.b64encode(original_bytes).decode("utf-8"),
                        },
                    },
                    {"text": "Here is the REDECORATED result:"},
                    {
                        "inline_data": {
                            "mime_type": generated_mime,
                            "data": base64.b64encode(generated_bytes).decode("utf-8"),
                        },
                    },
                ],
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
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

        verdict = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        print(f"critique_generated_image failed, treating as pass: {exc}")
        return True

    passed = verdict.upper().startswith("PASS")
    if not passed:
        print(f"critique_generated_image flagged a mismatch: {verdict}")
    return passed


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
                returned_mime_type = (
                    inline_data.get("mimeType")
                    or inline_data.get("mime_type")
                    or "image/png"
                )
                return f"data:{returned_mime_type};base64,{inline_data['data']}"

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


async def generate_with_mock(room_type: str, style: str, color: str, source_image: Image.Image) -> str:
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
        f"MOCK PREVIEW - {style}/{color} {room_type} (no AI call made)",
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


@router.post("/generate-room")
async def generate_room(
    room_type: str = Form(...),
    style: str = Form(...),
    color: str = Form(...),
    image: UploadFile = File(...),
    segment: bool = Form(False),
):
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    try:
        source_image = Image.open(BytesIO(image_bytes))
        source_image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        style_paragraph = await compose_design_prompt(room_type, style, color)
        # The composer only writes the style/furniture description - the hard
        # "don't cover the doors/windows/outlets" constraint always gets
        # appended here rather than trusted to the composer's output.
        prompt = f"{style_paragraph}\n\n{_STRUCTURE_PRESERVATION_CLAUSE}"
    except Exception as exc:
        print(f"compose_design_prompt failed, falling back to static prompt: {exc}")
        prompt = build_static_fallback_prompt(room_type, style, color)

    provider = os.getenv("AI_IMAGE_PROVIDER", "gemini").lower()

    async def run_provider() -> str:
        if provider == "mock":
            return await generate_with_mock(room_type, style, color, source_image)
        if provider == "replicate":
            return await generate_with_replicate(prompt, image_bytes, mime_type)
        if provider == "gemini":
            return await generate_with_gemini(prompt, image_bytes, mime_type)
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported AI_IMAGE_PROVIDER: {provider}",
        )

    generated_image = await run_provider()

    # Critic Agent: the mock provider just watermarks the original photo, so
    # there's nothing to judge. Real providers get one automatic retry if the
    # first attempt doesn't actually match what was asked for, comparing
    # against the original photo so it can specifically catch a covered or
    # missing door/window/outlet, not just a wrong room type/style.
    if provider != "mock":
        gen_bytes, gen_mime = decode_data_url(generated_image)
        passed = await critique_generated_image(
            image_bytes, mime_type, gen_bytes, gen_mime, room_type, style, color
        )
        if not passed:
            print("Critic Agent rejected the first attempt, regenerating once...")
            retry_reminder = (
                "\n\nIMPORTANT: the previous attempt failed review for covering or "
                "removing a door, window, outlet, or switch. Be strict about "
                "keeping all of them fully visible and unobstructed this time."
            )
            prompt = f"{prompt}{retry_reminder}"
            generated_image = await run_provider()

    response: dict[str, Any] = {
        "generated_image": generated_image,
        "products": mock_products(),
    }

    if segment:
        _, encoded = generated_image.split(",", 1)
        gen_bytes = base64.b64decode(encoded)
        seg_result = await run_segmentation(gen_bytes, "image/png")
        response["furniture_segments"] = seg_result.model_dump()

    return response
