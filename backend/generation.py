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

from db import save_generated_design
from segmentation import run_segmentation

router = APIRouter(tags=["generation"])

# ── Style/color descriptions fed to the prompt-composition agent ──────────
# Keep these in sync with the style/color ids the app sends
# (lib/features/home/presentation/home_screen.dart).

_STYLE_DESCRIPTIONS: dict[str, str] = {
    "japandi": (
        "Japandi (a warm, minimalist blend of Japanese and Scandinavian "
        "design): light oak and pale wood furniture, clean simple lines, "
        "natural textures, uncluttered layout"
    ),
    "industrial_loft": (
        "Industrial / Loft: black steel-framed furniture, dark stained wood "
        "pieces, leather upholstery, and exposed-bulb metal light fixtures - "
        "the raw-material look comes from the furniture and fixtures "
        "themselves, not from changing what the walls/floor/ceiling are made of"
    ),
    "modern_luxury": (
        "Modern Luxury: furniture with polished marble-top tables and glass "
        "accents, metallic gold or chrome hardware and fixtures, glossy-finish "
        "furniture pieces, upscale contemporary silhouettes"
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
You are an interior-design prompt writer for an AI image generator. You will
be shown a photo of the actual room the user wants redecorated, along with
the requested room type, design style, and accent color.

Look at the photo first: note the room's shape, where the door(s) and
window(s) are, the camera angle, and how much open floor space there is.
Then write ONE vivid English paragraph (80-150 words) instructing the
generator how to redecorate the room, choosing furniture and placement that
suit THIS specific room's layout - not a generic showroom description.

Rules:
- Ground placement decisions in what you actually see in the photo (e.g.
  only describe putting a piece "against the far wall" if the photo shows
  that wall has room for it).
- Favor a smaller number of well-placed pieces over an exhaustive shopping
  list - do not force in more furniture than the space comfortably fits.
- The room must end up furnished appropriately for its type - never describe
  a bare or empty result.
- Use natural descriptive language only. Never use attention-weighting
  syntax such as (word:1.2) - the target model does not support it.
- Mention concrete furniture, materials, and lighting appropriate to the
  style and color.
- Do not mention changing the room layout, walls, windows, doors, or
  camera angle - those must be preserved as-is.
- Never mention adding, moving, upgrading, or restyling any electrical
  outlet, switch, vent, or wall-mounted fixture - don't describe them at
  all. They are handled separately and must stay exactly as in the photo.
- Never describe changing what a wall, ceiling, or floor is physically made
  of (e.g. turning a painted wall into exposed brick, concrete, stone, or
  wood paneling, or tile into hardwood). Express the requested style only
  through furniture, textiles, rugs, lighting fixtures, decor, and wall
  paint color - the underlying construction material never changes.
- Output only the paragraph, no headings or extra commentary.
""".strip()

# Hard structural constraints that must reach the image model on every
# single request, regardless of whether the prompt-composition agent
# succeeded or the static fallback kicked in. The composer only writes the
# style/furniture description - it is explicitly told not to talk about
# layout changes, but that alone doesn't put "keep the doors/windows/outlets
# visible" text in front of the image model, which is what actually keeps it
# from painting over them. This clause is appended after whichever prompt
# path ran, so it's never at the mercy of what the composer happened to say.
#
# Extended 2026-09-10 after Experiment A's blind-rated results: the two
# biggest human-flagged failure patterns beyond simple blocking were (1) the
# model hallucinating NEW outlets/switches that weren't in the original photo
# at all, and (2) a style like "industrial" being interpreted as literally
# changing wall material to brick/concrete, which is physically impossible
# without demolition. Neither was covered by the original wording, which only
# talked about not *covering* existing fixtures - added explicit rules for
# both below.
_STRUCTURE_PRESERVATION_CLAUSE = """
Preserve the original room layout, walls, floor, windows, doors, and all visible fixtures exactly as shown in the photo. Do not change the room structure, camera angle, or perspective.

Every door, window, electrical outlet, and light switch visible in the original photo must remain fully visible and unobstructed in the result - do not place furniture, decor, or any object in front of, over, or blocking them. Leave enough open walking space.

Do not add, duplicate, relocate, or restyle any electrical outlet, switch, vent, or wall-mounted fixture. The only outlets/switches allowed in the result are the exact ones already visible in the original photo, in their exact original positions - do not invent new ones anywhere else in the room, and do not feature or draw attention to them.

Walls, ceiling, and floor must keep their exact original construction material - repainting or recoloring a wall to match the requested palette is fine, but never change what a surface is physically made of (for example: never turn a plain/painted wall into exposed brick, concrete, stone, or wood paneling; never turn floor tile into hardwood or carpet, or vice versa) even if the requested style is normally associated with that material. Express the style entirely through furniture, textiles, rugs, lighting fixtures, and decor instead.

The result must contain furniture and decor appropriate to the room type - it must not be left empty, bare, or mostly unfurnished.

Return a photorealistic decorated room image only.
""".strip()


def build_static_fallback_prompt(room_type: str, style: str, color: str) -> str:
    """Used if the prompt-composition agent call fails for any reason, so a
    hiccup in that extra LLM call never breaks the /generate-room endpoint.
    """
    style_desc = _STYLE_DESCRIPTIONS.get(style, style)
    color_desc = _COLOR_DESCRIPTIONS.get(color, color)
    return (
        f"Redesign this uploaded {room_type} as a realistic interior in the "
        f"following style: {style_desc}. Use {color_desc}. "
        f"Add realistic furniture and decor that fit the room's scale and "
        f"style - the room must not be left empty.\n\n"
        f"{_STRUCTURE_PRESERVATION_CLAUSE}"
    )


async def compose_design_prompt(
    room_type: str, style: str, color: str, image_bytes: bytes, image_mime: str
) -> str:
    """Prompt-composition agent: turns the room/style/color ids into a single
    descriptive, natural-language paragraph via a Gemini call that also sees
    the actual room photo. Seeing the photo lets it ground furniture choice
    and placement in the room's real shape/door/window layout instead of
    describing a generic showroom scene that may not fit the space - a
    static or photo-blind prompt can otherwise place furniture in ways that
    look plausible in the abstract but awkward once matched against the
    real room. Because the result now depends on the specific uploaded
    photo (not just the id combo), it is no longer cached across requests
    the way the old text-only version was.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the backend.")

    model = os.getenv("GEMINI_PROMPT_MODEL", "gemini-3.6-flash")
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
                            f"Color: {color_desc}\n\n"
                            "Here is the room photo:"
                        ),
                    },
                    {
                        "inline_data": {
                            "mime_type": image_mime,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        },
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

    return response_data["candidates"][0]["content"]["parts"][0]["text"].strip()


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
3. The room actually contains furniture and decor appropriate to the room
   type - it is not empty, bare, or only has one or two minor items. FAIL if
   the "redesign" is essentially still an empty room with a different wall
   color or texture.
4. Every door, window, and visible electrical outlet/light switch that
   appears in the ORIGINAL photo is still physically usable in the
   REDECORATED photo. The test is strictly functional, not proximity: could
   a person actually open the door, reach the window, or plug something in
   WITHOUT moving the furniture first? FAIL only when a large, solid piece
   of furniture (a sofa, bed, cabinet, bookshelf, table, etc.) directly and
   completely blocks that function - e.g. pressed flush against a door so
   it cannot swing open at all, or physically covering an outlet's face so
   a plug cannot be inserted. PASS when furniture is merely positioned
   nearby, facing, or a couple of feet in front of a door/window/fixture
   but there is still real clearance to open it, walk around the furniture,
   or reach it - this is completely normal in real furnished rooms (e.g. a
   desk a few feet in front of a door, with room for the door to swing
   open past it, is fine and should PASS). When in doubt about whether
   clearance is real or not, PASS - this check exists for furniture that
   makes something truly unusable, not for furniture that is merely in the
   same general area. Do NOT fail for a curtain, lamp, cord, rug, plant, or
   small decor item merely appearing near or partially over it - those are
   cosmetic and normal in real decorated rooms.
5. No electrical outlet, switch, vent, or wall-mounted fixture appears in the
   REDECORATED photo that isn't in the same position in the ORIGINAL photo.
   FAIL if the AI added a new one, duplicated one, or moved one to a
   different spot - outlets/switches must match the original 1:1, not just
   "not be blocked."
6. Walls, ceiling, and floor keep the same physical construction material as
   the ORIGINAL photo - a color/paint change is fine, but FAIL if a surface's
   material clearly changed (e.g. a plain painted wall became exposed brick,
   concrete, stone, or wood paneling; floor tile became hardwood or vice
   versa), even if that material fits the requested style.
7. The room's overall layout, walls, and camera angle still look like the
   same room, not a different one.

Ignore minor imperfections - only fail on a clear, obvious violation of one
of the checks above (e.g. a window that disappeared, a door blocked by a
sofa, a hallucinated extra outlet, a wall that changed material, an empty
room, the wrong room type, or a style/color that is clearly not what was
asked for).

Reply with exactly one line: either "PASS" or "FAIL: <short reason naming
which check above it violates>".
""".strip()


async def critique_generated_image(
    original_bytes: bytes,
    original_mime: str,
    generated_bytes: bytes,
    generated_mime: str,
    room_type: str,
    style: str,
    color: str,
) -> tuple[bool, str]:
    """Critic Agent: asks Gemini Vision to compare the generated image against
    the original photo, checking both style/room-type intent and that doors,
    windows, and outlets/switches from the original weren't covered or
    removed. Returns (passed, raw_verdict_text). passed is True whenever the
    check can't be completed (no API key, network error, unexpected
    response) - a broken critic should never block a user's generation,
    only a *confirmed* mismatch should.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return True, ""

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
        return True, ""

    passed = verdict.upper().startswith("PASS")
    if not passed:
        print(f"critique_generated_image flagged a mismatch: {verdict}")
    return passed, verdict


# How many times the Critic Agent gets to reject and force a regeneration
# before we just accept whatever came out. 4 as of 2026-09-10 (was 3, 2, 1).
# At 3: 11/12 case2+case4 chains eventually passed, 1/12 (a nightstand vs.
# outlet placement conflict) still failed after all 3 retries - user asked
# to push to 4 for the full-scale run to give that class of stubborn case
# one more shot. Each extra retry level raises worst-case cost per
# condition by one more image-gen call - this is now a real cost driver,
# not a rounding error, at full scope (see run_experiment_a.py's printed
# estimate before spending).
CRITIC_MAX_RETRIES = int(os.getenv("CRITIC_MAX_RETRIES", "4"))


def critic_retry_reminder(verdict: str) -> str:
    """Builds a targeted retry addendum from the critic's actual FAIL reason,
    rather than a generic one - there are now several distinct failure modes
    (empty room, hallucinated outlet, wall material change, blocked
    fixture, ...) and telling the model exactly what it got wrong works
    better than a one-size-fits-all reminder.
    """
    reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    return (
        f"\n\nIMPORTANT: the previous attempt failed review for this specific "
        f"reason: {reason}. Fix exactly that issue this time, while still "
        f"following every instruction above."
    )


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
            # Replicate throttles harder once account credit drops below $5
            # (429, e.g. "reduced to 6 requests per minute") - the response
            # names how long to wait, and it's typically only ~1s, so a
            # short bounded retry here is enough. A real user hitting this
            # mid-request shouldn't see a hard failure over a 1-second wait.
            for retry_n in range(4):
                response = await client.post(
                    f"https://api.replicate.com/v1/models/{model}/predictions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Prefer": "wait=60",
                    },
                    json={"input": prediction_input},
                )
                if response.status_code != 429 or retry_n == 3:
                    response.raise_for_status()
                    break
                retry_after = float(response.json().get("retry_after", 1))
                print(f"Replicate rate-limited (429), retrying in {retry_after}s...")
                await asyncio.sleep(retry_after)
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
    provider: str | None = Form(None),
):
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    try:
        source_image = Image.open(BytesIO(image_bytes))
        source_image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        style_paragraph = await compose_design_prompt(
            room_type, style, color, image_bytes, mime_type
        )
        # The composer only writes the style/furniture description - the hard
        # "don't cover the doors/windows/outlets" constraint always gets
        # appended here rather than trusted to the composer's output.
        prompt = f"{style_paragraph}\n\n{_STRUCTURE_PRESERVATION_CLAUSE}"
    except Exception as exc:
        print(f"compose_design_prompt failed, falling back to static prompt: {exc}")
        prompt = build_static_fallback_prompt(room_type, style, color)

    # Client may pick a provider per-request (e.g. a model-selector UI);
    # falls back to the server's env default when not specified.
    provider = (provider or os.getenv("AI_IMAGE_PROVIDER", "gemini")).lower()

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
    # there's nothing to judge. Real providers get up to CRITIC_MAX_RETRIES
    # automatic retries if the attempt doesn't actually match what was
    # asked for, comparing against the original photo so it can specifically
    # catch a covered/missing door/window/outlet, a hallucinated new one, a
    # changed wall material, or a still-empty room - not just a wrong room
    # type/style. Each retry re-checks with the critic rather than assuming
    # the fix worked, since one correction pass doesn't always stick.
    if provider != "mock":
        gen_bytes, gen_mime = decode_data_url(generated_image)
        passed, verdict = await critique_generated_image(
            image_bytes, mime_type, gen_bytes, gen_mime, room_type, style, color
        )
        retries_used = 0
        while not passed and retries_used < CRITIC_MAX_RETRIES:
            retries_used += 1
            print(
                f"Critic Agent rejected attempt {retries_used} "
                f"(of {CRITIC_MAX_RETRIES} retries allowed): {verdict}"
            )
            prompt = f"{prompt}{critic_retry_reminder(verdict)}"
            generated_image = await run_provider()
            gen_bytes, gen_mime = decode_data_url(generated_image)
            passed, verdict = await critique_generated_image(
                image_bytes, mime_type, gen_bytes, gen_mime, room_type, style, color
            )

    response: dict[str, Any] = {
        "generated_image": generated_image,
        "products": mock_products(),
    }

    furniture_items: list[dict[str, Any]] = []
    if segment:
        _, encoded = generated_image.split(",", 1)
        gen_bytes = base64.b64decode(encoded)
        seg_result = await run_segmentation(gen_bytes, "image/png")
        response["furniture_segments"] = seg_result.model_dump()
        furniture_items = [item.model_dump() for item in seg_result.items]

    # Every generation gets persisted (is_saved=false) regardless of whether
    # segmentation ran — the favorite button on the results screen is what
    # later flips is_saved to true, which is what the history screen reads.
    generated_bytes_for_save, _ = decode_data_url(generated_image)
    design_id = await asyncio.to_thread(
        save_generated_design,
        room_type=room_type,
        style=style,
        color=color,
        source_bytes=image_bytes,
        source_mime=mime_type,
        generated_bytes=generated_bytes_for_save,
        furniture_items=furniture_items,
    )
    if design_id:
        response["design_id"] = design_id

    return response
