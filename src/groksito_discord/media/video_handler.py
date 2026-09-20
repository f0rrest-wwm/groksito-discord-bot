"""
Centralized, modern video generation handler for Groksito.

This module is the single source of truth for:
- Text-to-Video (T2V)
- Image-to-Video (I2V / animation from reference image)

Modeled directly after the modernized `media/image_handler.py` for consistency.

Key modernizations (following the image pattern):
- Dedicated `_enhance_video_prompt()` — always-on quality improvement, especially strong for Spanish prompts.
  Adds motion, camera, style, and quality enhancers while preserving user intent.
- Unified auth, HTTP, retry, and long-running polling logic.
- First-class **extra_params support for future API fields (motion_strength, etc.).
- Robust error handling for the longer video generation lifecycle (start + poll).
- Natural, consistent user-facing messages (matching image delivery style: neutral Spanish captions).
- Direct delivery via image_delivery (register + consume + reply) for natural "typing..." UX.
- I2V aspect ratio inferred from the reference image (avoids model guessing 16:9 on portrait/square art).
- Daily per-user video cap (default 5) + single in-flight lock. Owner IDs unlimited.
- Clean separation of concerns.

Canonical video handler; public dispatch functions keep stable signatures.

Implementation is fully compatible with the normal chat flow, tool selection, and
DIRECT_DELIVERY_PERFORMED sentinel pattern.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import httpx

from ..utils.correlation import cid_prefix
from ..config import settings
from .delivery import build_video_caption, deliver_from_request, register_image_request
from . import video_quota

# Bearer (OAuth preferred)
try:
    from ..core.grok_oauth import get_grok_bearer
except Exception:
    get_grok_bearer = None  # type: ignore

logger = logging.getLogger("groksito.media.video_handler")


# =============================================================================
# Common Helpers
# =============================================================================

def _sniff_image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Best-effort width/height from PNG/JPEG/WebP headers (no extra deps)."""
    if len(data) < 24:
        return None
    # PNG: IHDR chunk at bytes 16-23
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        if w > 0 and h > 0:
            return w, h
    # JPEG: scan for SOF0/SOF2
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h = int.from_bytes(data[i + 5 : i + 7], "big")
                w = int.from_bytes(data[i + 7 : i + 9], "big")
                if w > 0 and h > 0:
                    return w, h
            if i + 3 >= len(data):
                break
            seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
            i += 2 + max(seg_len, 2)
    # WebP (lossy): 'VP8 ' chunk
    if len(data) >= 30 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        if data[12:16] == b"VP8 ":
            w = int.from_bytes(data[26:28], "little") & 0x3FFF
            h = int.from_bytes(data[28:30], "little") & 0x3FFF
            if w > 0 and h > 0:
                return w, h
    return None


def _dimensions_to_aspect_ratio(width: int, height: int) -> str:
    """Map pixel dimensions to xAI's discrete aspect_ratio labels."""
    if width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    if ratio >= 1.25:
        return "16:9"
    if ratio <= 0.8:
        return "9:16"
    return "1:1"


async def _infer_aspect_ratio_from_image_url(url: str) -> str | None:
    """Download image header bytes and infer the closest standard aspect ratio."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200 or not resp.content:
                return None
            dims = _sniff_image_dimensions(resp.content[:65536])
            if not dims:
                return None
            w, h = dims
            ar = _dimensions_to_aspect_ratio(w, h)
            logger.info(f"{cid_prefix()}[Video] Inferred aspect_ratio={ar} from reference image ({w}x{h})")
            return ar
    except Exception as e:
        logger.warning(f"{cid_prefix()}[Video] Could not infer aspect ratio from reference image: {e}")
        return None


def _resolve_api_key() -> str | None:
    """Resolve credential preferring fresh OAuth token."""
    if get_grok_bearer:
        try:
            tok = get_grok_bearer()
            if tok:
                return tok
        except Exception:
            pass
    return (
        os.getenv("XAI_API_KEY")
        or getattr(settings, "xai_api_key", None)
    )


# --- Modern Prompt Engineering for Video (new, modeled on image enhancer) ---

def _enhance_video_prompt(prompt: str, is_from_image: bool = False) -> str:
    """
    Always-on prompt enhancer for better video results, with special care for Spanish.

    - Detects Spanish-dominant prompts and keeps the user's voice.
    - Appends high-value motion, camera, physics, and quality descriptors.
    - For I2V (is_from_image=True) adds subtle animation guidance.
    - Style detection (gothic, cyberpunk, anime, etc.) mirrors the image enhancer.
    - Never changes the core subject/action the user requested.
    - Keeps output reasonable length for the video API.
    """
    if not prompt or len(prompt.strip()) < 2:
        base = "a beautiful stylized character in a simple scene with gentle motion"
        return base

    p = prompt.strip()
    lower = p.lower()

    # Simple Spanish detection (accents + common words)
    spanish_score = sum(
        1 for marker in (" el ", " la ", " de ", " una ", " con ", " para ", " que ", " está ")
        if marker in f" {lower} "
    )
    looks_spanish = spanish_score >= 1 or any(ch in p for ch in "áéíóúñü¿¡")

    enhancers: list[str] = []

    # Motion & camera (core for video quality)
    motion_hints = []
    if any(k in lower for k in ("cámara", "camera", "pan", "zoom", "movimiento", "mueve", "caminando", "corriendo")):
        motion_hints.append("smooth camera movement")
    else:
        motion_hints.append("natural smooth motion")

    if any(k in lower for k in ("acción", "lucha", "explosión", "rápido", "dinámico")):
        motion_hints.append("dynamic action, subtle physics")
    else:
        motion_hints.append("gentle cinematic motion, natural timing")

    enhancers.extend(motion_hints)

    # Style detection (shared with image system for consistency)
    if any(k in lower for k in ("gótica", "goth", "gotica", "dark", "negra", "oscura", "vampira")):
        enhancers.append("gothic cinematic style, moody dramatic lighting")
    elif any(k in lower for k in ("cyberpunk", "neon", "futurista", "sci-fi", "cyber")):
        enhancers.append("cyberpunk aesthetic, vibrant neon lighting, high-tech motion")
    elif any(k in lower for k in ("anime", "waifu", "manga", "chibi", "2d")):
        enhancers.append("detailed anime style, fluid animation, vibrant colors")
    elif any(k in lower for k in ("realista", "photoreal", "foto", "real")):
        enhancers.append("photorealistic live-action style, natural motion blur")
    else:
        enhancers.append("high-quality stylized animation, clean motion")

    # Quality / production value
    if not any(q in lower for q in ("detall", "calidad", "masterpiece", "cinematic", "4k", "alta")):
        enhancers.append("sharp details, high production quality, smooth animation")

    # I2V specific guidance (encourage faithful animation of the reference)
    if is_from_image:
        enhancers.append("animate the subject naturally from the reference image, coherent motion")

    # Assemble: user prompt first, then enhancers (comma separated)
    enhancer_str = ", ".join(enhancers)
    if any(w in p.lower() for w in ("estilo", "style", "animación", "motion", "movimiento", "cinematic")):
        final = f"{p}, {enhancer_str}"
    else:
        final = f"{p}, {enhancer_str}"

    final = " ".join(final.split()).strip().strip(",")

    # Reasonable cap for video prompts (videos are sensitive to length)
    if len(final) > 380:
        final = final[:377].rstrip() + "..."

    return final


# =============================================================================
# Video Polling (improved logging + resilience)
# =============================================================================

def _video_poll_max_wait_seconds(*, duration: int = 6, resolution: str = "480p") -> int:
    """Wall-clock budget for polling xAI video jobs (aligned with xAI SDK 10m default)."""
    base = int(getattr(settings, "video_poll_max_wait_seconds", 600))
    bonus = max(0, duration - 6) * 20
    if str(resolution).lower() == "720p":
        bonus += 120
    return max(base, 300 + bonus)


async def _poll_for_video_completion(
    http_client: httpx.AsyncClient,
    request_id: str,
    api_key: str,
    max_wait_seconds: int = 300,
    poll_interval: float = 5.0
) -> tuple[str, dict]:
    """Poll the video status endpoint until done/failed/expired/timeout or terminal HTTP error.

    Key fix: always inspect HTTP status_code. 4xx (e.g. 400) on status checks means
    the job has failed or the id is no longer valid on the backend — fail fast instead
    of spinning until the hard timeout.
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    consecutive_bad = 0
    poll_count = 0

    while True:
        poll_count += 1
        try:
            resp = await http_client.get(
                f"https://api.x.ai/v1/videos/{request_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            sc = resp.status_code
            try:
                data = resp.json() if resp.content else {}
            except Exception:
                data = {"_raw": (resp.text or "")[:300]}

            if sc < 400:
                # 2xx (incl. 200 success + 202 Accepted) and 3xx are non-error responses
                consecutive_bad = 0
                status = str((data or {}).get("status", "")).lower().strip()
                if status == "done":
                    return "succeeded", data
                if status in ("failed", "expired"):
                    return "failed", data
                # pending / empty body on 202 -> continue polling
                if poll_count <= 2 or poll_count % 6 == 0:
                    logger.debug(
                        f"{cid_prefix()}[Video] poll #{poll_count} for {request_id}: "
                        f"http={sc} status={status or 'unknown'}"
                    )
            else:
                consecutive_bad += 1
                logger.warning(
                    f"{cid_prefix()}[Video] Poll HTTP {sc} for {request_id} "
                    f"(poll#{poll_count}, consec={consecutive_bad}): {str(data)[:180]}"
                )
                if sc in (400, 404, 403, 410, 422):
                    # Terminal: job failed server-side or request became invalid.
                    # Return immediately so the turn can finish (instead of long spin).
                    return "failed", data
                if consecutive_bad >= 3:
                    # Persistent 5xx/429 etc — do not block the user for full timeout.
                    return "failed", data or {"error": {"message": f"persistent poll HTTP {sc}"}}

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            consecutive_bad += 1
            logger.warning(f"{cid_prefix()}[Video Polling] Transient network for {request_id}: {type(e).__name__}")
            if consecutive_bad >= 4:
                return "failed", {"error": {"message": "too many transient poll errors"}}
            await asyncio.sleep(min(10.0, poll_interval * 1.5))
            # re-evaluate timeout; skip normal sleep this iteration
            if loop.time() - start_time > max_wait_seconds:
                return "timeout", {}
            continue

        except Exception as e:
            consecutive_bad += 1
            logger.warning(f"{cid_prefix()}[Video Polling] Unexpected error for {request_id}: {e}")
            if consecutive_bad >= 4:
                return "failed", {"error": {"message": str(e)[:120]}}
            if loop.time() - start_time > max_wait_seconds:
                return "timeout", {}
            continue

        elapsed = loop.time() - start_time
        if elapsed > max_wait_seconds:
            logger.warning(
                f"{cid_prefix()}[Video] Poll timeout after {elapsed:.0f}s "
                f"(max={max_wait_seconds}s) for {request_id} — still in progress on xAI"
            )
            return "timeout", {}

        # normal or backoff sleep (only on clean poll cycle)
        sleep_for = poll_interval
        if consecutive_bad > 0:
            sleep_for = min(15.0, poll_interval * (1 + 0.5 * min(consecutive_bad, 4)))
        await asyncio.sleep(sleep_for)


# =============================================================================
# Video Schema (kept identical for compatibility)
# =============================================================================

def _generate_video_schema() -> dict:
    from ..llm.prompt_builder import VIDEO_TOOL_DESCRIPTION_FULL
    return {
        "type": "function",
        "name": "generate_video",
        "description": VIDEO_TOOL_DESCRIPTION_FULL,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Scene, action, or animation guidance for the video."
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds (default 6 if not specified).",
                    "default": 6
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Text-to-video only (16:9, 9:16, 1:1). Omit for image-to-video.",
                },
                "resolution": {
                    "type": "string",
                    "description": "Resolution: 480p or 720p. Defaults to 480p if not specified by the user.",
                    "default": "480p"
                }
            },
            "required": ["prompt"]
        }
    }


def _generate_video_schema_tiny() -> dict:
    """Minimal schema for native Grok video decisions on addressed turns (mirrors image tiny)."""
    from ..llm.prompt_builder import VIDEO_TOOL_DESCRIPTION_TINY
    return {
        "type": "function",
        "name": "generate_video",
        "description": VIDEO_TOOL_DESCRIPTION_TINY,
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Scene, action, or animation guidance for the video.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in seconds (default 6 if not specified).",
                    "default": 6,
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Text-to-video only (16:9, 9:16, 1:1). Omit for image-to-video.",
                },
                "resolution": {
                    "type": "string",
                    "description": "Resolution: 480p or 720p. Defaults to 480p if not specified by the user.",
                    "default": "480p"
                },
            },
            "required": ["prompt"],
        },
    }


# =============================================================================
# Core Video Tool Implementation (modernized)
# =============================================================================

async def _tool_generate_video(
    prompt: str,
    duration: int = 6,
    aspect_ratio: str | None = None,
    request_id: Optional[str] = None,
    source_image_url: str | None = None,
    resolution: str = "480p",
    **extra_params: Any,
) -> str:
    """
    Core video generation (T2V + I2V).

    Modernizations:
    - Always calls _enhance_video_prompt() for better Spanish + motion results.
    - **extra_params passed through (motion_strength, etc.).
    - Improved transient retry + clear error messages.
    - Natural delivery text consistent with the image system (neutral Spanish captions).
    """
    api_key = _resolve_api_key()
    if not api_key:
        return "No xAI credential configured for video generation (run --login-oauth or set XAI_API_KEY)."

    enforced_duration = min(max(duration, 3), 15)  # Grok max ~15s; default to 6s if user doesn't specify
    # Use user-specified resolution if provided (e.g. 720p), else default 480p
    final_resolution = resolution if resolution else "480p"
    is_from_image = bool(source_image_url)

    # === Modern prompt enhancement (always-on) ===
    enhanced_prompt = _enhance_video_prompt(prompt, is_from_image=is_from_image)

    resolved_aspect_ratio: str | None = aspect_ratio
    if is_from_image and source_image_url:
        inferred = await _infer_aspect_ratio_from_image_url(source_image_url)
        if inferred:
            if aspect_ratio and aspect_ratio != inferred:
                logger.info(
                    f"{cid_prefix()}[Video] Overriding model aspect_ratio={aspect_ratio} "
                    f"with {inferred} from reference image (prevents stretch)"
                )
            resolved_aspect_ratio = inferred
        else:
            # Grok web: let the API follow the reference image when we can't sniff dimensions.
            resolved_aspect_ratio = None
            if aspect_ratio:
                logger.info(
                    f"{cid_prefix()}[Video] Dropping model aspect_ratio={aspect_ratio} for I2V "
                    "(reference image should drive framing)"
                )

    try:
        video_payload: dict = {
            "model": extra_params.get("model", "grok-imagine-video"),
            "prompt": enhanced_prompt,
            "duration": enforced_duration,
            "resolution": final_resolution,
        }
        if resolved_aspect_ratio:
            video_payload["aspect_ratio"] = resolved_aspect_ratio

        # I2V support
        if source_image_url:
            video_payload["image_url"] = source_image_url
            video_payload["image"] = {"url": source_image_url}

        # Pass through modern/future params (motion_strength, etc.)
        for k in ("motion_strength", "guidance", "seed", "negative_prompt"):
            if k in extra_params and extra_params[k] is not None:
                video_payload[k] = extra_params[k]

        max_attempts = getattr(settings, "api_max_retries", 3)
        poll_data: dict = {}
        xai_video_id = None

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=settings.api_timeout_seconds) as http_client:
                    response = await http_client.post(
                        "https://api.x.ai/v1/videos/generations",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json=video_payload
                    )

                    if response.status_code != 200:
                        # Basic error surface for start failure
                        try:
                            err = response.json()
                            msg = err.get("error", {}).get("message", response.text)
                        except Exception:
                            msg = response.text[:200]
                        return f"Error starting video generation: {msg}"

                    data = response.json()
                    xai_video_id = data.get("id") or data.get("request_id") or data.get("video_id")

                    if not xai_video_id:
                        return "Error generating video: the API did not return a request ID."

                    poll_max = _video_poll_max_wait_seconds(
                        duration=enforced_duration,
                        resolution=final_resolution,
                    )
                    logger.info(
                        f"{cid_prefix()}[Video] Polling for completion of xAI video request "
                        f"{xai_video_id} (max_wait={poll_max}s, {final_resolution}, {enforced_duration}s)"
                    )
                    poll_status, poll_data = await _poll_for_video_completion(
                        http_client,
                        xai_video_id,
                        api_key,
                        max_wait_seconds=poll_max,
                        poll_interval=5.0,
                    )

                    if poll_status != "succeeded":
                        err_detail = ""
                        if isinstance(poll_data, dict):
                            err_obj = poll_data.get("error")
                            if isinstance(err_obj, dict):
                                err_detail = err_obj.get("message") or ""
                            elif err_obj:
                                err_detail = str(err_obj)
                            if not err_detail:
                                err_detail = poll_data.get("status") or poll_data.get("_raw") or ""
                        return f"Video generation {poll_status}. {err_detail}. Try a simpler prompt or different reference."

                    break  # success

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as net_err:
                if attempt < max_attempts - 1:
                    delay = 0.6 * (2 ** attempt)
                    logger.info(f"{cid_prefix()}[Video] Transient network on generation attempt {attempt+1}: {type(net_err).__name__} — retry in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                return f"Error generating video: transient network issue after retries ({type(net_err).__name__})."

        else:
            return "Error generating video after retries."

        # === Extract final video URL (robust, same defensive logic as before) ===
        video_url = None
        if isinstance(poll_data, dict):
            for key in ("url", "video_url", "download_url", "result_url", "file_url", "src", "mp4"):
                val = poll_data.get(key)
                if val and isinstance(val, str) and val.startswith("http"):
                    video_url = val
                    break

            if not video_url:
                for container_key in ("result", "output", "data", "video", "asset", "content"):
                    container = poll_data.get(container_key)
                    if isinstance(container, dict):
                        for key in ("url", "video_url", "download_url", "result_url", "file_url", "src"):
                            val = container.get(key)
                            if val and isinstance(val, str) and val.startswith("http"):
                                video_url = val
                                break
                        if video_url:
                            break
                    elif isinstance(container, list) and container:
                        for item in container:
                            if isinstance(item, dict):
                                for key in ("url", "video_url", "download_url", "src"):
                                    val = item.get(key)
                                    if val and isinstance(val, str) and val.startswith("http"):
                                        video_url = val
                                        break
                                if video_url:
                                    break
                        if video_url:
                            break

            if not video_url:
                def _find_video_urls(obj, found_list):
                    if len(found_list) > 0:
                        return
                    if isinstance(obj, dict):
                        for v in obj.values():
                            _find_video_urls(v, found_list)
                    elif isinstance(obj, list):
                        for v in obj:
                            _find_video_urls(v, found_list)
                    elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
                        low = obj.lower()
                        if any(h in low for h in (".mp4", ".webm", ".mov", "/video", "grok", "x.ai", "cdn", "download")):
                            found_list.append(obj)

                found = []
                _find_video_urls(poll_data, found)
                if found:
                    video_url = found[0]

        if not video_url:
            try:
                top_keys = list(poll_data.keys()) if isinstance(poll_data, dict) else str(type(poll_data))
                logger.warning(f"{cid_prefix()}[Video] Succeeded poll but no URL extracted. Top keys: {top_keys}. Raw sample: {str(poll_data)[:400]}")
            except Exception:
                pass
            return "The video was generated but no downloadable URL was obtained. Try again later."

        caption = build_video_caption(
            from_image=is_from_image,
            duration=enforced_duration,
            prompt=prompt,
        )

        if request_id and await deliver_from_request(
            request_id, caption=caption, urls=[video_url], kind="video"
        ):
            logger.info(
                f"{cid_prefix()}[MediaDelivery] Video delivered as attachment for request {request_id} "
                f"(xAI id: {xai_video_id})"
            )
            from ..llm.prompt_builder import DIRECT_DELIVERY_SUCCESS_VIDEO
            return DIRECT_DELIVERY_SUCCESS_VIDEO

        return f"Video generated successfully:\n{video_url}"

    except Exception as e:
        logger.exception(f"{cid_prefix()}[Video] Unexpected error in _tool_generate_video")
        return f"Error generating video: {str(e)}"


# =============================================================================
# Public Dispatch Handler (stable API)
# =============================================================================

async def _handle_generate_video(args: dict, original_message: Any, image_urls: list[str] | None = None) -> str:
    """
    Handles generate_video dispatch.

    - Request registration for direct delivery.
    - Daily quota + one-video-at-a-time lock.
    - Calls the modern _tool_generate_video (with prompt enhancement).
    """
    prompt = args.get("prompt", "")

    duration = int(args.get("duration", 6))
    aspect_ratio = args.get("aspect_ratio") or args.get("aspect") or None
    resolution = args.get("resolution", "480p")

    source_image_url = image_urls[0] if image_urls else None

    author = getattr(original_message, "author", None)
    user_id = getattr(author, "id", 0) or 0
    display_name = (
        getattr(author, "display_name", None)
        or getattr(author, "name", None)
        or ""
    )

    if video_quota.is_nsfw_request(prompt):
        logger.info(f"{cid_prefix()}[Video] NSFW blocked user={user_id}")
        return video_quota.NSFW_REPLY

    blocked = video_quota.check_can_start(user_id, display_name)
    if blocked:
        logger.info(f"{cid_prefix()}[Video] blocked user={user_id}: {blocked}")
        return blocked

    request_id = None
    try:
        request_id = await register_image_request(
            user_id=user_id,
            channel_id=getattr(original_message, "channel", None) and getattr(original_message.channel, "id", 0) or 0,
            message_id=getattr(original_message, "id", 0),
            operation_type="video",
            original_message=original_message,
        )
    except Exception:
        pass

    try:
        result = await _tool_generate_video(
            prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            request_id=request_id,
            source_image_url=source_image_url,
            resolution=resolution,
            **{k: v for k, v in args.items() if k not in ("prompt", "duration", "aspect_ratio", "aspect", "resolution")}
        )
        text = result if isinstance(result, str) else ""
        err = text.lower().startswith("error") or "could not" in text.lower()
        if text and not err:
            used = video_quota.mark_success(user_id, display_name)
            logger.info(f"{cid_prefix()}[Video] quota user={user_id} used={used}/{video_quota.DAILY_LIMIT}")
        return result
    finally:
        video_quota.release_lock()


# Lazy module-level re-export for DIRECT_* (test_guidance_centralization).
# Avoids top-level llm.prompt_builder import (cycle risk).
def __getattr__(name: str):
    if name == "DIRECT_DELIVERY_SUCCESS_VIDEO":
        from ..llm.prompt_builder import DIRECT_DELIVERY_SUCCESS_VIDEO
        return DIRECT_DELIVERY_SUCCESS_VIDEO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
