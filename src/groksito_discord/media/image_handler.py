"""
Image generation and editing (xAI Imagine). Single source of truth for generate_image / edit_image tools.

Uses media/delivery.py for direct delivery (DIRECT_DELIVERY_PERFORMED sentinel). See ARCHITECTURE.md.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import unicodedata
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from ..utils.correlation import cid_prefix
from ..config import settings
from .delivery import (
    _download_url,
    build_edit_caption,
    build_image_caption,
    consume_image_request,
    deliver_from_request,
    register_image_request,
)

# Bearer resolution (OAuth preferred with refresh, fallback to key)
try:
    from ..core.grok_oauth import get_grok_bearer
except Exception:
    get_grok_bearer = None  # type: ignore

logger = logging.getLogger("groksito.media.image_handler")


# =============================================================================
# Common Helpers (auth, prompt engineering, error classification)
# =============================================================================

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


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# =============================================================================
# IP / Copyright concept rewriter (for "reimaginación de esencia")
# =============================================================================

# Famous IPs / brands that commonly trigger upstream image policy blocks.
# We never want to send these names (or close variants) to the image API.
_KNOWN_IP_TRIGGERS = {
    "zelda", "princess zelda", "link", "legend of zelda", "nintendo",
    "mario", "luigi", "bowser", "pokemon", "pikachu", "ash ketchum",
    "disney", "cinderella", "ariel", "elsa", "anna", "moana", "mickey",
    "marvel", "spiderman", "iron man", "captain america", "thor", "deadpool",
    "star wars", "darth vader", "yoda", "lightsaber",
    "harry potter", "hermione", "voldemort", "hogwarts",
    "sonic", "sega", "final fantasy", "cloud strife", "sephiroth",
    "game of thrones", "daenerys", "jon snow",
}

# Signals that the user wants a *reimagination / "what if another creator made this"* concept.
_REIMAGINATION_SIGNALS = {
    "si el creador", "si hubiera sido", "si lo hubiera hecho", "como si", "reimagin",
    "reimaginacion", "reimaginación", "versión de", "versión miyazaki", "estilo miyazaki",
    "miyazaki", "fromsoftware", "dark souls", "elden ring", "bloodborne", "soulslike",
    "what if", "en vez de", "pero hecho por", "pero por", "si fuera", "director", "creador fuera",
}


def _contains_ip_trigger(text: str) -> bool:
    """True if the text mentions a known protected IP/character/brand."""
    if not text:
        return False
    t = _strip_accents(text.lower())
    for trigger in _KNOWN_IP_TRIGGERS:
        if trigger in t:
            return True
    return False


def _is_reimagination_request(text: str) -> bool:
    """True if this looks like the user wants the 'essence/vibe' of something famous but reimagined by another creator."""
    if not text:
        return False
    t = _strip_accents(text.lower())
    has_ip = _contains_ip_trigger(t)
    has_signal = any(sig in t for sig in _REIMAGINATION_SIGNALS)
    return has_ip and has_signal


def _rewrite_to_concept_essence(original: str) -> str:
    """
    Rewrites prompts that are trying to capture the *essence* of a famous IP
    (especially "The Legend of Zelda if it had been made by Hidetaka Miyazaki")
    into safe, high-signal prompts that avoid naming protected characters/brands.

    Goal: preserve the artistic intent ("ancient mysterious kingdom, exploration,
    temples, silent heroine, wonder + melancholy") + the requested alternative style
    (Miyazaki grim/dark fantasy, soulslike atmosphere, etc.).
    """
    if not original:
        return original

    p = original.strip()
    lower = _strip_accents(p.lower())

    # Detect the target "alternative style" the user actually wants
    wants_miyazaki = any(k in lower for k in (
        "miyazaki", "fromsoftware", "dark souls", "elden ring", "bloodborne",
        "souls", "soulslike", "grim", "melancol", "dread", "ruinas", "ruined"
    ))

    # Base concept for "Legend of Zelda" essence without naming it
    # (exploration, ancient kingdom, mysterious temples, a quiet legendary figure, forests, adventure)
    zelda_essence = (
        "vast ancient kingdom of forgotten glory, mysterious overgrown temples and crumbling stone shrines "
        "hidden in misty cursed forests, a silent heroic figure with sword and shield exploring lost ruins, "
        "intricate environmental storytelling, sense of faded legend and quiet wonder"
    )

    if wants_miyazaki:
        # The exact thing the user asked: Zelda's essence through Miyazaki's lens
        core = (
            f"melancholic dark fantasy world in the distinctive style of Hidetaka Miyazaki, {zelda_essence}, "
            "grim yet beautiful atmosphere, oppressive yet poetic ruined architecture, dramatic god rays "
            "and volumetric lighting, profound sense of lost history, quiet dread mixed with awe, "
            "high fantasy adventure with soulslike melancholy"
        )
    else:
        # Generic safe reimagination fallback
        core = (
            f"reimagined dark fantasy epic: {zelda_essence}, intricate ruined temples, "
            "atmospheric misty landscapes, legendary silent explorer, rich environmental detail, "
            "dramatic cinematic lighting"
        )

    # Preserve any extra artistic instructions the user gave (lighting, mood, composition, etc.)
    extra_hints = []
    for hint in ("iluminacion", "lighting", "dramatica", "atmósfera", "atmosphere", "detall", "cinematic",
                 "volumetric", "god rays", "chiaroscuro", "epic", "melancol", "oscura", "dark", "ruinas"):
        if hint in lower:
            extra_hints.append(hint)

    if extra_hints:
        core = f"{core}, {', '.join(extra_hints[:4])}"

    core = re.sub(r"\s+", " ", core).strip().strip(",")
    if len(core) > 420:
        core = core[:417].rstrip() + "..."
    return core


# --- Modern Prompt Engineering (always-on quality pass + Spanish-friendly) ---

def _enhance_prompt_for_api(original: str, is_edit: bool = False) -> str:
    """Light rewrite close to grok.com Imagine. Do not force illustration."""
    if not original or len(original.strip()) < 2:
        return original or (
            "a clear well-lit photograph of the requested subject"
            if not is_edit
            else "keep the subject, apply the requested change only"
        )

    p = original.strip()
    lower = _strip_accents(p.lower())

    if _is_reimagination_request(p) and not is_edit:
        rewritten = _rewrite_to_concept_essence(p)
        if rewritten and rewritten != p:
            p = rewritten
            lower = _strip_accents(p.lower())

    wants_photo = any(
        k in lower
        for k in (
            "photo", "photograph", "photoreal", "realistic", "realista",
            "realism", "35mm", "camera", "live action", "live-action",
            "real person", "real people", "cinematic still", "promo poster",
            "screenshot", "foto",
        )
    )
    wants_art = any(
        k in lower
        for k in (
            "anime", "waifu", "manga", "chibi", "cartoon", "sticker",
            "comic", "illustration", "painting", "drawing", "ink wash",
            "ilustracion",
        )
    )

    extras: list[str] = []
    if wants_photo and not wants_art:
        extras.append(
            "photoreal live-action photograph, natural skin and fabric, "
            "real lighting, 35mm, no illustration, no anime, no CGI, no cartoon"
        )
    elif wants_art and not wants_photo:
        extras.append("commit to the requested art style, clean detailed artwork")
    else:
        extras.append("high detail, natural lighting, follow the user's medium")

    if is_edit:
        extras.append(
            "preserve identity, face, outfit and pose from the reference unless asked to change them"
        )
        extras.append("crop out black bars, letterboxing and phone UI")

    extras.append("fully clothed, non-sexual")

    final = f"{p}, {', '.join(extras)}"
    final = re.sub(r"\s+", " ", final).strip().strip(",")
    if len(final) > 1200:
        final = final[:1197].rstrip() + "..."
    return final


# --- Safety softening (only for 422/policy retries - invisible to user) ---
# (Preserved and lightly modernized from previous permissive implementation)

def _de_risk_text(text: str) -> str:
    if not text:
        return text
    p = text
    risky = [
        (r"\b(horny)\b", "intense"),
        (r"\b(lewd|thicc|thic)\b", "curvy"),
    ]
    for pat, repl in risky:
        p = re.sub(pat, repl, p, flags=re.IGNORECASE)
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r",\s*,+", ",", p).strip().strip(",")
    p = re.sub(r"\ba elegant\b", "an elegant", p, flags=re.IGNORECASE)
    return p.strip()


def _ultra_safe_artistic_fallback(original_prompt: str) -> str:
    if not original_prompt:
        return "elegant refined artistic character portrait, highly detailed, tasteful composition, dramatic lighting, cinematic illustration"

    # If this was a reimagination request, prefer the smart concept rewriter even in ultra fallback
    if _is_reimagination_request(original_prompt):
        concept = _rewrite_to_concept_essence(original_prompt)
        if concept:
            return concept

    orig = original_prompt.strip()
    lower = _strip_accents(orig.lower())

    bad_stems = [
        "desnud", "nude", "naked", "sin ropa", "porn", "hentai", "xxx", "ahegao",
        "pussy", "vagina", "pene", "verga", "dick", "cock", "follar", "fuck", "sexo expl",
        "spread leg explicit", "pierna abierta sexual"
    ]
    safe_words = [
        w for w in orig.split()
        if len(w) > 1 and not any(b in _strip_accents(w).lower() for b in bad_stems)
    ]
    subject = " ".join(safe_words[:6]) or "character"

    style = ""
    if any(k in lower for k in ["gotica", "gothic", "goth"]):
        style = "gothic style, "
    elif any(k in lower for k in ["cyberpunk", "futurist", "neon", "sci-fi"]):
        style = "cyberpunk style, "

    base = f"{style}fantasy character of {subject}, curvy stylized, dramatic lighting, detailed artistic rendering, SFW fantasy"
    base = re.sub(r"\s+", " ", base).strip().strip(",")
    if len(base) > 380:
        base = base[:377].rstrip() + "..."
    return base


def _soften_prompt_for_artistic(prompt: str) -> str:
    """
    Relaxed/permissive safety rewrite used ONLY on actual policy/422 errors.
    Keeps bikini/lingerie/tetona/culona/curvy/sexy fantasy intent. Only remaps hard blocks.
    """
    if not prompt:
        return "a beautiful stylized artistic character portrait in dramatic lighting"

    original = prompt.strip()
    p = original

    replacements = [
        (r"\b(culona|culazo|nalgona|culon)\b", "curvy wide hips"),
        (r"\b(tetona|tetas?)\b", "busty tetona"),
        (r"\b(nude|naked|desnuda|desnudo|fully naked|completely nude|sin ropa)\b", "in a tiny sexy bikini"),
        (r"\btopless\b", "in a revealing low-cut bikini top"),
        (r"\bbottomless\b", "in a micro bikini bottom"),
        (r"\b(pussy|vagina|clit|clitoris)\b", "intimate area"),
        (r"\b(dick|cock|pene|verga|pija|balls)\b", ""),
        (r"\b(erect|erection|hard dick)\b", "intense expression"),
        (r"\b(follar|coger|fucking|fuck|having sex|intercourse|penetrat)\b", "intense dynamic pose"),
        (r"\b(blowjob|oral sex|69|cum|semen)\b", "dramatic expression"),
        (r"\b(nsfw|porn|porno|hentai|xxx|lewd|ecchi)\b", "artistic fantasy"),
        (r"\b(ahegao)\b", "ecstatic expression"),
        (r"\b(spread legs|legs spread|ass up|piernas abiertas|en cuatro|a cuatro patas)\b", "dynamic pose"),
    ]

    for pattern, repl in replacements:
        p = re.sub(pattern, repl, p, flags=re.IGNORECASE)

    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r",\s*,+", ",", p)
    p = re.sub(r"^[,\s]+", "", p)
    p = re.sub(r"[,\s]+$", "", p)
    p = p.strip()

    if not p or len(p) < 2:
        words = original.split()
        safe_words = [w for w in words if not any(bad in _strip_accents(w).lower() for bad in ["culo", "teta", "nude", "naked", "sex", "porn", "folla", "verga"])]
        p = " ".join(safe_words[:7]) or "stylized character"

    p = _de_risk_text(p)

    lower_orig = _strip_accents(original.lower())
    if any(k in lower_orig for k in ["gotica", "gothic", "goth", "dark", "negra", "oscura"]):
        enhancer = "gothic style, dramatic lighting, detailed rendering"
    elif any(k in lower_orig for k in ["cyberpunk", "futurist", "neon", "sci-fi"]):
        enhancer = "cyberpunk style, cinematic neon lighting, detailed"
    else:
        enhancer = "dramatic lighting, detailed artistic rendering"

    if any(w in p.lower() for w in ["style", "artistic", "illustration", "render", "masterpiece", "cinematic", "lighting"]):
        final = p
    else:
        final = f"{p}, {enhancer}"

    final = re.sub(r"\s+", " ", final).strip().strip(",")
    final = _de_risk_text(final)
    if len(final) > 380:
        final = final[:377].rstrip() + "..."
    return final


def soften_image_prompt(original_prompt: str) -> str:
    """Public helper for the error-path softening (kept for compatibility)."""
    if not original_prompt or len(original_prompt.strip()) < 3:
        return original_prompt or "a beautiful stylized artistic character portrait in dramatic lighting"
    return _soften_prompt_for_artistic(original_prompt)





# =============================================================================
# Core Generation
# =============================================================================

async def _tool_generate_image(
    prompt: str,
    count: int = 1,
    request_id: Optional[str] = None,
    aspect_ratio: str | None = None,
    user_caption_source: str | None = None,  # preferred for natural language in delivery caption
    **extra_params: Any,   # future-proofing for quality, style, etc.
) -> str:
    """
    Modern centralized image generation.

    - Always applies _enhance_prompt_for_api first for better Spanish + artistic results.
    - Original user prompt is used verbatim for all user-facing delivery text.
    - Safety softening only on real 422/policy (silent, never visible to user or model final output).
    - Excellent error handling with direct clean delivery for policy blocks.
    """
    api_key = _resolve_api_key()
    if not api_key:
        return "No xAI credential configured for image generation (run --login-oauth or set XAI_API_KEY)."

    requested_prompt = prompt
    # Modern improvement: always enhance for quality (Spanish support + artistic polish)
    current_prompt = _enhance_prompt_for_api(requested_prompt, is_edit=False)

    try:
        max_attempts = getattr(settings, "api_max_retries", 3)
        for attempt in range(max_attempts):
            async with httpx.AsyncClient(timeout=settings.api_timeout_seconds) as http_client:
                generation_payload = {
                    "model": extra_params.get("model", "grok-imagine-image-quality"),
                    "prompt": current_prompt,
                    "n": min(count, 4),
                    "response_format": "url",
                }
                if aspect_ratio:
                    generation_payload["aspect_ratio"] = aspect_ratio
                # Pass through any future modern params the API may support
                for k in ("quality", "style", "guidance_scale"):
                    if k in extra_params and extra_params[k] is not None:
                        generation_payload[k] = extra_params[k]

                try:
                    response = await http_client.post(
                        "https://api.x.ai/v1/images/generations",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json=generation_payload
                    )
                except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as net_err:
                    if attempt < max_attempts - 1:
                        delay = 0.4 * (2 ** attempt)
                        logger.info(f"{cid_prefix()}[Image] Transient network on generate {attempt+1}: {type(net_err).__name__} — retry {delay:.1f}s")
                        await asyncio.sleep(delay)
                        continue
                    raise

                if response.status_code != 200:
                    err_msg = "desconocido"
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            err_obj = data.get("error", {}) or {}
                            if isinstance(err_obj, dict):
                                err_msg = str(err_obj.get("message", "") or "").lower()
                            else:
                                err_msg = str(err_obj).lower()
                        else:
                            err_msg = str(data)[:200].lower()
                    except Exception:
                        err_msg = (response.text or "")[:300].lower()

                    is_policy = (
                        response.status_code in (422, 400) or
                        any(kw in err_msg for kw in ("policy", "safety", "content", "violation", "censored",
                                                     "inappropriate", "blocked", "forbidden", "refuse", "moderation"))
                    )

                    if is_policy and attempt == 0:
                        # Silent safety retry (permissive mode)
                        # First try normal sexual softening
                        softened = soften_image_prompt(requested_prompt)
                        if softened and softened != current_prompt:
                            logger.info(f"{cid_prefix()}[Image] 422/policy — silent soften retry. orig[:50]={requested_prompt[:50]!r}")
                            current_prompt = softened
                            continue

                        # If this was a "reimaginación de esencia" (Zelda if made by Miyazaki, etc.),
                        # use the smart concept-only rewriter instead of the generic ultra fallback.
                        if _is_reimagination_request(requested_prompt):
                            concept = _rewrite_to_concept_essence(requested_prompt)
                            if concept and concept != current_prompt:
                                logger.info(f"{cid_prefix()}[Image] 422/policy — IP concept rewrite (essence preserved). orig[:50]={requested_prompt[:50]!r}")
                                current_prompt = concept
                                continue

                        ultra = _ultra_safe_artistic_fallback(requested_prompt)
                        logger.info(f"{cid_prefix()}[Image] 422 — using ultra fallback. orig[:50]={requested_prompt[:50]!r}")
                        current_prompt = ultra
                        continue

                    if is_policy:
                        # Final policy block: prefer direct clean delivery
                        # For reimagination/essence requests we already tried hard to strip IPs,
                        # so give slightly more useful guidance in the internal return value.
                        if _is_reimagination_request(requested_prompt):
                            clean_user_msg = "No se pudo generar la imagen con esa descripción. Probá describir la atmósfera y el vibe sin nombrar personajes ni marcas famosas (tipo 'un reino antiguo melancólico con ruinas en un bosque brumoso, estilo soulslike')."
                        else:
                            clean_user_msg = "No se pudo generar la imagen."

                        if request_id:
                            try:
                                info = await consume_image_request(request_id)
                                if info and (orig_msg := info.get("original_message")):
                                    await orig_msg.reply(clean_user_msg, mention_author=False)
                                    # buffer for context
                                    try:
                                        ch = getattr(orig_msg, "channel", None)
                                        if ch and (ch_id := getattr(ch, "id", None)):
                                            from .. import context as ctx
                                            ctx.update_from_message(
                                                channel_id=ch_id, user_id=0, author_name="Groksito",
                                                content=clean_user_msg, is_bot=True
                                            )
                                    except Exception:
                                        pass
                                    logger.info(f"{cid_prefix()}[ImageDelivery] Direct clean policy message for generate {request_id}")
                                    from ..llm.prompt_builder import DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK
                                    return DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK
                            except Exception as dir_err:
                                logger.warning(f"{cid_prefix()}[ImageDelivery] Direct policy delivery failed: {dir_err}")

                        return (
                            "POLICY_BLOCKED. "
                            "Respond naturally, briefly and kindly that the image could not be generated. "
                            "If it looks like a reimagination request, gently suggest describing the mood/world without naming famous characters or companies. "
                            "Do NOT mention prompt, explícito, sugerente, or any internal details. Keep native Grok tone."
                        )

                    return f"Error generating image: {err_msg or response.text[:120]}"

                # Success
                data = response.json()
                urls = [img["url"] for img in data.get("data", []) if isinstance(img, dict) and "url" in img]
                if not urls:
                    return "Could not generate the images (empty API response)."

                urls_block = "\n".join(urls)
                logger.info(f"{cid_prefix()}[Image] generate success: request_id={request_id}, count={len(urls)}")
                try:
                    sent_prompt = (user_caption_source or requested_prompt or "")[:120]
                    logger.debug(f"{cid_prefix()}[Image] using prompt_for_caption[:120]={sent_prompt!r}")
                except Exception:
                    pass

                # Prefer original user phrasing for the caption (language consistency across servers).
                # The 'requested_prompt' may be model-adapted; we still pass it as fallback.
                caption = build_image_caption(requested_prompt, user_original=user_caption_source or requested_prompt)

                if request_id and await deliver_from_request(
                    request_id, caption=caption, urls=urls, kind="image"
                ):
                    from ..llm.prompt_builder import DIRECT_DELIVERY_SUCCESS_IMAGE
                    return DIRECT_DELIVERY_SUCCESS_IMAGE

                return f"Image URLs ready:\n{urls_block}"

    except Exception as e:
        logger.exception(f"{cid_prefix()}[Image] Unexpected error in generate")
        return f"Error generating image: {str(e)}"


# =============================================================================
# Core Editing (modernized)
# =============================================================================

def _validate_edit_references(reference_urls: list[str] | None) -> str | None:
    refs = (reference_urls or [])[:3]
    if not refs:
        return (
            "No reference images for edit_image. "
            "Ask the user to upload the photo(s) (or reply to a message containing the image) and describe the desired transformation."
        )
    return None


def _guess_image_mime(url: str, data: bytes) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png") or data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if path.endswith(".webp") or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        return "image/webp"
    if path.endswith(".gif") or data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def _needs_base64_for_edit_api(url: str) -> bool:
    """Discord CDN URLs are signed and often unreachable by xAI's edits fetcher."""
    host = (urlparse(url).hostname or "").lower()
    return any(
        marker in host
        for marker in ("discordapp.com", "discordapp.net", "discord.com")
    )


async def _resolve_edit_reference_url(url: str) -> str:
    """Return a public URL or base64 data URI that the xAI edits API can fetch."""
    if not url or not _needs_base64_for_edit_api(url):
        return url

    data = await _download_url(url)
    if not data:
        logger.warning(f"{cid_prefix()}[Image] Could not download Discord reference for edit; using raw URL")
        return url

    mime = _guess_image_mime(url, data)
    encoded = base64.b64encode(data).decode("ascii")
    logger.info(f"{cid_prefix()}[Image] Resolved Discord reference to base64 data URI ({len(data)} bytes)")
    return f"data:{mime};base64,{encoded}"


async def _resolve_edit_reference_urls(reference_urls: list[str]) -> list[str]:
    resolved: list[str] = []
    for url in reference_urls[:3]:
        resolved.append(await _resolve_edit_reference_url(url))
    return resolved


def _build_edit_payload(prompt: str, refs: list[str], aspect_ratio: str | None, **extra: Any) -> dict:
    image_entries = [{"url": url, "type": "image_url"} for url in refs]
    payload: dict = {
        "model": extra.get("model", "grok-imagine-image-quality"),
        "prompt": prompt,
        "response_format": "url",
    }
    if len(image_entries) == 1:
        payload["image"] = image_entries[0]
    else:
        payload["images"] = image_entries
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    for k in ("quality", "style"):
        if k in extra and extra[k] is not None:
            payload[k] = extra[k]
    return payload


def _extract_edit_urls(data: dict) -> list[str]:
    urls: list[str] = []
    if isinstance(data, dict):
        if "data" in data:
            urls = [i["url"] for i in data.get("data", []) if isinstance(i, dict) and "url" in i]
        elif "url" in data:
            urls = [data["url"]]
        elif "urls" in data:
            urls = data["urls"] if isinstance(data["urls"], list) else [data["urls"]]
        if not urls:
            for key in ("result", "output", "images"):
                val = data.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "url" in item:
                            urls.append(item["url"])
                        elif isinstance(item, str) and item.startswith("http"):
                            urls.append(item)
                elif isinstance(val, str) and val.startswith("http"):
                    urls.append(val)
                    break
    return urls


async def _try_direct_edit_delivery(request_id: str, urls: list[str], prompt: str | None = None) -> bool:
    if not request_id:
        return False
    return await deliver_from_request(
        request_id, caption=build_edit_caption(prompt=prompt), urls=urls, kind="image"
    )


async def _tool_edit_image(
    prompt: str,
    reference_urls: list[str] | None = None,
    aspect_ratio: str | None = None,
    request_id: Optional[str] = None,
    **extra_params: Any,
) -> str:
    """
    Modern centralized image editing.

    - Always enhances the edit instruction prompt for better results.
    - Strong reference validation.
    - Clean policy handling (returns POLICY_BLOCKED sentinel for consistency).
    - Direct delivery preferred.
    """
    api_key = _resolve_api_key()
    if not api_key:
        return "No xAI credential configured for image editing (run --login-oauth or set XAI_API_KEY)."

    validation_error = _validate_edit_references(reference_urls)
    if validation_error:
        return validation_error

    refs = (reference_urls or [])[:3]
    refs = await _resolve_edit_reference_urls(refs)
    # Enhance the user's edit instruction (modern quality pass)
    enhanced_prompt = _enhance_prompt_for_api(prompt, is_edit=True)

    try:
        max_attempts = getattr(settings, "api_max_retries", 3)
        payload = _build_edit_payload(enhanced_prompt, refs, aspect_ratio, **extra_params)

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=settings.api_timeout_seconds) as http_client:
                    response = await http_client.post(
                        "https://api.x.ai/v1/images/edits",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json=payload
                    )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as net_err:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.4 * (2 ** attempt))
                    continue
                raise

            if response.status_code != 200:
                try:
                    error = response.json().get("error", {})
                    err_msg = error.get("message", "unknown").lower()
                except Exception:
                    err_msg = (response.text or "")[:200].lower()

                if any(kw in err_msg for kw in ("policy", "safety", "content", "violation", "censored", "inappropriate", "blocked")):
                    return "POLICY_BLOCKED"
                return f"Error editing image: {error.get('message', 'unknown') if 'error' in locals() else err_msg}"

            data = response.json()
            break
        else:
            return "Error editing image: transient network failures after retries."

        urls = _extract_edit_urls(data)
        if not urls:
            return "Could not generate the edited images (empty response)."

        if await _try_direct_edit_delivery(request_id, urls, prompt=prompt):
            from ..llm.prompt_builder import DIRECT_DELIVERY_SUCCESS_EDIT
            return DIRECT_DELIVERY_SUCCESS_EDIT

        return (
            "EDIT_URLS_READY (direct Discord delivery failed). "
            "Include these edited image URLs in your final response:\n"
            + "\n".join(urls)
        )

    except Exception as e:
        logger.exception(f"{cid_prefix()}[Image] Unexpected error in edit")
        return f"Error editing/transforming image: {str(e)}"


# =============================================================================
# Public Dispatch Handlers (stable API for tools.py / execute_hybrid_tool)
# =============================================================================

async def _handle_generate_image(args: dict, original_message: Any) -> str:
    """
    Dispatcher for generate_image tool.
    Registers for direct delivery, calls the centralized modern generator,
    supports aspect_ratio and future params.
    """
    prompt = args.get("prompt", "")
    count = int(args.get("count", 1))
    aspect_ratio = args.get("aspect_ratio") or args.get("aspect") or None

    # Capture a human-facing source for the caption so it matches user's language
    # even if the model adapted the prompt arg for the generation API.
    user_caption_source = None
    if original_message:
        try:
            content = (getattr(original_message, "content", "") or "").strip()
            # Use the raw user text if it looks like a creation request (keeps Spanish tone etc.)
            if content and (len(content) < 280):
                user_caption_source = content
        except Exception:
            pass

    request_id = None
    if original_message:
        try:
            uid = getattr(getattr(original_message, "author", None), "id", 0)
            mid = getattr(original_message, "id", 0)
            request_id = await register_image_request(
                user_id=uid,
                channel_id=getattr(getattr(original_message, "channel", None), "id", 0) or 0,
                message_id=mid,
                operation_type="generate",
                original_message=original_message,
            )
        except Exception as reg_err:
            logger.warning(f"{cid_prefix()}[Image] Failed to register generate request: {reg_err}")

    # Note: we still pass the model's prompt to the generator (as before).
    # The caption will prefer user_caption_source inside build_image_caption when wired.
    result = await _tool_generate_image(
        prompt, count, request_id=request_id, aspect_ratio=aspect_ratio,
        user_caption_source=user_caption_source,
        # pass any extra future params from args if present
        **{k: v for k, v in args.items() if k not in ("prompt", "count", "aspect_ratio", "aspect")}
    )

    # If the inner generator did direct delivery using its own caption build,
    # the user_caption_source is best-effort for future improvements.
    # For now the main win is the build_image_caption signature + comment.
    return result


async def _handle_edit_image(args: dict, original_message: Any, image_urls: list[str] | None) -> str:
    """
    Dispatcher for edit_image tool.
    Uses vision-harvested or uploaded reference images.
    Centralized modern editing with prompt enhancement.
    """
    prompt = args.get("prompt", "")
    aspect = args.get("aspect_ratio") or args.get("aspect") or None

    ref_count = len(image_urls or [])
    if ref_count > 0:
        logger.info(f"{cid_prefix()}[Image] edit_image with {ref_count} reference(s)")
    else:
        logger.warning(f"{cid_prefix()}[Image] edit_image called with ZERO references")

    request_id = None
    if original_message:
        try:
            uid = getattr(getattr(original_message, "author", None), "id", 0)
            mid = getattr(original_message, "id", 0)
            request_id = await register_image_request(
                user_id=uid,
                channel_id=getattr(getattr(original_message, "channel", None), "id", 0) or 0,
                message_id=mid,
                operation_type="edit",
                original_message=original_message,
            )
        except Exception as reg_err:
            logger.warning(f"{cid_prefix()}[Image] Failed to register edit request: {reg_err}")

    return await _tool_edit_image(
        prompt,
        reference_urls=image_urls,
        aspect_ratio=aspect,
        request_id=request_id,
        **{k: v for k, v in args.items() if k not in ("prompt", "aspect_ratio", "aspect")}
    )


# Public handler symbols (imported by llm/media_tools.py)
__all__ = [
    "_handle_generate_image",
    "_handle_edit_image",
    "soften_image_prompt",
    "_enhance_prompt_for_api",
]

# Lazy module-level re-exports for DIRECT_* (used by test_guidance_centralization).
# Prevents top-level from ..llm.prompt_builder (would risk cycles on direct media imports).
def __getattr__(name: str):
    mapping = {
        "DIRECT_DELIVERY_SUCCESS_IMAGE": "DIRECT_DELIVERY_SUCCESS_IMAGE",
        "DIRECT_DELIVERY_SUCCESS_EDIT": "DIRECT_DELIVERY_SUCCESS_EDIT",
        "DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK": "DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK",
    }
    if name in mapping:
        from ..llm.prompt_builder import (
            DIRECT_DELIVERY_SUCCESS_IMAGE,
            DIRECT_DELIVERY_SUCCESS_EDIT,
            DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK,
        )
        return {
            "DIRECT_DELIVERY_SUCCESS_IMAGE": DIRECT_DELIVERY_SUCCESS_IMAGE,
            "DIRECT_DELIVERY_SUCCESS_EDIT": DIRECT_DELIVERY_SUCCESS_EDIT,
            "DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK": DIRECT_DELIVERY_SUCCESS_POLICY_BLOCK,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
