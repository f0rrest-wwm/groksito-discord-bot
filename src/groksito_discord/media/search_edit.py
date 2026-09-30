"""Search the web for a real photo, edit it with Imagine, deliver like edit_image."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from ..config import settings
from .image_handler import _handle_edit_image, _resolve_api_key, cid_prefix

logger = logging.getLogger("groksito.bot")

_IMG_EXT = re.compile(r"\.(?:jpe?g|png|webp|gif)(?:\?|$)", re.I)
_URL_RE = re.compile(r"https?://[^\s\]\)>'\"<>]+", re.I)


def _extract_urls(text: str) -> list[str]:
    seen: list[str] = []
    for raw in _URL_RE.findall(text or ""):
        url = raw.rstrip(").,;]")
        if url not in seen:
            seen.append(url)
    prefer = [u for u in seen if _IMG_EXT.search(u)]
    return (prefer or seen)[:3]


async def _search_web_image_urls(query: str) -> list[str]:
    key = _resolve_api_key()
    if not key:
        return []
    model = os.getenv("XAI_TEXT_MODEL") or getattr(settings, "model", None) or "grok-4.3"
    payload = {
        "model": model,
        "input": (
            f"Find public photos of: {query}\n"
            "Return only direct https image URLs, one per line. "
            "Prefer official game screenshots or key art. No commentary."
        ),
        "tools": [{"type": "web_search", "enable_image_search": True}],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/responses",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if resp.status_code != 200:
            logger.warning(
                f"{cid_prefix()}[Image] search_edit HTTP {resp.status_code}: {(resp.text or '')[:200]}"
            )
            return []
        data = resp.json()
    except Exception:
        logger.exception(f"{cid_prefix()}[Image] search_edit search failed")
        return []

    chunks: list[str] = []
    if isinstance(data.get("output_text"), str):
        chunks.append(data["output_text"])
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    chunks.append(str(part.get("text") or ""))
        text = item.get("text")
        if isinstance(text, str):
            chunks.append(text)
    blob = "\n".join(chunks)
    urls = _extract_urls(blob)
    logger.info(f"{cid_prefix()}[Image] search_edit query={query[:80]!r} urls={len(urls)}")
    return urls


async def _handle_search_edit_image(args: dict, original_message: Any) -> str:
    query = (args.get("search_query") or args.get("query") or "").strip()
    edit = (args.get("edit_prompt") or args.get("prompt") or "").strip()
    if not query:
        return "Need a search_query (who or what to find)."
    if not edit:
        edit = "Full-frame cinematic still of this character. Remove UI and watermarks. Keep identity. SFW."

    low = f"{query} {edit}".lower()
    if any(w in low for w in ("nude", "nsfw", "porn", "undress", "topless", "sex")):
        return "No NSFW allowed."

    urls = await _search_web_image_urls(query)
    if not urls:
        return "No usable image URL from web search. Try a more specific name."

    return await _handle_edit_image(
        {"prompt": edit, "aspect_ratio": args.get("aspect_ratio") or "9:16"},
        original_message,
        urls[:1],
    )
