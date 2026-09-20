"""Post one Grok-written motivational quote image daily at 09:00 Asia/Kathmandu.

Runs inside the already-running Discord process. No daily restart.

Env:
  MOTIVATION_CHANNEL_ID   required Discord channel id
  MOTIVATION_TZ           default Asia/Kathmandu
  MOTIVATION_HOUR         default 9
  MOTIVATION_MINUTE       default 0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from ..utils.correlation import cid_prefix

logger = logging.getLogger("groksito.media.daily_motivation")


def _tz():
    return ZoneInfo(os.getenv("MOTIVATION_TZ", "Asia/Kathmandu"))


def _data_path() -> Path:
    raw = os.getenv("GROKSITO_DATA_DIR") or os.getenv("DATA_DIR") or "/app/data"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p / "motivation_state.json"


def _already_sent_today(day: str) -> bool:
    path = _data_path()
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("last_day") == day
    except Exception:
        return False


def _mark_sent(day: str, quote: str) -> None:
    _data_path().write_text(
        json.dumps({"last_day": day, "quote": quote}, indent=2),
        encoding="utf-8",
    )


def _seconds_until_next_slot() -> float:
    now = datetime.now(_tz())
    hour = int(os.getenv("MOTIVATION_HOUR", "9"))
    minute = int(os.getenv("MOTIVATION_MINUTE", "0"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now >= target:
        target = target + timedelta(days=1)
    return max(5.0, (target - now).total_seconds())


def _resolve_key() -> str | None:
    try:
        from ..core.grok_oauth import get_grok_bearer
        tok = get_grok_bearer()
        if tok:
            return tok
    except Exception:
        pass
    return os.getenv("XAI_API_KEY")


async def _grok_quote(day: str) -> str:
    key = _resolve_key()
    fallback = "Show up. The mountain is still there."
    if not key:
        return fallback
    system = (
        "You write one original short motivational line for a Where Winds Meet / "
        "wuxia Discord guild named Everest. One or two sentences. No hashtags, "
        "no quotes around the line, no author name, no NSFW, no mentioning Grok "
        "or models. Can lightly use martial / wind / mountain imagery. English."
    )
    user = f"Date {day} (Nepal). New line. Do not repeat a generic poster slogan."
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("XAI_TEXT_MODEL", "grok-4.3"),
                    "temperature": 0.95,
                    "max_tokens": 80,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        if resp.status_code != 200:
            logger.warning("motivation quote HTTP %s: %s", resp.status_code, resp.text[:180])
            return fallback
        text = (
            ((resp.json().get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        ).strip().strip('"').strip("'")
        return text[:220] if text else fallback
    except Exception as exc:
        logger.warning("motivation quote failed: %s", exc)
        return fallback


async def _generate_quote_image(quote: str) -> bytes | None:
    key = _resolve_key()
    if not key:
        return None
    prompt = (
        "Vertical 9:16 cinematic wuxia key art, moonlit peaks and wind-blown pines, "
        "ink gold and deep blue, generous empty sky. Put this exact quote on the "
        f"poster in elegant readable English lettering: {quote} "
        "No logos, no extra titles, SFW, fully clothed if any figures."
    )
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.getenv("XAI_IMAGE_MODEL", "grok-imagine-image-2.0"),
                    "prompt": prompt,
                    "n": 1,
                    "response_format": "url",
                    "aspect_ratio": "9:16",
                },
            )
            if resp.status_code != 200:
                logger.warning("motivation image HTTP %s: %s", resp.status_code, resp.text[:200])
                return None
            url = ((resp.json().get("data") or [{}])[0] or {}).get("url")
            if not url:
                return None
            img = await client.get(url)
            if img.status_code != 200:
                return None
            return img.content
    except Exception as exc:
        logger.warning("motivation image failed: %s", exc)
        return None


async def _post_today(client) -> None:
    channel_id = os.getenv("MOTIVATION_CHANNEL_ID", "").strip()
    if not channel_id:
        logger.warning("daily motivation: MOTIVATION_CHANNEL_ID not set")
        return
    now = datetime.now(_tz())
    day = now.strftime("%Y-%m-%d")
    if _already_sent_today(day):
        logger.info("%s[Motivation] already sent %s", cid_prefix(), day)
        return
    quote = await _grok_quote(day)
    raw = await _generate_quote_image(quote)
    channel = client.get_channel(int(channel_id))
    if channel is None:
        try:
            channel = await client.fetch_channel(int(channel_id))
        except Exception as exc:
            logger.warning("daily motivation: bad channel %s (%s)", channel_id, exc)
            return
    if raw:
        import discord
        await channel.send(
            content="Good Morning Everest",
            file=discord.File(BytesIO(raw), filename="meepo_quote.png"),
        )
    else:
        await channel.send("Good Morning Everest")
    _mark_sent(day, quote)
    logger.info("%s[Motivation] posted %s", cid_prefix(), day)


async def run_loop(client) -> None:
    await asyncio.sleep(8)
    if not os.getenv("MOTIVATION_CHANNEL_ID", "").strip():
        logger.info("[Motivation] skipped — set MOTIVATION_CHANNEL_ID")
        return
    logger.info("[Motivation] daily 09:00 Asia/Kathmandu loop started (no daily restart)")
    while True:
        try:
            now = datetime.now(_tz())
            hour = int(os.getenv("MOTIVATION_HOUR", "9"))
            if now.hour == hour and now.minute < 20:
                await _post_today(client)
            wait = _seconds_until_next_slot()
            logger.info("[Motivation] next run in %.0fs (NPT)", wait)
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[Motivation] tick failed")
            await asyncio.sleep(60)
