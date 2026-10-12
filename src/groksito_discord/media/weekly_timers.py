"""Post the Aion 2 weekly reset checklist every Wednesday at 07:00 UTC.
Also posts once on first start if the channel has never received one.

Env:
  WEEKLY_TIMERS_CHANNEL_ID   required Discord channel id
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("groksito.media.weekly_timers")

UTC = timezone.utc
RESET_WEEKDAY = 2  # Wednesday
RESET_HOUR = 7


def _data_path() -> Path:
    raw = os.getenv("GROKSITO_DATA_DIR") or os.getenv("DATA_DIR") or "/app/data"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p / "weekly_timers_state.json"


def _load_state() -> dict:
    path = _data_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _already_sent(week: str) -> bool:
    return _load_state().get("last_week") == week


def _has_ever_posted() -> bool:
    return bool(_load_state().get("last_week"))


def _mark_sent(week: str) -> None:
    _data_path().write_text(
        json.dumps({"last_week": week}, indent=2),
        encoding="utf-8",
    )


def _next_reset(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    days_ahead = (RESET_WEEKDAY - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=RESET_HOUR, minute=0, second=0, microsecond=0
    )
    if target <= now:
        target += timedelta(days=7)
    return target


def _seconds_until_next() -> float:
    return max(5.0, (_next_reset() - datetime.now(UTC)).total_seconds())


def _ts(dt: datetime, style: str = "F") -> str:
    return f"<t:{int(dt.timestamp())}:{style}>"


def _next_weekday_at(weekday: int, hour: int, minute: int = 0, now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    days_ahead = (weekday - now.weekday()) % 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    if target <= now:
        target += timedelta(days=7)
    return target


def _next_rift(now: datetime | None = None) -> datetime:
    """Spacetime Rift portals every 3 hours at 02, 05, 08, 11, 14, 17, 20, 23 UTC+9.
    Global server clock is UTC+9, so those hours are 17, 20, 23, 02, 05, 08, 11, 14 UTC.
    """
    now = now or datetime.now(UTC)
    # Rift hours in UTC (server is UTC+9, rifts at 02/05/08/11/14/17/20/23 server)
    rift_hours_utc = (17, 20, 23, 2, 5, 8, 11, 14)
    candidates = []
    for day_offset in (0, 1):
        day = (now + timedelta(days=day_offset)).replace(minute=0, second=0, microsecond=0)
        for h in rift_hours_utc:
            t = day.replace(hour=h)
            if t > now:
                candidates.append(t)
    return min(candidates) if candidates else now + timedelta(hours=3)


def _message() -> str:
    now = datetime.now(UTC)
    reset = _next_reset(now)
    siege_mon = _next_weekday_at(0, 12, 0, now)  # Mon 21:00 UTC+9 = 12:00 UTC
    siege_thu = _next_weekday_at(3, 12, 0, now)
    siege_sat = _next_weekday_at(5, 12, 0, now)
    rift = _next_rift(now)
    return (
        "**Aion 2 — Weekly reset**\n"
        f"Next reset: {_ts(reset, 'F')} ({_ts(reset, 'R')})\n"
        "Global. The timestamp is your local time.\n\n"
        "**Entries that refill at that reset**\n"
        "• Daily Dungeon — 14\n"
        "• Ascension Trial — 3\n"
        "• Expeditions / Conquest / Transcendence — weekly cube limit\n"
        "• Abyss — 7 hours (membership higher)\n"
        "• Arena — victory / participation rewards\n"
        "• Battlefield — 3 victory + 3 participation\n"
        "• Shop limits — Odyle Energy, Spiritstones, Instant Clear\n\n"
        "**Still daily (not this post)**\n"
        "• Duty quests — 5\n"
        "• Nightmare — 2 a day, stores to 14\n"
        "• Shugo Festa keys — 3\n"
        "• Odyle Energy — +15 every 3 hours\n\n"
        "**Next timers**\n"
        f"• Artifact Siege — {_ts(siege_mon, 'F')} ({_ts(siege_mon, 'R')}) · Mon\n"
        f"• Artifact Siege — {_ts(siege_thu, 'F')} ({_ts(siege_thu, 'R')}) · Thu\n"
        f"• Artifact Siege — {_ts(siege_sat, 'F')} ({_ts(siege_sat, 'R')}) · Sat\n"
        f"• Next Spacetime Rift — {_ts(rift, 'F')} ({_ts(rift, 'R')})\n\n"
        "Burn leftover weekly entries before the reset. They do not stack."
    )


async def _post(client, week: str) -> None:
    channel_id = os.getenv("WEEKLY_TIMERS_CHANNEL_ID", "").strip()
    if not channel_id:
        logger.warning("weekly timers: WEEKLY_TIMERS_CHANNEL_ID not set")
        return
    channel = client.get_channel(int(channel_id))
    if channel is None:
        try:
            channel = await client.fetch_channel(int(channel_id))
        except Exception as exc:
            logger.warning("weekly timers: bad channel %s (%s)", channel_id, exc)
            return
    await channel.send(_message())
    _mark_sent(week)
    logger.info("[WeeklyTimers] posted %s", week)


async def run_loop(client) -> None:
    await asyncio.sleep(12)
    if not os.getenv("WEEKLY_TIMERS_CHANNEL_ID", "").strip():
        logger.info("[WeeklyTimers] skipped — set WEEKLY_TIMERS_CHANNEL_ID")
        return
    logger.info("[WeeklyTimers] Wednesday 07:00 UTC loop started")

    # First run: post immediately if the channel has never had one.
    if not _has_ever_posted():
        try:
            week = datetime.now(UTC).strftime("%G-W%V")
            await _post(client, week)
        except Exception:
            logger.exception("[WeeklyTimers] initial post failed")

    while True:
        try:
            now = datetime.now(UTC)
            if now.weekday() == RESET_WEEKDAY and now.hour == RESET_HOUR and now.minute < 20:
                week = now.strftime("%G-W%V")
                if not _already_sent(week):
                    await _post(client, week)
            wait = _seconds_until_next()
            logger.info("[WeeklyTimers] next run in %.0fs", wait)
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[WeeklyTimers] tick failed")
            await asyncio.sleep(60)
