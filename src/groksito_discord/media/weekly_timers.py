"""Post the Aion 2 weekly reset checklist every Wednesday at 07:00 UTC.
JP server (UTC+9). Also posts once on first start if the channel has never received one.

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


def _next_from_hours(hours: tuple[int, ...], now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    candidates = []
    for day_offset in (0, 1):
        day = (now + timedelta(days=day_offset)).replace(minute=0, second=0, microsecond=0)
        for h in hours:
            t = day.replace(hour=h)
            if t > now:
                candidates.append(t)
    return min(candidates) if candidates else now + timedelta(hours=3)


def _message() -> str:
    now = datetime.now(UTC)
    reset = _next_reset(now)
    # JP is UTC+9. Server hours below are converted to UTC.
    siege_mon = _next_weekday_at(0, 12, 0, now)   # Mon 21:00 JST
    siege_thu = _next_weekday_at(3, 12, 0, now)   # Thu 21:00 JST
    siege_sat = _next_weekday_at(5, 12, 0, now)   # Sat 21:00 JST
    exec_mon = _next_weekday_at(0, 12, 30, now)   # Mon 21:30 JST
    exec_thu = _next_weekday_at(3, 12, 30, now)
    exec_sat = _next_weekday_at(5, 12, 30, now)
    nahma_sun = _next_weekday_at(6, 12, 0, now)   # Sun 21:00 JST
    nahma_fri = _next_weekday_at(4, 12, 0, now)   # Fri 21:00 JST
    # Rifts: 02/05/08/11/14/17/20/23 JST -> 17/20/23/02/05/08/11/14 UTC
    rift = _next_from_hours((17, 20, 23, 2, 5, 8, 11, 14), now)
    # Watcher Kaira: 01/04/07/10/13/16/19/22 JST -> 16/19/22/01/04/07/10/13 UTC
    kaira = _next_from_hours((16, 19, 22, 1, 4, 7, 10, 13), now)
    # Arena of Tactics: 11:00 and 19:00 JST -> 02:00 and 10:00 UTC
    arena = _next_from_hours((2, 10), now)
    return (
        "**Aion 2 — JP server timers**\n"
        f"Next weekly reset: {_ts(reset, 'F')} ({_ts(reset, 'R')})\n"
        "Timestamps are your local time. Server clock is JST.\n\n"
        "**Entries that refill at the reset**\n"
        "• Daily Dungeon — 14\n"
        "• Ascension Trial — 3\n"
        "• Expeditions / Conquest / Transcendence — weekly cube limit\n"
        "• Abyss — 7 hours (membership higher)\n"
        "• Arena — victory / participation rewards\n"
        "• Battlefield — 3 victory + 3 participation\n"
        "• Shop limits — Odyle Energy, Spiritstones, Instant Clear\n\n"
        "**World bosses (Chaotic Lower Reshanta)**\n"
        f"• Watcher Kaira — {_ts(kaira, 'F')} ({_ts(kaira, 'R')})\n"
        f"• Executor Tamasa / Kaira / Argo — {_ts(exec_thu, 'F')} ({_ts(exec_thu, 'R')})\n"
        f"• Executor Tamasa / Kaira / Argo — {_ts(exec_sat, 'F')} ({_ts(exec_sat, 'R')})\n"
        f"• Executor Tamasa / Kaira / Argo — {_ts(exec_mon, 'F')} ({_ts(exec_mon, 'R')})\n"
        f"• Guardian Lord Nahma — {_ts(nahma_fri, 'F')} ({_ts(nahma_fri, 'R')})\n"
        f"• Guardian Lord Nahma — {_ts(nahma_sun, 'F')} ({_ts(nahma_sun, 'R')})\n\n"
        "**Other timers**\n"
        f"• Artifact Siege — {_ts(siege_thu, 'F')} ({_ts(siege_thu, 'R')})\n"
        f"• Artifact Siege — {_ts(siege_sat, 'F')} ({_ts(siege_sat, 'R')})\n"
        f"• Artifact Siege — {_ts(siege_mon, 'F')} ({_ts(siege_mon, 'R')})\n"
        f"• Next Spacetime Rift — {_ts(rift, 'F')} ({_ts(rift, 'R')})\n"
        f"• Next Arena of Tactics — {_ts(arena, 'F')} ({_ts(arena, 'R')})\n\n"
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
