"""Per-user daily video quota + one-at-a-time lock + NSFW gate."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("groksito.media.video_quota")

_LOCK = threading.Lock()
_IN_FLIGHT = False
_IN_FLIGHT_USER = None

DAILY_LIMIT = int(os.getenv("VIDEO_DAILY_LIMIT", "5"))
NSFW_REPLY = "No NSFW allowed."
NSFW_MARKERS = (
    "nsfw", "nude", "nudes", "naked", "undress", "remove clothes", "take off",
    "topless", "bottomless", "porn", "porno", "hentai", "xxx", "onlyfans",
    "blowjob", "handjob", "cumshot", "sex tape", "having sex", "make her naked",
    "strip tease", "striptease", "explicit sex",
)


def _data_dir() -> Path:
    raw = os.getenv("GROKSITO_DATA_DIR") or os.getenv("DATA_DIR") or "/app/data"
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path() -> Path:
    return _data_dir() / "video_usage.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _unlimited_ids() -> set[str]:
    blob = ",".join([
        os.getenv("VIDEO_UNLIMITED_USER_IDS", ""),
        os.getenv("GROKSITO_OWNER_ID", "253869773421674498"),
    ])
    return {x.strip() for x in blob.split(",") if x.strip()}


def _load() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"day": _today(), "users": {}, "in_flight": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"day": _today(), "users": {}, "in_flight": False}
        if data.get("day") != _today():
            data = {"day": _today(), "users": {}, "in_flight": False}
        data.setdefault("users", {})
        return data
    except Exception as exc:
        logger.warning("video_quota load failed: %s", exc)
        return {"day": _today(), "users": {}, "in_flight": False}


def _save(data: dict[str, Any]) -> None:
    path = _path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_unlimited(user_id) -> bool:
    return str(user_id) in _unlimited_ids()


def is_nsfw_request(text: str) -> bool:
    blob = (text or "").lower()
    return any(m in blob for m in NSFW_MARKERS)


def used_today(user_id) -> int:
    uid = str(user_id)
    with _LOCK:
        data = _load()
        row = data.get("users", {}).get(uid) or {}
        return int(row.get("count") or 0)


def check_can_start(user_id, display_name: str = ""):
    global _IN_FLIGHT, _IN_FLIGHT_USER
    uid = str(user_id)
    with _LOCK:
        if _IN_FLIGHT:
            return (
                "Please wait until the previous video generation is complete "
                "before requesting another one."
            )
        if not is_unlimited(uid):
            data = _load()
            row = data.get("users", {}).get(uid) or {}
            count = int(row.get("count") or 0)
            if count >= DAILY_LIMIT:
                return (
                    f"Daily video limit reached ({DAILY_LIMIT}/day). "
                    "Try again after UTC midnight."
                )
        _IN_FLIGHT = True
        _IN_FLIGHT_USER = uid
        return None


def mark_success(user_id, display_name: str = "") -> int:
    uid = str(user_id)
    with _LOCK:
        data = _load()
        users = data.setdefault("users", {})
        row = users.get(uid) or {"count": 0, "name": display_name}
        if not is_unlimited(uid):
            row["count"] = int(row.get("count") or 0) + 1
        if display_name:
            row["name"] = display_name
        users[uid] = row
        _save(data)
        return int(row.get("count") or 0)


def release_lock() -> None:
    global _IN_FLIGHT, _IN_FLIGHT_USER
    with _LOCK:
        _IN_FLIGHT = False
        _IN_FLIGHT_USER = None


def snapshot():
    with _LOCK:
        data = _load()
        data["in_flight_memory"] = _IN_FLIGHT
        data["in_flight_user"] = _IN_FLIGHT_USER
        data["unlimited"] = sorted(_unlimited_ids())
        data["daily_limit"] = DAILY_LIMIT
        return data


def status_message(user_id, display_name: str = "") -> str:
    uid = str(user_id)
    used = used_today(uid)
    limit = DAILY_LIMIT
    if is_unlimited(uid):
        return (
            f"Video quota for {display_name or uid}: unlimited "
            f"(creator). Reset is UTC midnight. Limit for others: {limit}/day."
        )
    left = max(0, limit - used)
    return (
        f"Video quota for {display_name or uid}: {used}/{limit} used today, "
        f"{left} left. Resets at UTC midnight."
    )
