"""Join the caller's Discord voice channel and play Meepo TTS there.

Not a music bot. One guild connection at a time.
Requires: PyNaCl, ffmpeg (already used by pydub), Connect + Speak permissions,
and intents.voice_states = True.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from typing import Any

import discord

logger = logging.getLogger("groksito.media.voice_channel")

OWNER_ID = 253869773421674498

_vc: discord.VoiceClient | None = None
_play_lock = asyncio.Lock()

_JOIN_RE = re.compile(
    r"\b(join|enter|come to|hop in)\b.{0,24}\b(vc|voice|call|channel)\b"
    r"|\b(vc|voice)\b.{0,12}\b(join|enter)\b",
    re.I,
)
_LEAVE_RE = re.compile(
    r"\b(leave|disconnect|gtfo|get out)\b.{0,24}\b(vc|voice|call|channel)\b"
    r"|\b(leave|disconnect)\b\s+(the\s+)?(vc|voice)\b",
    re.I,
)
_SPEAK_RE = re.compile(
    r"^(?:meepo|groksito)?[\s,:-]*(?:say|speak|talk|announce|tell them)\s+"
    r"(?:(?:in|on)\s+(?:the\s+)?(?:vc|voice|call)\s*[:,-]?\s*)?(?P<text>.+)$",
    re.I,
)


def _clean_activation(text: str) -> str:
    t = text or ""
    t = re.sub(r"<@!?\d+>", " ", t)
    t = re.sub(r"\b(@?meepo|@?groksito)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def is_voice_control(text: str) -> bool:
    raw = _clean_activation(text)
    if not raw:
        return False
    return bool(_JOIN_RE.search(raw) or _LEAVE_RE.search(raw) or _SPEAK_RE.search(raw))


def caller_channel(member: Any) -> discord.VoiceChannel | None:
    vs = getattr(member, "voice", None)
    ch = getattr(vs, "channel", None)
    if isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
        return ch
    return None


async def leave() -> str:
    global _vc
    if _vc and _vc.is_connected():
        await _vc.disconnect(force=True)
    _vc = None
    return "Left the voice channel."


async def join(member: Any) -> str:
    global _vc
    channel = caller_channel(member)
    if channel is None:
        return "Join a voice channel first, then ask me to join."

    perms = channel.permissions_for(channel.guild.me)
    if not perms.connect or not perms.speak:
        return "I need Connect + Speak in that voice channel."

    if _vc and _vc.is_connected():
        if _vc.channel and _vc.channel.id == channel.id:
            return f"Already in {channel.name}."
        await _vc.move_to(channel)
        return f"Moved to {channel.name}."

    _vc = await channel.connect(self_deaf=True, reconnect=True)
    return f"Joined {channel.name}. Say `@Meepo say in vc hello` and I will talk there."


async def _play_bytes(audio_bytes: bytes, filename: str = "meepo.mp3") -> str:
    global _vc
    if not _vc or not _vc.is_connected():
        return "I am not in a voice channel. Join one and say `@Meepo join vc` first."

    suffix = ".mp3"
    if filename.endswith(".ogg"):
        suffix = ".ogg"
    fd, path = tempfile.mkstemp(prefix="meepo_vc_", suffix=suffix)
    os.close(fd)
    try:
        with open(path, "wb") as fh:
            fh.write(audio_bytes)

        async with _play_lock:
            if _vc.is_playing():
                _vc.stop()
            done = asyncio.Event()

            def _after(err: Exception | None) -> None:
                if err:
                    logger.warning("vc play error: %s", err)
                done.set()

            source = discord.FFmpegPCMAudio(path)
            _vc.play(source, after=_after)
            try:
                await asyncio.wait_for(done.wait(), timeout=120)
            except asyncio.TimeoutError:
                if _vc.is_playing():
                    _vc.stop()
        return "Spoken in voice."
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def speak_text(text: str, voice: str = "eve", language: str = "en") -> str:
    from .audio_handler import _prepare_text_for_tts, _resolve_api_key
    import httpx

    api_key = _resolve_api_key()
    if not api_key:
        return "No xAI credential for TTS."
    prepared = _prepare_text_for_tts(text)
    if not prepared:
        return "Nothing to say."

    payload = {
        "text": prepared,
        "voice_id": voice,
        "language": language,
        "output_format": {"codec": "mp3", "sample_rate": 24000, "bit_rate": 128000},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post("https://api.x.ai/v1/tts", headers=headers, json=payload)
    if resp.status_code >= 400:
        return f"TTS failed ({resp.status_code})."
    if len(resp.content) < 100:
        return "TTS returned empty audio."
    return await _play_bytes(resp.content)


async def handle_voice_command(message: discord.Message) -> bool:
    """Return True if this message was a VC command and was handled."""
    raw = _clean_activation(message.content or "")
    if not raw:
        return False

    if _LEAVE_RE.search(raw):
        await message.channel.send(await leave())
        return True

    if _JOIN_RE.search(raw):
        await message.channel.send(await join(message.author))
        return True

    m = _SPEAK_RE.search(raw)
    if m:
        line = (m.group("text") or "").strip()
        line = re.sub(r"^(?:in|on)\s+(?:the\s+)?(?:vc|voice|call)\s*[:,-]?\s*", "", line, flags=re.I)
        if not line:
            await message.channel.send("Say what I should speak. Example: `@Meepo say in vc raid in 5`.")
            return True
        if not (_vc and _vc.is_connected()):
            joined = await join(message.author)
            if caller_channel(message.author) is None:
                await message.channel.send(joined)
                return True
        lang = "ne" if re.search(r"[\u0900-\u097F]", line) else "en"
        result = await speak_text(line, language=lang)
        await message.channel.send(result)
        return True

    return False
