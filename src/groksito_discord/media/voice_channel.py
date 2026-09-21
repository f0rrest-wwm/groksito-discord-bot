"""Voice channel: join, speak TTS, optional wake-word listen.

Requires: PyNaCl, davey, ffmpeg, discord-ext-voice-recv,
Connect + Speak, intents.voice_states = True.
"""
from __future__ import annotations

import asyncio
import audioop
import io
import logging
import os
import re
import tempfile
import wave
from collections import defaultdict
from typing import Any

import discord
import httpx

logger = logging.getLogger("groksito.media.voice_channel")

OWNER_ID = 253869773421674498

_vc: Any = None
_play_lock = asyncio.Lock()
_listen_on = False
_busy = False
_loop: asyncio.AbstractEventLoop | None = None
_buffers: dict[int, bytearray] = defaultdict(bytearray)
_last_pkt: dict[int, float] = {}

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
_LISTEN_ON_RE = re.compile(r"\b(listen|start listen|wake word|hear me)\b", re.I)
_LISTEN_OFF_RE = re.compile(r"\b(stop listen|dont listen|don't listen|deaf)\b", re.I)
_SPEAK_RE = re.compile(
    r"^(?:meepo|groksito)?[\s,:-]*(?:say|speak|talk|announce|tell them)\s+"
    r"(?:(?:in|on)\s+(?:the\s+)?(?:vc|voice|call)\s*[:,-]?\s*)?(?P<text>.+)$",
    re.I,
)
_WAKE_RE = re.compile(
    r"\b(hey\s+)?(meepo|mipo|meep0|mipoji|mipu)\b",
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
    return bool(
        _JOIN_RE.search(raw)
        or _LEAVE_RE.search(raw)
        or _SPEAK_RE.search(raw)
        or _LISTEN_ON_RE.search(raw)
        or _LISTEN_OFF_RE.search(raw)
    )


def caller_channel(member: Any) -> discord.VoiceChannel | None:
    vs = getattr(member, "voice", None)
    ch = getattr(vs, "channel", None)
    if isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
        return ch
    return None


def _recv_cls():
    try:
        from discord.ext import voice_recv
        return voice_recv.VoiceRecvClient
    except Exception as exc:
        logger.warning("voice_recv missing: %s", exc)
        return None


async def leave() -> str:
    global _vc, _listen_on
    _listen_on = False
    if _vc and getattr(_vc, "is_listening", lambda: False)():
        try:
            _vc.stop_listening()
        except Exception:
            pass
    if _vc and _vc.is_connected():
        await _vc.disconnect(force=True)
    _vc = None
    return "Left the voice channel."


async def join(member: Any) -> str:
    global _vc, _loop
    channel = caller_channel(member)
    if channel is None:
        return "Join a voice channel first, then ask me to join."
    _loop = asyncio.get_running_loop()

    if _vc and _vc.is_connected():
        if _vc.channel and _vc.channel.id == channel.id:
            return f"Already in {channel.name}."
        try:
            await _vc.move_to(channel)
            return f"Moved to {channel.name}."
        except discord.Forbidden:
            return f"Discord refused move into #{channel.name}."

    try:
        # Speak-only client. VoiceRecvClient starts a packet router that
        # crashes on DAVE-encrypted incoming audio (OpusError: corrupted stream).
        _vc = await channel.connect(self_deaf=True, reconnect=True)
    except discord.ClientException as exc:
        return f"Could not join: {exc}"
    except discord.Forbidden:
        return (
            f"Discord refused Connect/Speak in #{channel.name}. "
            "Fix channel overwrites for Meepo, then reinvite."
        )
    except Exception as exc:
        logger.exception("vc connect failed")
        return f"Join failed: {type(exc).__name__}: {exc}"
    extra = " Say `@Meepo listen` if you want wake-word replies."
    return f"Joined {channel.name}.{extra}"


async def _play_bytes(audio_bytes: bytes, filename: str = "meepo.mp3") -> str:
    global _vc
    if not _vc or not _vc.is_connected():
        return "I am not in a voice channel. `@Meepo join vc` first."

    suffix = ".ogg" if filename.endswith(".ogg") else ".mp3"
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

            _vc.play(discord.FFmpegPCMAudio(path), after=_after)
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


def _resolve_api_key() -> str | None:
    try:
        from ..core.grok_oauth import get_grok_bearer
        tok = get_grok_bearer()
        if tok:
            return tok
    except Exception:
        pass
    return os.getenv("XAI_API_KEY")


async def speak_text(text: str, voice: str = "eve", language: str = "en") -> str:
    from .audio_handler import _prepare_text_for_tts

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


def _pcm_to_wav(pcm: bytes, rate: int = 48000, width: int = 2, channels: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def _stt(pcm: bytes) -> str:
    api_key = _resolve_api_key()
    if not api_key or len(pcm) < 48000:
        return ""
    wav = _pcm_to_wav(pcm)
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("clip.wav", wav, "audio/wav")}
    data = {
        "model": "grok-voice-transcribe-2.0",
        "language": "en",
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post("https://api.x.ai/v1/stt", headers=headers, files=files, data=data)
    if resp.status_code >= 400:
        logger.warning("stt failed %s %s", resp.status_code, resp.text[:200])
        return ""
    try:
        body = resp.json()
    except Exception:
        return ""
    return str(body.get("text") or "").strip()


async def _reply_from_speech(user: Any, spoken: str) -> None:
    global _busy
    if _busy:
        return
    _busy = True
    try:
        rest = _WAKE_RE.sub(" ", spoken, count=1)
        rest = re.sub(r"\s+", " ", rest).strip(" ,.-")
        if not rest:
            await speak_text("Yeah. What.")
            return
        api_key = _resolve_api_key()
        answer = rest
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": os.getenv("GROK_MODEL", "grok-4.3"),
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are Meepo in a Discord voice call for Everest guild. "
                                        "One or two short spoken sentences. No markdown. "
                                        "Never name Grok or xAI. Never roast f0rest."
                                    ),
                                },
                                {"role": "user", "content": f"{getattr(user, 'display_name', 'someone')} said: {rest}"},
                            ],
                            "max_tokens": 120,
                        },
                    )
                if r.status_code < 400:
                    answer = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("listen llm failed: %s", exc)
        lang = "ne" if re.search(r"[\u0900-\u097F]", rest + answer) else "en"
        await speak_text(answer, language=lang)
    finally:
        _busy = False


def _flush_user(uid: int, user: Any) -> None:
    pcm = bytes(_buffers.pop(uid, b""))
    _last_pkt.pop(uid, None)
    if not _loop or len(pcm) < 48000 * 2:
        return

    async def _run() -> None:
        text = await _stt(pcm)
        if not text:
            return
        logger.info("vc stt uid=%s text=%r", uid, text[:120])
        if not _WAKE_RE.search(text):
            return
        await _reply_from_speech(user, text)

    asyncio.run_coroutine_threadsafe(_run(), _loop)


def _start_sink() -> Any:
    from discord.ext import voice_recv

    class WakeSink(voice_recv.AudioSink):
        def wants_opus(self) -> bool:
            return False

        def write(self, user, data) -> None:
            if not _listen_on or user is None or _busy:
                return
            if getattr(user, "bot", False):
                return
            pcm = getattr(data, "pcm", None) or b""
            if not pcm:
                return
            uid = int(getattr(user, "id", 0) or 0)
            if not uid:
                return
            _buffers[uid].extend(pcm)
            import time
            _last_pkt[uid] = time.time()
            # cap ~8s stereo 48k 16bit
            if len(_buffers[uid]) > 48000 * 2 * 2 * 8:
                _flush_user(uid, user)

        def cleanup(self) -> None:
            _buffers.clear()

    return WakeSink()


async def _silence_watch() -> None:
    import time
    while _listen_on and _vc and _vc.is_connected():
        await asyncio.sleep(0.4)
        now = time.time()
        for uid, ts in list(_last_pkt.items()):
            if now - ts >= 1.1:
                user = None
                try:
                    ch = getattr(_vc, "channel", None)
                    guild = getattr(ch, "guild", None)
                    user = guild.get_member(uid) if guild else None
                except Exception:
                    user = None
                _flush_user(uid, user)


async def start_listen(member: Any) -> str:
    global _listen_on, _vc
    if not _vc or not _vc.is_connected():
        msg = await join(member)
        if not _vc or not _vc.is_connected():
            return msg
    cls = _recv_cls()
    if cls is None:
        return (
            "Voice receive library missing. Add `discord-ext-voice-recv` to requirements.txt "
            "and redeploy."
        )
    if not hasattr(_vc, "listen"):
        channel = _vc.channel
        try:
            await _vc.disconnect(force=True)
        except Exception:
            pass
        try:
            _vc = await channel.connect(cls=cls, self_deaf=False, reconnect=True)
        except Exception as exc:
            return (
                f"Could not enable listen ({exc}). "
                "Discord DAVE encryption often breaks incoming decode. "
                "Speak-in-VC still works without listen."
            )
    if _listen_on and getattr(_vc, "is_listening", lambda: False)():
        return "Already listening. Say Meepo then your question."
    try:
        if getattr(_vc, "is_listening", lambda: False)():
            _vc.stop_listening()
    except Exception:
        pass
    _listen_on = True
    try:
        await _vc.guild.change_voice_state(channel=_vc.channel, self_deaf=False, self_mute=False)
    except Exception:
        pass
    try:
        _vc.listen(_start_sink())
    except Exception as exc:
        _listen_on = False
        return f"Could not start listen: {exc}"
    asyncio.create_task(_silence_watch())
    return "Listening. Say **Meepo** then the question. `@Meepo stop listen` to mute me."


async def stop_listen() -> str:
    global _listen_on
    _listen_on = False
    if _vc and getattr(_vc, "is_listening", lambda: False)():
        try:
            _vc.stop_listening()
        except Exception:
            pass
    try:
        if _vc and _vc.channel:
            await _vc.guild.change_voice_state(channel=_vc.channel, self_deaf=True, self_mute=False)
    except Exception:
        pass
    return "Stopped listening. Still in the call. I will only speak if you ask in text."


async def handle_voice_command(message: discord.Message) -> bool:
    raw = _clean_activation(message.content or "")
    if not raw:
        return False

    if _LEAVE_RE.search(raw):
        await message.channel.send(await leave())
        return True
    if _LISTEN_OFF_RE.search(raw):
        await message.channel.send(await stop_listen())
        return True
    if _LISTEN_ON_RE.search(raw):
        await message.channel.send(await start_listen(message.author))
        return True
    if _JOIN_RE.search(raw):
        await message.channel.send(await join(message.author))
        return True

    m = _SPEAK_RE.search(raw)
    if m:
        line = (m.group("text") or "").strip()
        line = re.sub(r"^(?:in|on)\s+(?:the\s+)?(?:vc|voice|call)\s*[:,-]?\s*", "", line, flags=re.I)
        if not line:
            await message.channel.send("Example: `@Meepo say in vc raid in 5`.")
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
