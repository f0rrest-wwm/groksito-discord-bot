"""
Public media-tools surface for Groksito.

Callers (llm/tools.py, llm/client.py, health.py) import from here for:
- Feature flags (ENABLE_VIDEO_GENERATION)
- Explicit intent gates (has_explicit_video_intent, has_explicit_audio_intent) — re-exported
  from core.intent (central home for all keyword signals, following ARCHITECTURE.md)
- Handler entry points re-exported from the canonical media handlers

Implementations live in:
- media/image_handler.py  (Text-to-Image + Image-Edit)
- media/video_handler.py  (Text-to-Video + Image-to-Video)
- media/audio_handler.py  (TTS)
- media/delivery.py       (direct delivery + request tracking)
"""

from __future__ import annotations

from ..config import settings


# =============================================================================
# Feature Flags
# =============================================================================

ENABLE_VIDEO_GENERATION: bool = settings.enable_video_generation


# Re-export explicit intent gates (authoritative defs live in core/intent.py alongside other
# keyword signals, per architecture). This keeps the media_tools surface stable.
from ..core.intent import (
    has_explicit_video_intent,
    has_explicit_audio_intent,
)


# =============================================================================
# Handler entry points (canonical implementations in media/*_handler.py)
# =============================================================================

# Video — now served from the dedicated modern handler (following the image_handler pattern)
from ..media.video_handler import (
    _generate_video_schema,
    _generate_video_schema_tiny,
    _handle_generate_video,
)

# Image generation & editing — now served from the dedicated modern handler
from ..media.image_handler import (
    _handle_generate_image,
    _handle_edit_image,
)
from ..media.search_edit import _handle_search_edit_image

# Audio / TTS — served from dedicated handler (src/groksito_discord/media/audio_handler.py)
# The handler resolves tts_default_voice / tts_default_language from settings (pydantic + .env, editable in web UI)
# with sensible fallbacks ("eve", "es"). Language is required by the official /v1/tts API.
from ..media.audio_handler import (
    _generate_audio_schema,
    _handle_generate_audio,
)

# Optional: re-export the new prompt enhancer for advanced use / tests
from ..media.image_handler import _enhance_prompt_for_api, soften_image_prompt

# Explicit intent detectors are re-exported above from core.intent (central location for
# all keyword signals per ARCHITECTURE.md). The implementations were consolidated to keep
# visual/audio detection co-located with is_pure_* and edit detectors.

