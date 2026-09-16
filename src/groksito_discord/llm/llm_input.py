"""
LLM Input Builder for Groksito (Responses API).

Sent to the model on addressed turns (via ``build_responses_input``):
- Exactly one system message: the fixed ``SYSTEM_PROMPT`` (stable per-user cache prefix)
- User message, optionally prefixed with gated dynamic context:
  - Referenced / reply-chain [R:] blocks (direct replies or @mentions)
  - Compact per-guild emoji header (top-used emotes for current server)
- Multimodal vision blocks when images are attached

NOT sent automatically (by design — "let Grok be Grok"):
- Per-user memory / profile buffers (removed from ``context/core.py`` in #112)
- Channel history or rolling summaries (available only via ``get_recent_context`` tool)
- Proactive summarization output (disabled by default in config)

Light context classification (minimal/normal/image_gen) is used only for logging
and native tool gating — not for injecting stored memory.

This module is the single source of truth for ``initial_input`` sent to the API.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from ..config import settings
from ..core.intent import (
    is_image_edit_request,
    is_pure_image_generation_request,
    is_pure_video_generation_request,
)
from ..utils.correlation import cid_prefix
from ..utils.text import filter_unreliable_vision_urls
from ..utils.token_usage import log_context_injection
from .prompt_builder import SYSTEM_PROMPT

logger = logging.getLogger("groksito.llm")


def _classify_query_context_need(text: str, is_reply_continuation: bool = False) -> str:
    """Minimal context-need shim for logging and native tool gating.

    Classification is now extremely light (post-#24). "need" is primarily for:
    - logging and metrics
    - gating native search offering (casual/image_gen get none)
    - pure_*_gen ultra-light paths
    The model decides almost everything via native reasoning + SYSTEM_PROMPT + tool schemas.
    We deliberately avoid reintroducing keyword-heavy tiers.
    """
    t = (text or "").strip()
    if not t:
        return "minimal"
    try:
        if is_pure_video_generation_request(t) or is_pure_image_generation_request(t):
            return "image_gen"
    except Exception:
        pass
    return "normal"


class ResponsesInputData(TypedDict):
    """Lightweight structural type for the return value of build_responses_input.

    This documents the shape without changing any runtime behavior.
    Used by the LLM orchestrator to unpack initial_input, classification need,
    and various context blocks for logging/caching decisions.
    """
    initial_input: list[dict]
    stable_prefix_len: int
    need: str
    user_id: str
    user_message_text: str
    dynamic_context_block: str
    emoji_full_block: str


def _build_multimodal_user_content(
    user_message: str,
    image_urls: list[str] | None,
    attachments: list[dict] | None = None,
) -> list[dict] | str:
    """Construct the user message content for the Responses API.

    Plain string when no images; otherwise multimodal input_text + input_image blocks.
    Vision URLs pass through filter_unreliable_vision_urls as a last-mile guard (#40).
    attachments param accepted (for signature parity with builder) but block injection
    handled in build_responses_input so that attachments always prepend early (before
    context_note) and work for both plain and multimodal cases.
    """
    if not image_urls:
        return user_message or ""

    safe_urls = filter_unreliable_vision_urls(image_urls)
    if not safe_urls:
        return user_message or ""

    content: list[dict] = []
    text = (user_message or "").strip()
    if is_image_edit_request(text, has_reference_image=True):
        text = (
            f"{text}\n\n"
            "[System note: The user attached a reference image and wants a visual transformation. "
            "You MUST call the edit_image tool with their instructions. Do not describe a finished "
            "edit in text alone — the tool delivers the edited image as an attachment.]"
        ).strip()
    content.append({"type": "input_text", "text": text})

    for url in safe_urls[:3]:
        content.append({
            "type": "input_image",
            "image_url": url,
            "detail": "high",
        })

    return content


_DO_NOT_REPEAT_NOTE = (
    "(Context only — do not repeat or paste the bracketed text in your reply.)"
)


def _is_bot_context(ctx: dict) -> bool:
    """True when referenced/chain context was authored by this bot."""
    if ctx.get("is_bot"):
        return True
    author = (ctx.get("author") or "").strip().lower()
    return author in ("groksito", "grok")


def _format_referenced_context_line(
    ref_summary: dict,
    *,
    is_reply_to_bot: bool,
) -> str:
    """Format the direct referenced message for model context."""
    ref_content = (ref_summary.get("content") or "").strip()[:150]
    if is_reply_to_bot and _is_bot_context(ref_summary):
        return (
            "The user is now replying to my previous response:\n"
            f"[My last message] {ref_content}\n"
            f"{_DO_NOT_REPEAT_NOTE}"
        )
    author = ref_summary.get("author", "?")
    return f"[R:{author}] {ref_content}"


def _format_chain_ancestor_line(ctx: dict, index: int) -> str | None:
    """Format a deeper reply-chain ancestor (skips index 0 — same as direct ref)."""
    content = (ctx.get("content") or "").strip()[:100]
    links = (ctx.get("external_links") or [])[:1]
    link_note = f" (link: {links[0]})" if links else ""
    if not (content or links):
        return None
    if _is_bot_context(ctx):
        return f"[My earlier message {index}] {content}"
    author = ctx.get("author", "?")
    return f"[Chain ancestor {index} by {author}]{link_note} {content}"


def _build_dynamic_referenced_context_block(
    *,
    referenced_context: dict | None,
    reply_chain_contexts: list[dict] | None,
    is_reply_to_bot: bool,
    is_mentioned: bool,
) -> str:
    """Build the [R:] + reply-chain ancestor block for addressed turns."""
    if not (is_reply_to_bot or is_mentioned):
        return ""

    context_parts: list[str] = []

    if referenced_context:
        ref_summary = referenced_context
        x_links = ref_summary.get("x_links") or []
        ext_links = ref_summary.get("external_links") or []
        addr = "reply_to_bot" if is_reply_to_bot else "mentioned_in_reply"

        if ref_summary.get("image_urls"):
            logger.info(f"{cid_prefix()}[LLM] High-priority reply context with {len(ref_summary['image_urls'])} image(s) injected (addr={addr})")
        if x_links:
            logger.info(f"{cid_prefix()}[LLM] High-priority reply context with {len(x_links)} X link(s) - model should use x_search if needed (addr={addr})")
        if ext_links and not x_links:
            logger.info(f"{cid_prefix()}[LLM] High-priority reply context with {len(ext_links)} external link(s) (e.g. YouTube) injected (addr={addr})")
        if not (ref_summary.get("image_urls") or x_links or ext_links):
            logger.info(f"{cid_prefix()}[LLM] High-priority reply context injected (text only, addr={addr})")

        context_parts.append(
            _format_referenced_context_line(ref_summary, is_reply_to_bot=is_reply_to_bot)
        )

    if reply_chain_contexts:
        ancestor_lines = []
        for i, ctx in enumerate(reply_chain_contexts[1:3]):
            line = _format_chain_ancestor_line(ctx, i + 1)
            if line:
                ancestor_lines.append(line)
        if ancestor_lines:
            context_parts.append("\n".join(ancestor_lines))
            logger.info(f"{cid_prefix()}[LLM] Injected {len(ancestor_lines)} reply chain ancestor(s) for text referent resolution")

    if not context_parts:
        return ""
    return "\n\n".join(context_parts)


def _build_emoji_block_if_addressed(
    *,
    original_message: Any,
    is_reply_to_bot: bool,
    is_mentioned: bool,
    image_gen_intent: bool = False,
) -> str:
    """Return stable compact emoji header only on addressed turns.

    IMPORTANT (cross-server consistency + emote preservation):
    - Emote knowledge (descriptions + usage) is still fully available and used for *normal* chat.
    - We deliberately SKIP on pure image/video creation turns (image_gen_intent).
      Reason: those turns give the model only a tiny generate_* tool and expect zero text output.
      The English emote header was causing language drift ("sometimes speaks English")
      and inconsistent prompt text passed to the image API across guilds with different emote sets.
    - This does NOT remove Groksito's ability to understand or use emotes.
      Normal addressed conversation turns continue to receive the block so the model
      can decide when/how to use :shortcode: (which is then normalized per-guild).
    """
    if not (is_reply_to_bot or is_mentioned):
        return ""
    if image_gen_intent:
        # Pure creation request (first-turn "genera una imagen...", etc.).
        # Keep input minimal and stable across servers. Emote block is irrelevant here.
        return ""

    try:
        from ..utils import emoji_registry

        gid = None
        guild_obj = None
        try:
            if original_message and getattr(original_message, "guild", None):
                guild_obj = original_message.guild
                gid = getattr(guild_obj, "id", None)
        except Exception:
            pass

        # Now uses top ~8 most-used by real usage (with short descs so model knows when to use).
        # Strictly for current server only. Live guild_obj ensures current emotes + correct IDs.
        # Lightweight and gated to addressed turns.
        emoji_compact_block = emoji_registry.get_emoji_compact_header(gid, guild_obj=guild_obj)
        if emoji_compact_block:
            logger.debug(f"{cid_prefix()}[CONTEXT] Injected server emote knowledge (top used for this guild, addressed turn)")
        return emoji_compact_block
    except Exception as emoji_ctx_err:
        logger.debug(f"{cid_prefix()}[Emoji] emoji prompt injection skipped (non-fatal): {emoji_ctx_err}")
        return ""


def _format_attachment_size(num_bytes: Any) -> str:
    """Human-readable size (e.g. 2.3MB, 4.1KB) matching design examples. Safe on bad input."""
    try:
        b = int(num_bytes or 0)
    except Exception:
        b = 0
    if b <= 0:
        return ""
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.1f}MB"
    if b >= 1024:
        return f"{b / 1024:.1f}KB"
    return f"{b}B"


def _guess_fence_lang(filename: str, content_type: str) -> str:
    """Lightweight fence lang hint for text_content (python, json etc). Empty ok."""
    fn = (filename or "").lower()
    ct = (content_type or "").lower()
    if fn.endswith(".py") or "python" in ct:
        return "python"
    if fn.endswith(".js") or "javascript" in ct:
        return "javascript"
    if fn.endswith(".ts"):
        return "typescript"
    if fn.endswith(".json"):
        return "json"
    if fn.endswith((".md", ".markdown")):
        return "markdown"
    if fn.endswith((".yml", ".yaml")):
        return "yaml"
    if fn.endswith(".sh") or "shell" in ct or "bash" in ct:
        return "bash"
    if fn.endswith(".html") or "html" in ct:
        return "html"
    if fn.endswith(".css"):
        return "css"
    if fn.endswith(".log"):
        return "log"
    return ""


def _build_attachments_block(attachments: list[dict] | None) -> str:
    """New private builder (Task 3).

    Produces the exact compact block per design spec:
    [Attachments sent with this message:
    - funny.gif (image/gif, 2.3MB)
    - main.py (text/x-python, 4.1KB)
      ```python
      ...
      ```
    - report.pdf (application/pdf, 1.4MB)
    ]

    - Always metadata for all attachments (current + some refs).
    - If "text_content" present (small text files from harvest), show indented fenced snippet.
    - Truncates over-long text_content defensively + adds note.
    - Safe, no secrets (size/format already from meta).
    - Vision images listed here for filename awareness (separate input_image still added).
    - Language neutral header; folded into user content (cache friendly).
    """
    if not attachments:
        return ""
    lines: list[str] = ["[Attachments sent with this message:"]
    for a in attachments:
        if not isinstance(a, dict):
            continue
        filename = str(a.get("filename") or "unknown").strip() or "unknown"
        content_type = str(a.get("content_type") or "").strip()
        size_str = _format_attachment_size(a.get("size"))
        if content_type and size_str:
            meta = f"{content_type}, {size_str}"
        elif content_type:
            meta = content_type
        elif size_str:
            meta = size_str
        else:
            meta = ""
        line = f"- {filename} ({meta})" if meta else f"- {filename}"
        lines.append(line)

        tc = a.get("text_content")
        if isinstance(tc, str) and tc.strip():
            display = tc.strip()
            if len(display) > 2000:
                display = display[:2000] + "\n... [truncated for context]"
            lang = _guess_fence_lang(filename, content_type)
            fence = f"  ```{lang}\n{display}\n  ```"
            lines.append(fence)
    lines.append("]")

    return "\n".join(lines)


async def build_responses_input(
    *,
    user_message: str,
    channel_id: int,
    original_message: Any,
    image_urls: list[str] | None,
    referenced_context: dict | None,
    reply_chain_contexts: list[dict] | None = None,  # deeper ancestors from reply chain for text referents (YouTube links, "what the user said", etc.)
    is_reply_continuation: bool,
    has_x_link_intent: bool,
    image_gen_intent: bool = False,
    is_reply_to_bot: bool = False,   # Direct reply to one of our messages
    is_mentioned: bool = False,      # Direct @mention of the bot (strong address signal; also enables referenced context + recent summary)
    attachments: list[dict] | None = None,  # from harvest: current (+ref on continuations); may include "text_content" for small text files
) -> dict[str, Any]:
    """
    The single, correct, non-duplicated input builder for the first Responses API call.

    It is the ONLY place in the codebase that constructs `initial_input`.
    Always uses the concise SYSTEM_PROMPT.

    Referenced context ([R:]) + reply_chain_contexts (deeper ancestors for text) are provided for:
    - Direct replies to the bot, or
    - Direct @mentions (with or without formal reply) when referring to recent user content
      ("what the user said", "the video/image the user posted", links in the thread, etc.).
    Enables deeper text chain walking while staying lightweight (gated + small depth/truncation).

    Recent conversation context is deliberately NOT pre-injected (see #19); it is available
    on-demand only if the model calls the get_recent_context custom tool.
    """
    user_id = ""
    user_message_text = user_message or ""
    try:
        if original_message and getattr(original_message, "author", None):
            user_id = str(getattr(original_message.author, "id", ""))
    except Exception:
        pass
    speaker_name = ""
    try:
        if original_message and getattr(original_message, "author", None):
            speaker_name = (
                str(getattr(original_message.author, "display_name", "") or "")
                or str(getattr(original_message.author, "name", "") or "")
            )
    except Exception:
        pass
    speaker_line = ""
    if user_id or speaker_name:
        speaker_line = (
            f"[Speaker] discord_user_id={user_id} display_name={speaker_name}. "
            "Use this to apply creator/guild rules. Do not print this line."
        )

    # === Context Classification (for tool offering decisions) ===
    # "casual" / "minimal" / "image_gen" -> zero custom tools + no native search (ultra light).
    # "normal" / "rich" -> may offer native web/x_search (x_search only on clear signals for efficiency).
    smart_mode = getattr(settings, "context_smart_mode", True)
    need = "normal"
    try:
        need = _classify_query_context_need(user_message_text, is_reply_continuation=is_reply_continuation)
    except Exception:
        need = "normal"

    if (is_mentioned or is_reply_to_bot) and need not in ("image_gen",):
        need = "normal"
    elif is_reply_continuation and need in ("casual", "minimal"):
        need = "normal"

    if smart_mode and (need == "image_gen" or image_gen_intent):
        logger.debug(f"{cid_prefix()}[CONTEXT] IMAGE_GEN ultra mode (user={user_id[:6]}...)")

    dynamic_context_block = _build_dynamic_referenced_context_block(
        referenced_context=referenced_context,
        reply_chain_contexts=reply_chain_contexts,
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
    )

    emoji_full_block = _build_emoji_block_if_addressed(
        original_message=original_message,
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
        image_gen_intent=image_gen_intent,
    )

    # Attachments block (new for Task 3): metadata + optional text_content for small files.
    # Computed here, prepended early (before [R:]/emoji context_note) into user content.
    # Gated for pure image/video gen paths (like emoji block) to keep input minimal.
    attachments_block = _build_attachments_block(attachments) if attachments and not image_gen_intent else ""

    try:
        injected_chars = len(dynamic_context_block)
        injected_tokens = max(30, injected_chars // 4)
        log_context_injection(
            injected_tokens=injected_tokens,
            description=f"smart:{need}",
            has_memory=False,
            has_summary=False,
        )
    except Exception:
        pass

    user_content = _build_multimodal_user_content(user_message, image_urls, attachments=attachments)

    # Exactly ONE system message: the fixed SYSTEM_PROMPT.
    # This guarantees an identical leading prefix for every first-turn under the
    # same per-user prompt_cache_key. Maximizes reuse of the ~2447-char stable block.
    #
    # Dynamic [R:]/chain and emoji info (when present on addressed turns) are folded
    # as a short prefix *inside the user message*. This eliminates variable extra
    # system messages that previously reduced reliable prefix matching, while still
    # surfacing the referent and emoji shortcode guidance to the model at the start
    # of the user turn (where attention is strong).
    #
    # Previous design (separate system msgs after the first) was intended to protect
    # the prefix but in practice the varying message count + varying block content
    # limited cache effectiveness (observed 5-17% hit rates).
    system_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Build tiny optional context note (dynamic ref + compact emoji header).
    # Both are already gated to addressed turns inside their builders.
    # Attachments block (if present) is prepended BEFORE the ref/emoji note, per design
    # (it's "Attachments sent with this message", most immediate to current turn content).
        context_prefix_parts: list[str] = []
    if speaker_line:
        context_prefix_parts.append(speaker_line)
    if attachments_block:
        context_prefix_parts.append(attachments_block)
    if dynamic_context_block:
        context_prefix_parts.append(dynamic_context_block)
    if emoji_full_block:
        context_prefix_parts.append(emoji_full_block)
    context_note = "\n\n".join(context_prefix_parts).strip()

    # Prepend context note (now may include attachments_block prepended to it) into
    # user_content (handles both str case and multimodal list case for vision).
    if context_note:
        if isinstance(user_content, list):
            # Multimodal path (vision). Put note as the first text block.
            if user_content and isinstance(user_content[0], dict) and user_content[0].get("type") == "input_text":
                orig = user_content[0].get("text") or ""
                user_content[0]["text"] = f"{context_note}\n\n{orig}".strip()
            else:
                user_content = [{"type": "input_text", "text": context_note}] + user_content
        else:
            user_content = f"{context_note}\n\n{user_content}".strip() if (user_content or "").strip() else context_note

    initial_input: list[dict] = system_messages + [{"role": "user", "content": user_content}]
    stable_prefix_len = len(SYSTEM_PROMPT)

    result: ResponsesInputData = {
        "initial_input": initial_input,
        "stable_prefix_len": stable_prefix_len,
        "need": need,
        "user_id": user_id,
        "user_message_text": user_message_text,
        "dynamic_context_block": dynamic_context_block,
        "emoji_full_block": emoji_full_block,
    }
    return result
