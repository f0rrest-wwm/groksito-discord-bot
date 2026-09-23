"""
LLM / Responses API layer for Groksito (standalone).

Clean modular orchestrator:
- Input construction lives exclusively in llm_input.py (single source of truth)
- Helpers in llm_utils.py
- This file focuses on call flow, multi-round tool execution, and orchestration.

Public API preserved for compatibility:
- call_grok_for_groksito (and alias call_grok_with_tools)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from ..utils.correlation import cid_prefix
from ..utils.errors import format_tool_execution_error, is_image_fetch_404_error

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIError

from ..config import settings
from ..media.delivery import DIRECT_DELIVERY_PERFORMED
from .prompt_builder import DIRECT_DELIVERY_DETECTOR_PHRASES
from .tools import (
    ASSET_RESOLVER_TOOLS,
    get_tools_for_request,
    log_tool_selection,
    execute_hybrid_tool,
)

# For the explicit video intent guard (Python-level safety net in addition to prompt/schema)
from .media_tools import has_explicit_video_intent, has_explicit_audio_intent

# Import from the new sibling modules (clean separation)
from .llm_input import build_responses_input
from ..context import should_offer_light_decision_tools
from .llm_utils import (
    _extract_final_text,
    _build_stub_response,
    _get_prompt_cache_key,
    _build_native_search_tools,
    _detect_visual_intent,
    _detect_image_creation_intent,
    is_image_edit_request,
    _infer_tools_set_name,
    _extract_and_log_token_usage,
    _maybe_proactive_summarize,
    is_pure_image_generation_request,
    is_pure_video_generation_request,
    _call_responses_with_retry,
)

try:
    from ..config import settings as _settings
except Exception:
    _settings = None  # type: ignore

# OAuth / unified bearer support (lazy; central resolver prefers OAuth token when available)
try:
    from ..core.grok_oauth import get_grok_bearer as _get_grok_bearer
except Exception:
    _get_grok_bearer = None  # type: ignore

logger = logging.getLogger("groksito.llm")


def _is_policy_denied(err: BaseException) -> bool:
    """xAI content/policy 403 — not a dead OAuth token."""
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    if "can't help with that request" in text or "permission-denied" in text:
        return True
    if status == 403 and "expired" not in text and "unauthor" not in text:
        return True
    return False



# =============================================================================
# previous_response_id Multi-Turn Contract
# =============================================================================
# - Custom tools are minimized on continuations (get_continuation_tools) because
#   the model retains prior tool declarations via previous_response_id.
# - Native search re-inclusion is conservative: default [] on continuations;
#   re-offered only when _should_reoffer_native_search_on_continuation detects
#   prior-round search activity and no respond_directly/delivery short-circuit.
# - Vision images are sent only on the first turn of a logical user message;
#   continuations carry text/tool results only (previous_response_id chains state).
# - DIRECT_DELIVERY short-circuit must happen before sending tool outputs back
#   (guarantees no duplicate Discord replies via DIRECT_DELIVERY_PERFORMED).
# - First-turn `input` (via llm_input.build_responses_input) always uses *exactly one*
#   system message (the fixed SYSTEM_PROMPT) + a single user message. Light context
#   notes ([R:] refs + compact emoji header) are folded into the user content on
#   addressed turns. This design maximizes prompt_cache_key prefix hit rate on the
#   stable prefix while previous_response_id handles multi-round tool state.

MEDIA_ACTION_TOOLS = frozenset({
    "generate_image",
    "edit_image",
    "generate_video",
    "generate_audio",
    "reply_to_user",
})

_MEDIA_DELIVERY_TOOLS = frozenset({
    "generate_image",
    "edit_image",
    "generate_video",
    "generate_audio",
})

_CONTINUATION_NO_SEARCH_REOFFER_TOOLS = frozenset({
    "respond_directly",
    *MEDIA_ACTION_TOOLS,
})

_DIRECT_DELIVERY_SUCCESS_PHRASES = DIRECT_DELIVERY_DETECTOR_PHRASES


def _is_direct_delivery_success(result_str: str, tool_name: str, cid_p: str) -> bool:
    """Detect explicit direct-delivery success from media/reply_to_user tool results."""
    lowered = result_str.lower()
    if any(phrase in lowered for phrase in _DIRECT_DELIVERY_SUCCESS_PHRASES):
        logger.info(
            f"{cid_p}[LLM] Direct delivery SUCCESS for tool '{tool_name}' "
            "— suppressing final text reply"
        )
        return True
    if "policy blocked" in lowered and "clean direct message delivered" in lowered:
        return True
    return False


def _finalize_response(response: Any, direct_delivery_performed: bool, cid_p: str) -> str | object:
    """Extract final text or return DIRECT_DELIVERY_PERFORMED sentinel."""
    final_text = _extract_final_text(response)

    if direct_delivery_performed:
        logger.info(
            f"{cid_p}[LLM] Direct media/action delivery performed "
            "— returning DIRECT_DELIVERY_PERFORMED sentinel (no second reply)"
        )
        return DIRECT_DELIVERY_PERFORMED

    if final_text:
        return final_text.strip()

    return "✅ "


def _should_offer_light_decision(
    user_message_text: str,
    user_message: str,
    *,
    is_mentioned: bool,
    is_reply_to_bot: bool,
    context_need: str,
) -> bool:
    """Wrap should_offer_light_decision_tools with safe defaulting."""
    try:
        return should_offer_light_decision_tools(
            is_mentioned=is_mentioned,
            is_reply_to_bot=is_reply_to_bot,
            context_need=context_need,
            user_message=user_message_text or user_message,
        )
    except Exception:
        return False


def _should_reoffer_native_search_on_continuation(
    prev_response: Any,
    *,
    native_search_tools: list[dict],
    executed_tool_names: set[str],
) -> bool:
    """Conservative native-search re-offer on continuation rounds.

    Default is no re-offer (rely on previous_response_id). Re-include schemas only
    when the prior response shows search activity. Short-circuit when the model
    just invoked respond_directly or a delivery tool in the current round.
    """
    if not native_search_tools:
        return False

    if executed_tool_names & _CONTINUATION_NO_SEARCH_REOFFER_TOOLS:
        return False

    try:
        prev_output = getattr(prev_response, "output", None) or []
        for item in prev_output:
            itype = getattr(item, "type", None)
            if isinstance(item, dict):
                itype = item.get("type")
            itype_str = str(itype or "").lower()

            if itype_str == "function_call":
                name = getattr(item, "name", None)
                if name is None and isinstance(item, dict):
                    name = item.get("name")
                if name in ("web_search", "x_search"):
                    return True

            if itype_str and (
                "web_search" in itype_str
                or "x_search" in itype_str
                or "search_call" in itype_str
            ):
                return True
    except Exception:
        pass
    return False


async def _prepare_first_turn_data(
    *,
    user_message: str,
    channel_id: int,
    original_message: Any,
    image_urls: list[str] | None,
    attachments: list[dict] | None = None,
    referenced_context: dict | None,
    reply_chain_contexts: list[dict] | None,
    is_reply_continuation: bool,
    has_x_link_intent: bool,
    is_reply_to_bot: bool,
    is_mentioned: bool,
) -> dict[str, Any]:
    """Phase 1 prep: pure-intent detection + authoritative build_responses_input."""
    pure_video_gen_intent = False
    pure_image_gen_intent = False
    try:
        if (
            is_pure_video_generation_request(user_message)
            and not bool(image_urls)
            and not is_reply_continuation
        ):
            pure_video_gen_intent = True
        elif (
            is_pure_image_generation_request(user_message)
            and not bool(image_urls)
            and not is_reply_continuation
        ):
            pure_image_gen_intent = True
    except Exception:
        pure_video_gen_intent = False
        pure_image_gen_intent = False

    input_data = await build_responses_input(
        user_message=user_message,
        channel_id=channel_id,
        original_message=original_message,
        image_urls=image_urls,
        attachments=attachments,
        referenced_context=referenced_context,
        reply_chain_contexts=reply_chain_contexts,
        is_reply_continuation=is_reply_continuation,
        has_x_link_intent=has_x_link_intent,
        image_gen_intent=pure_image_gen_intent or pure_video_gen_intent,
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
    )

    return {
        "input_data": input_data,
        "pure_video_gen_intent": pure_video_gen_intent,
        "pure_image_gen_intent": pure_image_gen_intent,
    }


def _select_tools_for_first_turn(
    *,
    user_message_text: str,
    user_message: str,
    need: str,
    image_urls: list[str] | None,
    has_visual_intent: bool,
    is_mentioned: bool,
    is_reply_to_bot: bool,
    is_addressed: bool,
    pure_image_gen_intent: bool,
    pure_video_gen_intent: bool,
) -> dict[str, Any]:
    """Phase 2: intent signals + custom/native tool schema selection for first turn."""
    explicit_video_intent = has_explicit_video_intent(user_message_text)
    explicit_audio_intent = has_explicit_audio_intent(user_message_text)

    creation_visual_intent = (
        has_visual_intent
        or _detect_image_creation_intent(
            user_message_text,
            has_reference_image=bool(image_urls),
        )
        or (bool(image_urls) and is_image_edit_request(user_message_text, has_reference_image=True))
    )
    vision_or_visual_query = (
        bool(image_urls) or _detect_visual_intent(user_message_text) or creation_visual_intent or explicit_video_intent
    )
    effective_visual_intent = creation_visual_intent

    offer_light_decision_tools = _should_offer_light_decision(
        user_message_text,
        user_message,
        is_mentioned=is_mentioned,
        is_reply_to_bot=is_reply_to_bot,
        context_need=need,
    )

    custom_tools = get_tools_for_request(
        query_need=need,
        has_visual_intent=effective_visual_intent,
        has_explicit_video_intent=explicit_video_intent,
        has_explicit_audio_intent=explicit_audio_intent,
        is_tool_continuation=False,
        pure_image_gen=pure_image_gen_intent,
        pure_video_gen=pure_video_gen_intent,
        offer_light_decision_tools=offer_light_decision_tools,
    )

    if need in ("casual", "image_gen") or (need == "minimal" and not is_addressed):
        native_search_tools: list[dict] = []
    else:
        native_search_tools = _build_native_search_tools(
            query_text=user_message_text,
            context_need=need,
            has_visual_intent=vision_or_visual_query,
            has_attached_images=bool(image_urls),
        )

    return {
        "custom_tools": custom_tools,
        "native_search_tools": native_search_tools,
        "effective_visual_intent": effective_visual_intent,
        "explicit_video_intent": explicit_video_intent,
        "explicit_audio_intent": explicit_audio_intent,
        "offer_light_decision_tools": offer_light_decision_tools,
    }


async def _execute_tool_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    response: Any,
    need: str,
    user_id: str,
    stable_prefix_len: int,
    effective_visual_intent: bool,
    explicit_video_intent: bool,
    explicit_audio_intent: bool,
    pure_image_gen_intent: bool,
    pure_video_gen_intent: bool,
    native_search_tools: list[dict],
    offered_custom_tool_names: set[str],
    original_message: Any,
    image_urls: list[str] | None,
    is_addressed: bool,
    cid_p: str,
    max_tool_rounds: int = 3,
) -> tuple[Any, bool, bool, bool]:
    """Phase 3: multi-round tool execution + continuation via previous_response_id.

    Returns (final_response, direct_delivery_performed, model_chose_search, model_chose_direct).
    """
    direct_delivery_performed = False
    model_chose_search = False
    model_chose_direct = False
    asset_references_resolved = False
    media_delivered = False

    for round_num in range(1, max_tool_rounds + 1):
        client_tool_outputs: list[dict] = []
        round_executed_tools: set[str] = set()

        output_items = getattr(response, "output", None) or []
        for item in output_items:
            if getattr(item, "type", None) != "function_call":
                continue

            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")

            if name in ("web_search", "x_search"):
                model_chose_search = True
            if name == "respond_directly":
                model_chose_direct = True

            raw_args = getattr(item, "arguments", None)
            if raw_args is None and isinstance(item, dict):
                raw_args = item.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}

            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            if call_id is None and isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id")

            try:
                from ..config import settings as _s
                if getattr(_s, "log_tool_selection", True):
                    from .tools import tools_logger as _tl
                    decision_tool_names = {
                        "web_search",
                        "x_search",
                        "get_recent_context",
                        "get_user_avatar",
                        "get_top_server_emoji",
                        "respond_directly",
                    }
                    if name in decision_tool_names:
                        _arg_keys = (
                            list((raw_args or {}).keys())
                            if isinstance(raw_args, dict)
                            else []
                        )
                        _tl.info(
                            f"{cid_p}[GROK_CHOICE] tool={name} | round={round_num} | "
                            f"keys={_arg_keys} | addressed={is_addressed}"
                        )
            except Exception:
                pass

            logger.info(
                f"{cid_p}[LLM] Round {round_num}: executing custom tool '{name}' "
                f"(args keys: {list((raw_args or {}).keys()) if isinstance(raw_args, dict) else 'n/a'})"
            )

            available = name in offered_custom_tool_names
            tool_type = "custom" if name in offered_custom_tool_names else "native-or-unknown"
            try:
                from .tools import tools_logger
                tools_logger.debug(
                    f"{cid_p}[TOOLS] execution | tool={name} | available={str(available).lower()} "
                    f"| type={tool_type} | round={round_num}"
                )
            except Exception:
                pass

            try:
                result = await execute_hybrid_tool(
                    name=name or "unknown_tool",
                    args=raw_args if isinstance(raw_args, dict) else {},
                    original_message=original_message,
                    image_urls=image_urls,
                )
            except Exception as tool_exec_err:
                arg_keys = (
                    list((raw_args or {}).keys())
                    if isinstance(raw_args, dict)
                    else None
                )
                result = format_tool_execution_error(
                    name or "unknown_tool",
                    tool_exec_err,
                    round_num=round_num,
                    arg_keys=arg_keys,
                )
                logger.error(f"{cid_p}[TOOLS] {result}", exc_info=True)

            if name:
                round_executed_tools.add(name)

            result_str = str(result)
            logger.info(
                f"{cid_p}[LLM] Round {round_num}: tool '{name}' completed, "
                f"result length={len(result_str)}"
            )

            if name in ASSET_RESOLVER_TOOLS and "RESOLVED" in result_str.upper():
                asset_references_resolved = True

            if name in MEDIA_ACTION_TOOLS:
                if _is_direct_delivery_success(result_str, name or "", cid_p):
                    direct_delivery_performed = True
                    if name in _MEDIA_DELIVERY_TOOLS:
                        media_delivered = True

            client_tool_outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_str[:4000],
            })

        if not client_tool_outputs:
            logger.info(
                f"{cid_p}[LLM] Round {round_num}: no more client-side tool calls "
                "— conversation complete."
            )
            break

        if direct_delivery_performed:
            premature_reply = (
                asset_references_resolved
                and not media_delivered
                and bool(round_executed_tools & {"reply_to_user"})
                and not (round_executed_tools & _MEDIA_DELIVERY_TOOLS)
            )
            if premature_reply:
                logger.info(
                    f"{cid_p}[LLM] Round {round_num}: suppressing premature reply_to_user "
                    "short-circuit — asset reference resolved but media tool not called yet"
                )
                direct_delivery_performed = False
            else:
                logger.info(
                    f"{cid_p}[LLM] Round {round_num}: direct delivery performed "
                    "— short-circuiting (skip sending tool results back + no further rounds). "
                    "Natural + cheap."
                )
                break

        logger.info(
            f"{cid_p}[LLM] Round {round_num}: sending back {len(client_tool_outputs)} "
            "tool result(s) using previous_response_id"
        )

        try:
            prev_id = getattr(response, "id", None)
            cache_key = _get_prompt_cache_key(original_message)

            has_resolved_asset_references = asset_references_resolved and bool(image_urls)
            pending_media_after_asset = (
                has_resolved_asset_references
                and (explicit_video_intent or effective_visual_intent)
                and not media_delivered
            )

            continuation_tools = get_tools_for_request(
                query_need=need,
                has_visual_intent=effective_visual_intent,
                has_explicit_video_intent=explicit_video_intent,
                has_explicit_audio_intent=explicit_audio_intent,
                is_tool_continuation=True,
                pure_image_gen=pure_image_gen_intent,
                pure_video_gen=pure_video_gen_intent,
                has_resolved_asset_references=has_resolved_asset_references,
                pending_media_after_asset=pending_media_after_asset,
            )

            continuation_native_search_tools = (
                native_search_tools
                if _should_reoffer_native_search_on_continuation(
                    response,
                    native_search_tools=native_search_tools,
                    executed_tool_names=round_executed_tools,
                )
                else []
            )

            offered_custom_tool_names = {t.get("name") for t in continuation_tools if t.get("name")}
            try:
                img_search = False
                img_understand = False
                for t in continuation_native_search_tools:
                    if t.get("type") == "web_search":
                        img_search = t.get("enable_image_search", False)
                        img_understand = t.get("enable_image_understanding", False)
                        break

                log_tool_selection(
                    turn_type="continuation",
                    query_need=need,
                    has_visual_intent=effective_visual_intent,
                    custom_tools=continuation_tools,
                    native_search_tools=continuation_native_search_tools,
                    enable_image_search=img_search,
                    enable_image_understanding=img_understand,
                )
            except Exception as log_err:
                logger.debug(f"{cid_p}[TOOLS] selection logging failed (continuation): {log_err}")

            response = await _call_responses_with_retry(
                client,
                model=model,
                input=client_tool_outputs,
                previous_response_id=prev_id,
                tools=[
                    *continuation_native_search_tools,
                    *continuation_tools,
                ],
                extra_body={"prompt_cache_key": cache_key},
            )
        except Exception as continue_err:
            logger.warning(
                f"{cid_p}[LLM] Continuation with previous_response_id failed: {continue_err} "
                "— stopping tool loop"
            )
            break

        continuation_cache_context = {
            "turn_type": "continuation",
            "query_need": need,
            "has_visual_intent": effective_visual_intent,
            "custom_tools_count": len(continuation_tools),
            "custom_tools_set": _infer_tools_set_name(need, effective_visual_intent, True),
            "user_id": user_id,
            "prefix_stability_indicator": f"sys~{stable_prefix_len}",
        }

        _extract_and_log_token_usage(
            response,
            model=model,
            has_images=bool(image_urls),
            category="Tool",
            is_tool_continuation=True,
            cache_context=continuation_cache_context,
        )

    return response, direct_delivery_performed, model_chose_search, model_chose_direct


async def call_grok_for_groksito(
    user_message: str,
    author_name: str,
    channel_id: int,
    original_message: Any = None,
    image_urls: list[str] | None = None,
    attachments: list[dict] | None = None,
    referenced_context: dict | None = None,
    reply_chain_contexts: list[dict] | None = None,  # deeper reply ancestors for text referents (links, "what the user said", etc.)
    has_visual_intent: bool = False,
    is_reply_continuation: bool = False,
    has_x_link_intent: bool = False,
    is_reply_to_bot: bool = False,  # Direct reply to one of our messages (affects referenced context + some heuristics)
    is_mentioned: bool = False,
) -> str | object:
    """
    Main entry point from the conversational flow (Responses API + hybrid tool loop).

    Key efficiency features on continuations (using previous_response_id):
    - Custom tools are minimized via get_continuation_tools (only reply_to_user by default).
    - Native search tools (web_search + x_search) are NOT re-sent by default.
      (x_search is offered less frequently overall due to stricter signal checks on first turn.)
      We rely on the model retaining prior tool declarations via previous_response_id
      (see comment in the continuation block for details + conditional re-inclusion
      if search was used in the prior round). This reduces token bloat from tool
      schemas/descriptions on follow-up turns.

    Returns:
        - str: normal final assistant text
        - DIRECT_DELIVERY_PERFORMED sentinel: a media tool already replied directly
    """
    cid_p = cid_prefix()
    logger.debug(f"{cid_p}[LLM] call_grok_for_groksito called for {author_name}")

    model = getattr(settings, "grok_model", None) or "grok-4.3"

    # Resolve bearer via central helper: prefers valid OAuth token (proactive refresh) when available,
    # with seamless fallback to XAI_API_KEY. Works for GROK_AUTH_MODE=auto / oauth / (even api_key if token present).
    bearer: Optional[str] = None
    if _get_grok_bearer:
        bearer = _get_grok_bearer()
    if not bearer:
        bearer = getattr(settings, "xai_api_key", None) or os.getenv("XAI_API_KEY")  # last-ditch env
    if not bearer:
        logger.warning("[LLM] No Grok credential (no OAuth token and no XAI_API_KEY). Using stub response.")
        return _build_stub_response(user_message, author_name, image_urls)

    if settings.using_oauth or (settings.auth_mode == "auto" and bearer and len(bearer) < 100):  # rough heuristic: oauth tokens are JWT-ish
        logger.debug("[LLM] Using xAI Grok OAuth bearer (SuperGrok / X Premium+)")
    else:
        logger.debug("[LLM] Using XAI_API_KEY (api_key mode or oauth fallback)")

    try:
        # Apply configured timeout to the client (affects all responses.create calls).
        # This prevents indefinite hangs on slow/stuck xAI endpoints.
        client = AsyncOpenAI(
            api_key=bearer,
            base_url="https://api.x.ai/v1",
            timeout=settings.api_timeout_seconds,
        )

        # Proactive summarization ΓÇö disabled by default (maximum nativeness).
        # Grok's large context handles long threads naturally. Enable only for pathological cases.
        if getattr(settings, "summarization_enabled", False):
            await _maybe_proactive_summarize(channel_id, original_message, client)

        # === Phase 1: Credential prep (above) + input construction ===
        prep = await _prepare_first_turn_data(
            user_message=user_message,
            channel_id=channel_id,
            original_message=original_message,
            image_urls=image_urls,
            attachments=attachments,
            referenced_context=referenced_context,
            reply_chain_contexts=reply_chain_contexts,
            is_reply_continuation=is_reply_continuation,
            has_x_link_intent=has_x_link_intent,
            is_reply_to_bot=is_reply_to_bot,
            is_mentioned=is_mentioned,
        )
        input_data = prep["input_data"]
        pure_video_gen_intent = prep["pure_video_gen_intent"]
        pure_image_gen_intent = prep["pure_image_gen_intent"]

        initial_input = input_data["initial_input"]
        stable_prefix_len = input_data["stable_prefix_len"]
        need = input_data["need"]
        user_id = input_data["user_id"]
        user_message_text = input_data["user_message_text"]
        is_addressed = bool(is_mentioned or is_reply_to_bot)

        if has_x_link_intent:
            logger.info(f"{cid_p}[LLM] X/Link intent detected in reply — boosting context awareness for referenced links")

        # === Phase 2: First-turn tool selection + native search offering ===
        tool_selection = _select_tools_for_first_turn(
            user_message_text=user_message_text,
            user_message=user_message,
            need=need,
            image_urls=image_urls,
            has_visual_intent=has_visual_intent,
            is_mentioned=is_mentioned,
            is_reply_to_bot=is_reply_to_bot,
            is_addressed=is_addressed,
            pure_image_gen_intent=pure_image_gen_intent,
            pure_video_gen_intent=pure_video_gen_intent,
        )
        custom_tools = tool_selection["custom_tools"]
        native_search_tools = tool_selection["native_search_tools"]
        effective_visual_intent = tool_selection["effective_visual_intent"]
        explicit_video_intent = tool_selection["explicit_video_intent"]
        explicit_audio_intent = tool_selection["explicit_audio_intent"]

        # Structured tool selection logging
        offered_custom_tool_names = {t.get("name") for t in custom_tools if t.get("name")}
        try:
            img_search = False
            img_understand = False
            for t in native_search_tools:
                if t.get("type") == "web_search":
                    img_search = t.get("enable_image_search", False)
                    img_understand = t.get("enable_image_understanding", False)
                    break

            log_tool_selection(
                turn_type="first_turn",
                query_need=need,
                has_visual_intent=effective_visual_intent,
                custom_tools=custom_tools,
                native_search_tools=native_search_tools,
                enable_image_search=img_search,
                enable_image_understanding=img_understand,
            )
        except Exception as log_err:
            logger.debug(f"{cid_p}[TOOLS] selection logging failed: {log_err}")

        # Lightweight instrumentation for addressed turns only.
        # Capture start for first-turn + full tool loop latency; flag for search schemas offered.
        addressed_turn_start = time.time() if is_addressed else None
        search_schemas_offered = bool(native_search_tools)

        # First call to Responses API (vision images are sent here)
        # Uses retry helper for transients (429/5xx/timeout); non-transients (policy, auth, bad payload) fail fast for caller classification.
        try:
            cache_key = _get_prompt_cache_key(original_message)
            response = await _call_responses_with_retry(
                client,
                model=model,
                input=initial_input,
                tools=[
                    *native_search_tools,
                    *custom_tools,
                ],
                extra_body={"prompt_cache_key": cache_key},
            )
        except Exception as api_err:
            is_404 = is_image_fetch_404_error(api_err, has_images=bool(image_urls))
            has_media = bool(image_urls) or bool(attachments)
            cache_key = _get_prompt_cache_key(original_message)

            async def _retry_first_turn(
                *,
                retry_images: list,
                retry_attachments: list | None,
                retry_ref: dict | None,
                retry_chain: list | None,
                retry_reply_cont: bool,
                label: str,
            ):
                rebuilt = await build_responses_input(
                    user_message=user_message,
                    channel_id=channel_id,
                    original_message=original_message,
                    image_urls=retry_images,
                    attachments=retry_attachments,
                    referenced_context=retry_ref,
                    reply_chain_contexts=retry_chain,
                    is_reply_continuation=retry_reply_cont,
                    has_x_link_intent=has_x_link_intent,
                    image_gen_intent=pure_image_gen_intent or pure_video_gen_intent,
                    is_reply_to_bot=False,
                    is_mentioned=is_mentioned,
                )
                logger.warning(f"{cid_p}[LLM] {label}")
                return await _call_responses_with_retry(
                    client,
                    model=model,
                    input=rebuilt["initial_input"],
                    tools=[*native_search_tools, *custom_tools],
                    extra_body={"prompt_cache_key": cache_key},
                )

            if _is_policy_denied(api_err):
                logger.warning(
                    f"{cid_p}[LLM][POLICY] First turn denied ({api_err}). "
                    "Retrying stripped (no media, no reply context)."
                )
                try:
                    response = await _retry_first_turn(
                        retry_images=[],
                        retry_attachments=[],
                        retry_ref=None,
                        retry_chain=None,
                        retry_reply_cont=False,
                        label="[POLICY] stripped retry",
                    )
                    image_urls = []
                    attachments = []
                    referenced_context = None
                    reply_chain_contexts = None
                    logger.info(f"{cid_p}[LLM][POLICY] Stripped retry succeeded.")
                except Exception as policy_retry_err:
                    logger.error(f"{cid_p}[LLM][POLICY] Stripped retry failed: {policy_retry_err}")
                    raise

            elif is_404 or has_media:
                if is_404:
                    logger.warning(
                        f"{cid_p}[LLM][VISION] Image fetch 404 from xAI backend for {len(image_urls or [])} provided URL(s). "
                        f"These were likely stale Discord signed attachment URLs or transient embed previews from recent channel history. "
                        f"Retrying first turn WITHOUT images (attachments metadata preserved if any)."
                    )
                else:
                    logger.warning(
                        f"{cid_p}[LLM][ATTACHMENTS] Responses API first-turn trouble (non-404 processing error, e.g. GIF/unsupported) "
                        f"with attachments={len(attachments or [])} (image_urls={len(image_urls or [])}). "
                        f"Retrying with image_urls cleared but attachments kept so model receives metadata."
                    )
                try:
                    response = await _retry_first_turn(
                        retry_images=[],
                        retry_attachments=attachments,
                        retry_ref=referenced_context,
                        retry_chain=reply_chain_contexts,
                        retry_reply_cont=is_reply_continuation,
                        label="[ATTACHMENTS] retry images cleared",
                    )
                    image_urls = []
                    logger.info(f"{cid_p}[LLM][ATTACHMENTS] First-turn retry with attachments (images cleared) succeeded.")
                except Exception as retry_err:
                    logger.error(f"{cid_p}[LLM][ATTACHMENTS] Retry without images also failed: {retry_err}")
                    raise
            else:
                raise

        # Capture first-turn prompt tokens for addressed metrics (minimal extraction, reuse patterns from _extract)
        first_turn_prompt_tokens = 0
        try:
            usage = getattr(response, "usage", None)
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            if usage:
                if hasattr(usage, "input_tokens"):
                    first_turn_prompt_tokens = getattr(usage, "input_tokens", 0) or 0
                elif isinstance(usage, dict):
                    first_turn_prompt_tokens = usage.get("input_tokens", 0) or 0
        except Exception:
            first_turn_prompt_tokens = 0

        # Token logging for first turn
        category = "Vision" if image_urls else "Conversation"
        first_turn_cache_context = {
            "turn_type": "first_turn",
            "query_need": need,
            "has_visual_intent": effective_visual_intent,
            "custom_tools_count": len(custom_tools),
            "custom_tools_set": _infer_tools_set_name(need, effective_visual_intent, False),
            "user_id": user_id,
            "prefix_stability_indicator": f"sys~{stable_prefix_len}",
        }
        _extract_and_log_token_usage(
            response,
            model=model,
            has_images=bool(image_urls),
            category=category,
            is_tool_continuation=False,
            cache_context=first_turn_cache_context,
        )

        # === Phase 3: Tool execution loop + continuation (previous_response_id) ===
        response, direct_delivery_performed, model_chose_search, model_chose_direct = (
            await _execute_tool_loop(
                client=client,
                model=model,
                response=response,
                need=need,
                user_id=user_id,
                stable_prefix_len=stable_prefix_len,
                effective_visual_intent=effective_visual_intent,
                explicit_video_intent=explicit_video_intent,
                explicit_audio_intent=explicit_audio_intent,
                pure_image_gen_intent=pure_image_gen_intent,
                pure_video_gen_intent=pure_video_gen_intent,
                native_search_tools=native_search_tools,
                offered_custom_tool_names=offered_custom_tool_names,
                original_message=original_message,
                image_urls=image_urls,
                is_addressed=is_addressed,
                cid_p=cid_p,
            )
        )

        # Emit addressed-turn metrics (lightweight, defensive).
        if is_addressed and addressed_turn_start is not None:
            try:
                latency_ms = (time.time() - addressed_turn_start) * 1000.0
                from ..utils.token_usage import log_addressed_turn_metrics
                log_addressed_turn_metrics(
                    latency_ms=latency_ms,
                    prompt_tokens=first_turn_prompt_tokens,
                    search_schemas_offered=search_schemas_offered,
                    model_chose_search=model_chose_search,
                    model_chose_direct=model_chose_direct,
                    query_need=need or "unknown",
                )
            except Exception:
                pass  # never break main flow for metrics

        # Final extraction + sentinel handling
        return _finalize_response(response, direct_delivery_performed, cid_p)

    except Exception as e:
        logger.exception(f"{cid_p}Error during real Responses API call + tool loop")

        if _is_policy_denied(e):
            return "Can't do that request."

        if image_urls and not attachments:
            logger.warning(f"{cid_p}[LLM][VISION] Failing request had {len(image_urls)} image(s) attached.")
            return "Error. <@253869773421674498>, Check logs to fix it."

        if attachments:
            logger.warning(f"{cid_p}[LLM][ATTACHMENTS] Failing request had {len(attachments)} attachment(s).")

        if isinstance(e, (RateLimitError,)) or "rate" in str(e).lower() or "429" in str(e).lower():
            return "Error. <@253869773421674498>, Check logs to fix it."
        if isinstance(e, (APITimeoutError, APIConnectionError)) or "timeout" in str(e).lower() or "connection" in str(e).lower():
            return "Error. <@253869773421674498>, Check logs to fix it."

        if isinstance(e, APIError):
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600:
                return "Error. <@253869773421674498>, Check logs to fix it."
            if status == 401:
                return "Error. <@253869773421674498>, Check logs to fix it."

        return "Error. <@253869773421674498>, Check logs to fix it."

    # (auth error hints are also emitted by get_grok_bearer / refresh logic and --test-auth)


# Backwards-compatibility aliases (used by some older wiring / tests)
_call_grok_with_tools = call_grok_for_groksito
_call_grok_responses_api = call_grok_for_groksito

# Public alias for call_grok_with_tools (kept for compatibility)
# This allows existing call sites (e.g. conversation.py) to continue working without immediate breakage.
call_grok_with_tools = call_grok_for_groksito
