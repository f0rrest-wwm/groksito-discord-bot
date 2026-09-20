"""
Discord Client + Connection Ownership for Groksito (Standalone Conversational Bot)

This module is the sole owner of the persistent Discord Gateway WebSocket
connection for the conversational @Groksito experience.

Key responsibilities:
- Singleton Discord client + Gateway connection
- Guild whitelist enforcement (early security gate)
- Per-user rate limiting (6 requests / 60s)
- Thin on_message orchestration (activation, context update, then delegate)
- Slash command registration
- Wiring to conversation.py + LLM stack (no custom memory; no automatic injection)
- Liveness heartbeats for the independent web dashboard

Important invariants (do not break):
- This process is the *only* owner of the Discord Gateway for conversation.
- Guild whitelist checked in both on_message and every slash command.
- Rate limit check happens *before* invoking the LLM path.
- Context (short-term channel history) is always updated for *every* message.
- Activation: @mentions, bare name (meepo/groksito), or direct replies to the bot.
- Direct media delivery uses the DIRECT_DELIVERY_PERFORMED sentinel
  (cooperates with media/delivery.py + llm/client.py for exactly one reply).
- Background heartbeat task keeps the web UI informed of connection status.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from typing import Any, Deque, Optional

from ..utils.correlation import (
    cid_prefix,
    generate_correlation_id,
    set_correlation_id,
)
from ..utils.errors import log_auxiliary_failure

import discord

# Suppress voice-related warnings (voice is intentionally unsupported).
try:
    from discord.voice_client import VoiceClient as _DiscordVoiceClient
    _DiscordVoiceClient.warn_nacl = False
    _DiscordVoiceClient.warn_dave = False
except Exception:
    pass

from ..config import settings
from ..core.safety import safe_reply as _safe_reply

# Steam + Twitch integrations (extracted for client hygiene).
# Data fetching and game resolution live in discord/integrations/.
from .integrations import gamemeca, steam, thelog, twitch

from ..utils.text import extract_urls_from_text


async def _periodic_gamemeca_ranking_update(gamemeca_module):
    """Background job: refresh Gamemeca ranking JSON ~daily.
    The page is updated weekly (results reflected next week per site notice).
    Daily check is safe, cheap, and prevents hitting the site on every user /korea50.
    Command itself reads from the persisted JSON in data/gamemeca_ranking.json .
    """
    # Run once soon after bot start
    try:
        await gamemeca_module.refresh_ranking()
    except Exception as e:
        logger.debug(f"[Gamemeca] initial refresh failed (non-fatal): {e}")

    while True:
        await asyncio.sleep(24 * 3600)  # check daily
        try:
            await gamemeca_module.refresh_ranking()
            logger.info("[Gamemeca] ranking JSON refreshed via background job")
        except Exception as e:
            logger.warning(f"[Gamemeca] background refresh failed: {e}")


# Dedicated /audio slash (reuses 100% of audio_handler.py for TTS + fancy voice delivery
# via the image_delivery direct-delivery tracker; no duplication of generation or bubble logic).
from ..media.delivery import register_image_request
from ..media.audio_handler import (
    AUDIO_WRAPPING_TAGS,
    _tool_generate_audio,
    apply_wrapping_speech_tag,
    build_audio_speech_tags_embed,
    prepare_text_from_interaction,
)

logger = logging.getLogger("groksito.client")


# =============================================================================
# Guild Whitelist Security
# =============================================================================
_ALLOWED_GUILD_IDS: set[int] = set(settings.allowed_guild_ids)


def is_guild_allowed(guild_id: int | None) -> bool:
    if not _ALLOWED_GUILD_IDS:
        return True
    if guild_id is None:
        return False
    return guild_id in _ALLOWED_GUILD_IDS


# =============================================================================
# Global State
# =============================================================================
_discord_client: "discord.Client | None" = None
_discord_ready = asyncio.Event()
_discord_task: asyncio.Task | None = None

rate_limiter: Any = None
tree: Any = None


# =============================================================================
# Rate Limiter
# =============================================================================
# Simple per-user sliding window rate limiter (6 requests per 60 seconds).
# Enforced in on_message (before LLM invocation) and in /mislimites.
# This is a basic defense against abuse; the actual heavy lifting for
# conversational rate limiting and cost control lives in the LLM/tool layer.
class RateLimiter:
    def __init__(self, max_requests: int = 6, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.records: dict[int, Deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> tuple[bool, int]:
        now = time.time()
        user_records = self.records[user_id]
        while user_records and now - user_records[0] > self.window:
            user_records.popleft()
        used = len(user_records)
        if used >= self.max_requests:
            return False, 0
        user_records.append(now)
        return True, self.max_requests - used

    def get_remaining(self, user_id: int) -> int:
        now = time.time()
        user_records = self.records[user_id]
        while user_records and now - user_records[0] > self.window:
            user_records.popleft()
        return max(0, self.max_requests - len(user_records))


# =============================================================================
# Versus embed builder (/versus)
# =============================================================================
_VERSUS_COLORS = (0x3498DB, 0xE74C3C)  # blue vs red
_VERSUS_EMOJIS = ("🔵", "🔴")


def _format_metric(value: int | None, *, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    return f"**{value:,}**{suffix}"


def _build_versus_embeds(
    game1_name: str,
    game2_name: str,
    steam_games: list[dict[str, Any]],
    twitch_games: list[dict[str, Any]],
) -> list[discord.Embed]:
    """Build header + two side-by-side-style game embeds for /versus."""
    steam_by_original = {g["original_name"].lower(): g for g in steam_games}
    twitch_by_original = {g["original_name"].lower(): g for g in twitch_games}

    pairs: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []
    for name in (game1_name, game2_name):
        key = name.lower()
        pairs.append((name, steam_by_original.get(key), twitch_by_original.get(key)))

    header = discord.Embed(
        title="⚔️ Versus",
        description=f"**{game1_name}** vs **{game2_name}**",
        color=0x9B59B6,
    )
    header.set_footer(text="Datos en vivo de Steam y Twitch")

    embeds: list[discord.Embed] = [header]

    for idx, (original, steam_data, twitch_data) in enumerate(pairs):
        color = _VERSUS_COLORS[idx]
        emoji = _VERSUS_EMOJIS[idx]
        display_name = (
            (steam_data or {}).get("matched_name")
            or (twitch_data or {}).get("matched_name")
            or original
        )

        steam_found = steam_data is not None
        twitch_found = bool(twitch_data and twitch_data.get("found"))

        if not steam_found and not twitch_found:
            embeds.append(
                discord.Embed(
                    title=f"{emoji} {display_name}",
                    description="This game was not found on Steam or Twitch.",
                    color=color,
                )
            )
            continue

        embed = discord.Embed(
            title=f"{emoji} {display_name}",
            color=color,
            url=(
                f"https://store.steampowered.com/app/{steam_data['appid']}/"
                if steam_data and steam_data.get("appid")
                else None
            ),
        )

        if steam_data:
            pc = steam_data.get("player_count")
            steam_line = _format_metric(pc, suffix=" players on Steam")
            if steam_data.get("player_count_source") == "demo" and "demo" not in display_name.lower():
                steam_line += " (via Demo)"
            embed.add_field(name="🎮 Steam", value=steam_line, inline=False)
        else:
            embed.add_field(
                name="🎮 Steam",
                value="Not found on Steam",
                inline=False,
            )

        if twitch_data and twitch_data.get("configured"):
            if twitch_found:
                viewers = twitch_data.get("viewer_count")
                streams = twitch_data.get("live_streams")
                twitch_line = _format_metric(viewers, suffix=" viewers on Twitch")
                if isinstance(streams, int):
                    twitch_line += f"\n{streams:,} live streams"
                embed.add_field(name="📺 Twitch", value=twitch_line, inline=False)
            else:
                embed.add_field(
                    name="📺 Twitch",
                    value="Category not found on Twitch",
                    inline=False,
                )
        elif twitch_data and not twitch_data.get("configured"):
            embed.add_field(
                name="📺 Twitch",
                value="Twitch not configured (TWITCH_CLIENT_ID/SECRET)",
                inline=False,
            )

        thumb = None
        if steam_data and steam_data.get("image_url"):
            thumb = steam_data["image_url"]
        elif twitch_data and twitch_data.get("image_url"):
            thumb = twitch_data["image_url"]
        if thumb:
            embed.set_thumbnail(url=thumb)

        embeds.append(embed)

    # Winner callouts when both sides have comparable metrics
    steam_counts = [
        (pairs[i][0], (pairs[i][1] or {}).get("player_count"))
        for i in range(2)
        if pairs[i][1] and pairs[i][1].get("player_count") is not None
    ]
    if len(steam_counts) == 2:
        if steam_counts[0][1] > steam_counts[1][1]:
            header.add_field(
                name="🏆 Steam",
                value=f"**{steam_counts[0][0]}** leads in players",
                inline=True,
            )
        elif steam_counts[1][1] > steam_counts[0][1]:
            header.add_field(
                name="🏆 Steam",
                value=f"**{steam_counts[1][0]}** leads in players",
                inline=True,
            )
        else:
            header.add_field(name="🏆 Steam", value="Tie!", inline=True)

    twitch_counts = [
        (pairs[i][0], (pairs[i][2] or {}).get("viewer_count"))
        for i in range(2)
        if pairs[i][2] and pairs[i][2].get("found") and pairs[i][2].get("viewer_count") is not None
    ]
    if len(twitch_counts) == 2:
        if twitch_counts[0][1] > twitch_counts[1][1]:
            header.add_field(
                name="🏆 Twitch",
                value=f"**{twitch_counts[0][0]}** leads in viewers",
                inline=True,
            )
        elif twitch_counts[1][1] > twitch_counts[0][1]:
            header.add_field(
                name="🏆 Twitch",
                value=f"**{twitch_counts[1][0]}** leads in viewers",
                inline=True,
            )
        else:
            header.add_field(name="🏆 Twitch", value="Tie!", inline=True)

    return embeds


# =============================================================================
# Steam embed builder (shared by /steamchart, /stmchr, /topgames)
# =============================================================================
def _build_steam_game_embeds(games: list[dict[str, Any]]) -> list[discord.Embed]:
    """Build one Discord embed per game from ``get_steam_game_data()`` results."""
    embeds: list[discord.Embed] = []
    for g in games:
        name = g["matched_name"]
        appid = g["appid"]
        player_count = g.get("player_count")
        image_url = g.get("image_url")
        color = steam.get_game_color(name)
        if player_count is not None:
            description = f"**{player_count:,}** players now"
            if g.get("player_count_source") == "demo" and "demo" not in name.lower():
                description += " (via Steam Demo)"
        else:
            description = "Player count unavailable on Steam Charts right now."
        embed = discord.Embed(
            title=name,
            description=description,
            color=color,
            url=f"https://store.steampowered.com/app/{appid}/",
        )
        thumb_url = image_url or f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg"
        embed.set_thumbnail(url=thumb_url)
        embeds.append(embed)
    return embeds


def _build_topkorea_embed(ranking: list[dict[str, Any]]) -> discord.Embed:
    """Build a single compact embed for the TheLog top 10 ranking (전체).

    Styling:
    - #1 uses 🥇 medal + bold (to stand out as the top / "golden")
    - Rank changes: 🟢▲ for up (green), 🔴▼ for down (red), ⚪= for same
      (the arrow + number get the color via emoji)

    All displayed text is in English per user request.
    Uses english_name / english_publisher when available.
    Added 2026-06-22 for /topkorea.
    """
    lines: list[str] = []
    for g in ranking:
        ch = thelog.format_rank_change(g.get("change", 0))
        display_name = g.get("english_name") or g.get("name", "")
        display_pub = g.get("english_publisher") or g.get("publisher", "")

        if g["rank"] == 1:
            # Special standout for #1 (golden / top highlight)
            # Using medal + bold to simulate "golden letters" effect in Discord
            line = f"🥇 **{display_name}** — **{g['shares']:.2f}%** ({display_pub}) {ch}"
        else:
            line = f"{g['rank']}. **{display_name}** — **{g['shares']:.2f}%** ({display_pub}) {ch}"

        lines.append(line)

    description = "\n".join(lines) if lines else "No data available."

    embed = discord.Embed(
        title="🎮 Top 10 PC Bang Game Rankings (Overall)",
        description=description,
        color=0x00A8E8,
        url="https://www.thelog.co.kr/index.do",
    )
    # Date from API payload (targetDate is YYYYMMDD)
    target = None
    if ranking:
        raw0 = ranking[0].get("raw", {})
        td = raw0.get("targetDate")
        if isinstance(td, str) and len(td) == 8:
            target = f"{td[:4]}.{td[4:6]}.{td[6:]}"
    footer = "Source: thelog.co.kr • Real collected data"
    if target:
        footer += f" • {target}"
    embed.set_footer(text=footer)
    return embed


def _build_korea50_embed(ranking: list[dict[str, Any]]) -> discord.Embed:
    """Build a single compact embed for the Gamemeca weekly top 50 popularity ranking.

    Everything translated to English where possible (game names via dictionary).
    Shows genre + model (e.g. AOS / Partial Payment), publisher (English), and change for all 1-50.
    """
    # Small translations for genre/model so no Korean appears in the list
    GENRE_TRANSLATIONS = {
        "스포츠": "Sports",
        "기타": "Other",
        "어드벤쳐": "Adventure",
        "액션 RPG": "Action RPG",
        "롤플레잉": "Role-Playing",
        "슈팅": "Shooter",
    }
    MODEL_TRANSLATIONS = {
        "부분유료화": "Partial Payment",
        "정액제": "Subscription",
        "개발중": "In Development",
        "유료화": "Paid",
    }

    lines: list[str] = []
    for g in ranking:
        display_name = g.get("english_name") or g.get("name", "")
        display_pub = g.get("english_publisher") or g.get("publisher", "")
        genre = g.get("genre", "")
        model = g.get("model", "")

        # Translate genre/model if we have English version
        disp_genre = GENRE_TRANSLATIONS.get(genre, genre)
        disp_model = MODEL_TRANSLATIONS.get(model, model)

        ch = g.get("change", 0)
        ch_str = gamemeca.format_rank_change(ch)  # always include for 1-50

        extra = ""
        if disp_genre or disp_model:
            extra = f" — {disp_genre} / {disp_model}".strip() if disp_genre and disp_model else f" — {disp_genre or disp_model}".strip()

        if g["rank"] == 1:
            line = f"🥇 **{display_name}**{extra} ({display_pub}) {ch_str}".strip()
        else:
            line = f"{g['rank']}. **{display_name}**{extra} ({display_pub}) {ch_str}".strip()
        lines.append(line)

    description = "\n".join(lines) if lines else "No data available."

    embed = discord.Embed(
        title="🎮 Gamemeca Weekly Popularity Ranking (Top 50)",
        description=description,
        color=0x00A8E8,
        url="https://www.gamemeca.com/ranking.php",
    )
    week = None
    # week is stored in the module cache after fetch
    try:
        week = gamemeca._game_rank_cache.get("week")
    except Exception:
        pass
    footer = "Source: gamemeca.com • Weekly ranking"
    if week:
        footer += f" • {week}"
    embed.set_footer(text=footer)
    return embed


# =============================================================================
# Slash Command Registration
# =============================================================================
def register_slash_commands(
    tree: "discord.app_commands.CommandTree", client: "discord.Client"
) -> None:
    """Register Groksito slash commands. Steam commands delegate to discord/integrations/steam.py."""
    # /mislimites ΓÇö shows remaining requests for the current user (rate limit info)
    @tree.command(
        name="mislimites", description="Show how many requests you have left this minute"
    )
    async def mislimites(interaction: discord.Interaction):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        user_id = interaction.user.id
        rl = getattr(client, "rate_limiter", rate_limiter)
        remaining = rl.get_remaining(user_id)
        await interaction.response.send_message(
            f"**{interaction.user.display_name}**, you have **{remaining}/6** requests left this minute.",
            ephemeral=True,
        )

    # /steamchart ΓÇö optional juegos (comma-separated). Falls back to a sensible default list.
    # Now renders exactly like /stmchr: one rich embed per game (name + current players,
    # game-themed color when known, clickable title to Steam store, and thumbnail image).
    # Thumbnails use the same robust resolver as /stmchr (multiple CDNs + store scrape fallback).
    # If no thumbnail resolves, falls back to the standard Steam header.jpg (same as /stmchr).
    @tree.command(
        name="steamchart",
        description="Live Steam player counts. Ex: /steamchart dota 2, path of exile 2",
    )
    async def steamchart(interaction: discord.Interaction, juegos: Optional[str] = None):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        if not juegos or not juegos.strip():
            juegos = "path of exile 2, black desert, crimson desert, lost ark"

        games_data = await steam.get_steam_game_data(juegos, max_games=8)

        if not games_data:
            await interaction.followup.send(
                "Could not match that name on Steam. "
                "Use the store name (e.g. 'dota 2', 'counter-strike 2') "
                "or run /stmchr for the fixed list."
            )
            return

        await interaction.followup.send(embeds=_build_steam_game_embeds(games_data))

    # /stmchr ΓÇö always shows the fixed list of 8 games (no parameters needed).
    # One embed per game, sorted by current players (highest first).
    # Thumbnails use a robust resolver (multiple CDN patterns + store page fallback)
    # so games with only hashed asset paths (WWM, TBH, etc.) still get images.
    # Each embed gets a game-specific color and the title links to the Steam store page.
    @tree.command(
        name="stmchr", description="Fixed list: Black Desert, PoE2, GW2, Lost Ark, and more"
    )
    async def stmchr(interaction: discord.Interaction):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        games_data = await steam.get_steam_game_data(
            steam.stmchr_game_names_csv(),
            preresolved=steam.stmchr_preresolved_map(),
            max_games=len(steam._STMCHR_GAMES),
        )

        if games_data:
            await interaction.followup.send(embeds=_build_steam_game_embeds(games_data))
        else:
            await interaction.followup.send("Could not fetch Steam Charts data right now.")

    # /topgames ΓÇö shows the real-time Top 10 (or so) games by current players
    # directly from https://steamcharts.com/top (not a fixed list like /stmchr).
    # Renders exactly like /stmchr and the resolved /steamchart: one rich embed
    # per game with current players, thumbnail (robust resolver), store link,
    # and themed color when the game is one of the known curated titles.
    # Fetches fresh current counts via the official Steam API for consistency.
    @tree.command(
        name="topgames",
        description="Top 10 games by current Steam players (steamcharts.com/top)"
    )
    async def topgames(interaction: discord.Interaction):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        top_list = await steam.get_top_steam_games(10)
        if not top_list:
            await interaction.followup.send("Could not fetch the top games list right now.")
            return

        names_csv = ", ".join(name for name, _ in top_list)
        preresolved = {name: appid for name, appid in top_list}

        games_data = await steam.get_steam_game_data(
            names_csv,
            preresolved=preresolved,
            max_games=len(top_list),
        )

        if not games_data:
            await interaction.followup.send("Could not fetch Steam Charts data right now.")
            return

        await interaction.followup.send(embeds=_build_steam_game_embeds(games_data))

    # /topkorea — live top 10 from TheLog (게임순위 전체) for Korean PC bangs.
    # All visible text (title, labels, game/publisher names) is shown in English.
    # Uses clean JSON endpoint. Added 2026-06-22.
    @tree.command(
        name="topkorea",
        description="Top 10 PC Bang game rankings in Korea (TheLog overall)"
    )
    async def topkorea(interaction: discord.Interaction):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Groksito is not available in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        ranking = await thelog.get_top_korea_rank(10)
        if not ranking:
            await interaction.followup.send(
                "Could not fetch the TheLog ranking right now."
            )
            return

        embed = _build_topkorea_embed(ranking)
        await interaction.followup.send(embed=embed)

    # /korea50 — weekly top 50 from Gamemeca 인기 게임 순위 (main ranking page).
    # Uses HTML table parsing (no clean JSON feed available, see plan).
    # English display consistent with /topkorea. Added 2026-06-22.
    @tree.command(
        name="korea50",
        description="Top 50 weekly game popularity from Gamemeca"
    )
    async def korea50(interaction: discord.Interaction):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Groksito is not available in this server.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        ranking = await gamemeca.get_korea_weekly_ranking(50)
        if not ranking:
            await interaction.followup.send(
                "Could not fetch the Gamemeca ranking right now."
            )
            return

        embed = _build_korea50_embed(ranking)
        await interaction.followup.send(embed=embed)

    # /versus — compare two games with live Steam player counts + Twitch viewers.
    @tree.command(
        name="versus",
        description="Compare two games: Steam players and Twitch viewers",
    )
    async def versus(
        interaction: discord.Interaction,
        juego1: str,
        juego2: str,
    ):
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        g1 = (juego1 or "").strip()
        g2 = (juego2 or "").strip()
        if not g1 or not g2:
            await interaction.response.send_message(
                "Give two games to compare. Example: `/versus dota 2 counter-strike 2`",
                ephemeral=True,
            )
            return

        if g1.lower() == g2.lower():
            await interaction.response.send_message(
                "Pick two different games for versus.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=False)

        lookups = [
            (g1, steam.normalize_game_name_for_lookup(g1) or g1),
            (g2, steam.normalize_game_name_for_lookup(g2) or g2),
        ]
        lookup_csv = ", ".join(lookup for _, lookup in lookups)

        steam_task = steam.get_steam_game_data(lookup_csv, max_games=2)
        twitch_task = twitch.get_twitch_game_data_batch([lookup for _, lookup in lookups])
        steam_games, twitch_games = await asyncio.gather(steam_task, twitch_task)

        for i, game in enumerate(steam_games):
            if i < len(lookups):
                game["original_name"] = lookups[i][0]
        for i, game in enumerate(twitch_games):
            if i < len(lookups):
                game["original_name"] = lookups[i][0]

        if not steam_games and not any(t.get("found") for t in twitch_games):
            await interaction.followup.send(
                "Could not recognize either game on Steam or Twitch. "
                "Use store/Twitch names (e.g. 'dota 2', 'counter-strike 2')."
            )
            return

        embeds = _build_versus_embeds(g1, g2, steam_games, twitch_games)
        await interaction.followup.send(embeds=embeds)

    # /audio — dedicated TTS/voice slash command.
    # - text is optional (for reply-to use case).
    # - voice uses app_commands.choices (eve/ara/rex/sal/leo) with friendly names.
    # - When invoked as reply to another message (no or with text): reads/combines using helper.
    # - Uses guild whitelist + rate limiter (counts as a user request).
    # - Defers ephemerally for "generating" UX.
    # - Registers for direct delivery + calls core _tool_generate_audio (reuses EVERYTHING:
    #   text prep, xAI call, pydub transcode, real waveform, voice flag bubble, context log).
    # - Ephemeral confirmation after; the voice bubble itself is delivered publicly in channel.
    @tree.command(
        name="audio",
        description="Generate a TTS voice message. Optional: reply to a message.",
    )
    @discord.app_commands.describe(
        text="Text to speak. Inline: [pause], [laugh], [sigh], [breath], [chuckle], [long-pause].",
        voice="Grok TTS voice (eve recommended).",
        estilo="Optional delivery style: whisper, soft, slow, loud, emphasis, singing.",
    )
    @discord.app_commands.choices(
        voice=[
            discord.app_commands.Choice(name="Eve (energetic, recommended)", value="eve"),
            discord.app_commands.Choice(name="Ara (warm)", value="ara"),
            discord.app_commands.Choice(name="Rex (professional)", value="rex"),
            discord.app_commands.Choice(name="Sal (balanced)", value="sal"),
            discord.app_commands.Choice(name="Leo (authoritative)", value="leo"),
        ],
        estilo=[
            discord.app_commands.Choice(name=label, value=tag)
            for label, tag in AUDIO_WRAPPING_TAGS
        ],
    )
    async def audio_slash(
        interaction: discord.Interaction,
        text: Optional[str] = None,
        voice: Optional[discord.app_commands.Choice[str]] = None,
        estilo: Optional[discord.app_commands.Choice[str]] = None,
    ):
        # Guild whitelist (same as every other slash command)
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        # Rate limit (audio gen consumes resources like a conversational request)
        rl = getattr(client, "rate_limiter", rate_limiter)
        can_use, _ = rl.check(interaction.user.id)
        if not can_use:
            await interaction.response.send_message(
                "Slow down — you already used your 6 requests this minute.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        # Build text (reply support if user replied to a msg then ran /audio)
        provided = (text or "").strip()
        final_text = await prepare_text_from_interaction(interaction, provided)

        if not final_text:
            await interaction.followup.send(
                embed=build_audio_speech_tags_embed(),
                ephemeral=True,
            )
            return

        selected_style = estilo.value if estilo else None
        final_text = apply_wrapping_speech_tag(final_text, selected_style)

        # Resolve voice (user choice or config default) + language
        selected_voice = voice.value if voice else getattr(settings, "tts_default_voice", "eve") or "eve"
        selected_lang = getattr(settings, "tts_default_language", "es") or "es"

        # Register using the Interaction as original_message (delivery code now supports it for public channel send)
        request_id = None
        try:
            ch = getattr(interaction, "channel", None)
            request_id = await register_image_request(
                user_id=interaction.user.id,
                channel_id=getattr(ch, "id", 0) or 0,
                message_id=getattr(interaction, "id", 0),
                operation_type="audio",
                original_message=interaction,
            )
        except Exception as reg_err:
            logger.warning(f"[AudioSlash] Failed to register audio request: {reg_err}")

        # Call core tool directly (it will consume the request and do direct delivery of voice bubble if possible)
        result = await _tool_generate_audio(
            text=final_text,
            voice=selected_voice,
            language=selected_lang,
            request_id=request_id,
        )

        # Ephemeral confirmation to the user who ran the command (the audio itself is public via direct delivery)
        if result and "SUCCESS" in result:
            style_note = f" · estilo **{selected_style}**" if selected_style else ""
            await interaction.followup.send(
                f"✅ Audio generado con la voz **{selected_voice}**{style_note} y enviado al canal.",
                ephemeral=True,
            )
        else:
            # Surface the (Spanish) error or note from the handler
            await interaction.followup.send(
                result or "No se pudo generar el audio.",
                ephemeral=True,
            )

    # =============================================================================
    # Message Context Menu: "🔊 Leer en voz alta" (Read Aloud)
    # =============================================================================
    # Right-click any message > Apps > "🔊 Leer en voz alta".
    # Quick way to TTS the message content using the configured default voice/language.
    # Reuses the exact same audio pipeline as /audio (text prep, xAI TTS, waveform bubble,
    # direct delivery via image_delivery sentinel, rate limiting, guild whitelist).
    # This restores the dedicated "context menu read aloud" UX (no reply + slash needed).
    @tree.context_menu(name="Read aloud")
    async def read_aloud_context(
        interaction: discord.Interaction,
        message: discord.Message,
    ):
        # Guild whitelist (identical to every other command)
        if interaction.guild and not is_guild_allowed(interaction.guild.id):
            await interaction.response.send_message(
                "Meepo is not available in this server.", ephemeral=True
            )
            return

        # Rate limit (audio is a resource-consuming action, same bucket as chat requests)
        rl = getattr(client, "rate_limiter", rate_limiter)
        can_use, _ = rl.check(interaction.user.id)
        if not can_use:
            await interaction.response.send_message(
                "Slow down — you already used your 6 requests this minute.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        # Extract text directly from the right-clicked target message.
        # (For pure image/attachment messages without text we gracefully error;
        # the conversational LLM + vision path can describe images instead.)
        final_text = (getattr(message, "content", "") or "").strip()
        if not final_text:
            await interaction.followup.send(
                "El mensaje no contiene texto para leer en voz alta. "
                "Responde al mensaje con /audio o menciona a Groksito para analizar imágenes.",
                ephemeral=True,
            )
            return

        # Always use the configured defaults (context menus don't support parameter choices
        # in the initial click; user can follow up with /audio + voice if they want a different one).
        selected_voice = getattr(settings, "tts_default_voice", "eve") or "eve"
        selected_lang = getattr(settings, "tts_default_language", "es") or "es"

        # Register for direct delivery (same as /audio slash so the voice bubble is sent
        # publicly by the tool without a duplicate reply from the command handler).
        request_id = None
        try:
            ch = getattr(interaction, "channel", None)
            request_id = await register_image_request(
                user_id=interaction.user.id,
                channel_id=getattr(ch, "id", 0) or 0,
                message_id=getattr(message, "id", 0) or getattr(interaction, "id", 0),
                operation_type="audio",
                original_message=message,  # target message gives correct channel for delivery
            )
        except Exception as reg_err:
            logger.warning(f"[ReadAloudContext] Failed to register audio request: {reg_err}")

        # Core generation + delivery (identical reuse as the slash and the generate_audio tool)
        result = await _tool_generate_audio(
            text=final_text,
            voice=selected_voice,
            language=selected_lang,
            request_id=request_id,
        )

        # Ephemeral confirmation to the invoker (the actual audio is delivered publicly
        # to the channel by the shared audio handler / direct-delivery logic).
        if result and "SUCCESS" in str(result).upper():
            await interaction.followup.send(
                f"✅ Audio generado con la voz **{selected_voice}** y enviado al canal.",
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                result or "No se pudo generar el audio.",
                ephemeral=True,
            )


# =============================================================================
# Main Connection Function (Conversational Only)
# =============================================================================
async def ensure_discord_connected(conversational: bool = True) -> "discord.Client":
    """
    Ensures the Discord client is connected.

    In the standalone Groksito bot, we always run with conversational=True.
    This function owns the persistent Gateway WebSocket.
    """
    global _discord_client, _discord_task, rate_limiter, tree

    if _discord_client is not None:
        await _discord_ready.wait()
        return _discord_client

    if not settings.discord_bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is not configured in .env")

    intents = discord.Intents.default()
    intents.guilds = True
    intents.members = True
    intents.message_content = True  # Required for conversational bot

    _discord_client = discord.Client(intents=intents)

    logger.info("=== GROKSITO DISCORD BOT (STANDALONE) ===")
    logger.info("CONVERSATIONAL OWNER: This process owns the persistent Gateway connection.")
    logger.info("Full @Groksito experience enabled (native vision via Responses API, channel context, tools, image/video gen).")

    rate_limiter = RateLimiter(max_requests=6, window_seconds=60)
    tree = discord.app_commands.CommandTree(_discord_client)

    _discord_client.rate_limiter = rate_limiter
    _discord_client.command_tree = tree

    # Register slash commands.
    # This call must happen after the client and rate_limiter are attached.
    # All three commands (/mislimites, /steamchart, /stmchr) are now defined
    # in register_slash_commands above.
    # register_slash_commands(tree, _discord_client)

    # Lazy import of conversational stack (keeps things clean)
    from .. import context
    # No custom memory system at all (removed for 100% Grok nativeness)
    from ..core.conversation import (
        _resolve_referenced_and_activation,
        _build_referenced_context,
        _invoke_groksito,
    )

# on_ready
    @_discord_client.event
    async def on_ready():
        logger.info(f"Γ£à Groksito connected as {_discord_client.user} (ID: {_discord_client.user.id})")
        logger.info(f"[Discord] discord.py version: {discord.__version__} (target: >=2.7.0,<3.0 for modern voice + features)")

        if _ALLOWED_GUILD_IDS:
            logger.info(f"[SECURITY] Guild whitelist ACTIVE ΓÇö {len(_ALLOWED_GUILD_IDS)} allowed guild(s)")
        else:
            logger.warning("[SECURITY] No ALLOWED_GUILD_IDS set ΓÇö bot will respond in ANY server.")

        try:
            await _discord_client.change_presence(activity=discord.Game(name="yapping"))
            await tree.sync()
            logger.info("Γ£à Slash commands synchronized")
        except Exception as e:
            logger.error(f"Error syncing slash commands: {e}")

        _discord_ready.set()

        # Emoji / custom emote discovery (metadata only on startup).
        # Vision descriptions + popularity ranking are done *lazily* only for emotes that actually get used
        # in messages the bot sees. This is the efficient path for servers with 100-200+ emotes.
        # Data lives in data/emoji_knowledge.json.
        try:
            from ..utils import emoji_registry
            asyncio.create_task(emoji_registry.scan_all_accessible_emojis(_discord_client))
            logger.info("[Emoji] Background emote metadata scan launched (vision + usage ranking is lazy on real use)")
        except Exception as emoji_err:
            logger.debug(f"[Emoji] Could not start emote scan (non-fatal): {emoji_err}")

        try:
            from .integrations import steam as steam_integration
            asyncio.create_task(steam_integration.warmup_steam_app_list())
            logger.info("[Steam] Background app list cache warmup launched")
        except Exception as steam_err:
            logger.debug(f"[Steam] Could not start app list warmup (non-fatal): {steam_err}")

        # Gamemeca weekly ranking JSON updater (avoids live scrape on every /korea50)
        try:
            from .integrations import gamemeca as gamemeca_integration
            asyncio.create_task(_periodic_gamemeca_ranking_update(gamemeca_integration))
            logger.info("[Gamemeca] Background weekly ranking JSON updater launched (daily check)")
        except Exception as gm_err:
            logger.debug(f"[Gamemeca] Could not start ranking updater (non-fatal): {gm_err}")

        # Write initial heartbeat + supporting snapshots so the web dashboard has good data immediately.
        try:
            from ..core.health import (
                write_bot_heartbeat,
                write_bot_guilds_snapshot,
                write_bot_stats,
                write_bot_health_snapshot,
            )
            guilds_list = getattr(_discord_client, "guilds", []) or []
            guilds = len(guilds_list)
            lat = getattr(_discord_client, "latency", None)
            write_bot_heartbeat(
                connected=True,
                user=str(_discord_client.user),
                user_id=_discord_client.user.id if _discord_client.user else None,
                guilds=guilds,
                latency=lat if (lat is not None and lat > 0) else None,
            )
            write_bot_guilds_snapshot(guilds_list)
            write_bot_stats()
            write_bot_health_snapshot()
        except Exception as health_err:
            log_auxiliary_failure(
                logger,
                "initial health snapshot write",
                health_err,
                feature="Health",
            )

    # Extra lifecycle events for more accurate web dashboard status
    @_discord_client.event
    async def on_disconnect():
        try:
            from ..core.health import write_bot_heartbeat
            write_bot_heartbeat(connected=False)
        except Exception as health_err:
            log_auxiliary_failure(
                logger,
                "disconnect heartbeat write",
                health_err,
                feature="Health",
            )

    @_discord_client.event
    async def on_resumed():
        try:
            from ..core.health import (
                write_bot_heartbeat,
                write_bot_guilds_snapshot,
                write_bot_stats,
                write_bot_health_snapshot,
            )
            guilds_list = getattr(_discord_client, "guilds", []) or []
            guilds = len(guilds_list)
            lat = getattr(_discord_client, "latency", None)
            write_bot_heartbeat(
                connected=True,
                user=str(getattr(_discord_client, "user", None)),
                user_id=getattr(getattr(_discord_client, "user", None), "id", None),
                guilds=guilds,
                latency=lat if (lat is not None and lat > 0) else None,
            )
            write_bot_guilds_snapshot(guilds_list)
            write_bot_stats()
            write_bot_health_snapshot()
        except Exception as health_err:
            log_auxiliary_failure(
                logger,
                "resume health snapshot write",
                health_err,
                feature="Health",
            )

    @_discord_client.event
    async def on_guild_join(guild):
        # Ensure emotes for newly joined guild (for testing multi-server scenarios)
        # so the server-specific list is populated from live data immediately.
        try:
            from ..utils import emoji_registry
            asyncio.create_task(emoji_registry.ensure_guild_emojis_registered(guild))
            logger.info(f"[Emoji] on_guild_join: registered live emotes for guild {getattr(guild, 'id', '?')}")
        except Exception as emoji_join_err:
            logger.debug(f"[Emoji] on_guild_join ensure skipped (non-fatal): {emoji_join_err}")

    # on_message - thin orchestrator (most logic lives in conversation.py)
    #
    # Invariants maintained here:
    # - Bot's own messages are ignored immediately.
    # - Guild whitelist is enforced first (after correlation).
    # - Context is *always* updated for every incoming message (for optional
    #   recent context summaries and legacy tools).
    # - Rate limit is checked *before* any expensive work or LLM call.
    # - Activation decision is delegated to conversation._resolve_referenced_and_activation
    #   (the authoritative strict policy that prevents bot replies to random
    #   user-to-user conversations).
    # - The actual Grok call + tools + vision happens in _invoke_groksito.
    @_discord_client.event
    async def on_message(message: discord.Message):
        cid_p = ""  # default if we error very early
        try:
            if message.author.id == _discord_client.user.id:
                return

            author_display = getattr(message.author, "display_name", None) or getattr(message.author, "name", "Usuario")

            # Generate correlation ID for this message (for full-trace logging of the interaction).
            # Set early so activation/resolve/vision logs are associated with it.
            cid = generate_correlation_id()
            set_correlation_id(cid)
            cid_p = cid_prefix()  # e.g. "cid=abc12345 "

            # Guild whitelist guard
            if message.guild and not is_guild_allowed(message.guild.id):
                logger.info(f"{cid_p}[SECURITY] Ignoring message from unauthorized guild {message.guild.id}")
                return

            # Bootstrap live emotes for *this* server so top-used list and normalize are always current.
            try:
                from ..utils import emoji_registry
                asyncio.create_task(emoji_registry.ensure_guild_emojis_registered(message.guild))
            except Exception:
                pass

            # Learn which custom emotes are actually used in this server (efficient local tracking).
            # This lets us surface only the popular ones + do vision descriptions lazily instead of
            # processing every single one of the 100-200 emotes some servers have.
            try:
                from ..utils import emoji_registry
                emoji_registry.record_emojis_from_message(message)
            except Exception as emoji_track_err:
                logger.debug(f"{cid_p}[Emoji] record_emojis_from_message failed (non-fatal): {emoji_track_err}")

            # Always track context (for get_recent_context tool and optional summarization)
            # Also capture images and links so the on-demand recent context summarizer (used by tool)
            # can analyze images (vision) and do surface search on links.
            image_urls: list[str] = []
            links: list[str] = []
            try:
                # Direct attachments
                for att in getattr(message, "attachments", []) or []:
                    ct = getattr(att, "content_type", "") or ""
                    if "image" in ct.lower() and getattr(att, "url", None):
                        image_urls.append(att.url)
                # Embeds (thumbnails / images)
                for emb in getattr(message, "embeds", []) or []:
                    for key in ("image", "thumbnail"):
                        obj = getattr(emb, key, None)
                        if obj and getattr(obj, "url", None):
                            image_urls.append(obj.url)
                # Links / URLs from text content
                # Centralized URL extraction (utils/text.py).
                # duplication with conversation.py extractors. Behavior is identical.
                if message.content:
                    for clean in extract_urls_from_text(message.content):
                        if clean and clean not in links:
                            links.append(clean)
            except Exception as attach_err:
                logger.warning(f"{cid_p}[Message] attachment/link extraction failed (non-fatal): {attach_err}")

            context.update_from_message(
                channel_id=message.channel.id,
                user_id=message.author.id,
                author_name=author_display,
                content=message.content or "",
                is_bot=False,
                image_urls=image_urls,
                links=links,
            )

            # Activation decision
            # The resolve function now contains the authoritative strict logic (refined across iterations)
            # and emits clear per-decision logs. We still keep a defensive guard here.
            result = await _resolve_referenced_and_activation(
                message=message,
                client_user=_discord_client.user,
                author_display=author_display,
            )
            # result is now 6-tuple: ... , has_x_link_intent, has_image_creation_intent
            if len(result) >= 6:
                referenced, is_reply_to_bot, explicit_visual, is_reply_cont, has_x_link_intent, has_image_creation = result
            else:
                referenced, is_reply_to_bot, explicit_visual, is_reply_cont, has_x_link_intent = result if len(result) == 5 else (*result, False)
                has_image_creation = False

            # is_reply_to_bot + is_mentioned are passed down. Referenced context is injected for
            # direct replies to Groksito OR when the bot is @mentioned inside a reply to another user
            # (e.g. " @groksito describe the video in that link my friend just posted").

            is_mentioned = _discord_client.user in getattr(message, "mentions", [])
            raw_low = (message.content or "").lower()
            name_called = bool(re.search(r"(?<!\w)(meepo|groksito)(?!\w)", raw_low))
            if name_called:
                is_mentioned = True

            # === ACTIVATION GUARD ===
            # @mention, bare name (meepo / groksito), or direct reply to the bot.
            if not is_mentioned and not is_reply_to_bot:
                return

            # Rate limit
            rl = getattr(_discord_client, "rate_limiter", rate_limiter)
            can_use, _ = rl.check(message.author.id)
            if not can_use:
                await _safe_reply(message, "Slow down — you already used your 6 requests this minute.", mention_author=False)
                return

            # Rich context + meta detection
            # Note: referenced may have been fetched in resolve; fetch again only if missing
            if message.reference and message.reference.message_id and referenced is None:
                try:
                    referenced = await message.channel.fetch_message(message.reference.message_id)
                    logger.info(f"{cid_p}[Reply] Fetched referenced message in client fallback")
                except Exception as ref_fetch_err:
                    logger.warning(f"{cid_p}[Reply] Client fallback fetch for referenced message failed: {ref_fetch_err}")

            referenced_context = await _build_referenced_context(referenced) if referenced else None

            is_meta = False
            try:
                is_meta = context.is_conversation_meta_question(message.content or "")
            except Exception as meta_err:
                logger.debug(f"{cid_p}[Meta] conversation meta detection failed (non-fatal): {meta_err}")

            # NOTE: No custom memory / rich channel context computation here.
            # Only referenced message is passed; classification (is_meta) still used for logging/heuristics.
            # All (minimal) injection decided inside llm_input.build_responses_input ([R:] on bot replies + mention-in-reply cases).
            # Recent conversation context: on-demand via get_recent_context tool only (no pre-injection, #19).
            # No custom memory at all (removed for maximum nativeness).

            # Invoke Groksito (native context via llm_input, vision, tools)
            # cid is already set in contextvar for all downstream logging.
            async with message.channel.typing():
                await _invoke_groksito(
                    message=message,
                    referenced=referenced,
                    referenced_context=referenced_context,
                    author_display=author_display,
                    is_meta_convo=is_meta,
                    explicit_visual_reply_intent=explicit_visual,
                    is_reply_continuation=is_reply_cont,
                    has_x_link_intent=has_x_link_intent,  # X/link intent signal (affects native x_search offering + ref enrichment)
                    is_reply_to_bot=is_reply_to_bot,
                    has_image_creation_intent=has_image_creation,
                    is_mentioned=is_mentioned,
                )

        except Exception as e:
            logger.exception(f"{cid_p}Unhandled error in on_message: {e}")

    # Start the bot
    async def _runner():
        try:
            await _discord_client.start(settings.discord_bot_token)
        except Exception as exc:
            logger.error(f"Discord connection failed: {exc}", exc_info=True)
            _discord_ready.clear()

    _discord_task = asyncio.create_task(_runner())
    logger.info("Starting Groksito Discord connection (CONVERSATIONAL OWNER)...")

    try:
        await asyncio.wait_for(_discord_ready.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Timeout waiting for Discord connection. Check token and network.")

    # -------------------------------------------------------------------------
    # Background heartbeat task (lets the separate web dashboard know we're alive)
    # Writes every ~35s so the web can show a green "Connected" indicator + basic stats.
    # -------------------------------------------------------------------------
    async def _heartbeat_updater() -> None:
        while True:
            try:
                await asyncio.sleep(35)
                if _discord_client and getattr(_discord_client, "is_ready", lambda: False)():
                    try:
                        from ..core.health import (
                            write_bot_heartbeat,
                            write_bot_guilds_snapshot,
                            write_bot_stats,
                            write_bot_health_snapshot,
                        )
                        guilds_list = getattr(_discord_client, "guilds", []) or []
                        guilds = len(guilds_list)
                        lat = getattr(_discord_client, "latency", None)
                        write_bot_heartbeat(
                            connected=True,
                            user=str(getattr(_discord_client, "user", None)),
                            user_id=getattr(getattr(_discord_client, "user", None), "id", None),
                            guilds=guilds,
                            latency=lat if (lat is not None and lat > 0) else None,
                        )
                        write_bot_guilds_snapshot(guilds_list)
                        write_bot_stats()
                        write_bot_health_snapshot()
                    except Exception as health_err:
                        log_auxiliary_failure(
                            logger,
                            "periodic health snapshot write",
                            health_err,
                            feature="Health",
                            level=logging.DEBUG,
                        )
            except asyncio.CancelledError:
                break
            except Exception as heartbeat_err:
                # Never let the heartbeat task kill the bot
                log_auxiliary_failure(
                    logger,
                    "heartbeat updater tick",
                    heartbeat_err,
                    feature="Health",
                )
                await asyncio.sleep(10)

    asyncio.create_task(_heartbeat_updater())

    return _discord_client


__all__ = ["ensure_discord_connected", "is_guild_allowed", "rate_limiter"]
