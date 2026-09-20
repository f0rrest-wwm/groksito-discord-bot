# Meepo

Discord bot for the **Everest** guild (Where Winds Meet).

Fork of [Groksito](https://github.com/lupintic/groksito-discord-bot). The live bot is named **Meepo**. Mention `@Meepo` in Discord.

Hosted on Railway with SuperGrok OAuth. Data and OAuth tokens live on a volume (`/app/data`, persist path for tokens).

## What it does

- Chat when mentioned or replied to. Matches language (including Nepali / romanized Nepali).
- Images: `generate_image` / `edit_image` (up to **3** reference photos on edit).
- Video: text-to-video or **attach one still** and ask to animate (`generate_video`).
- TTS audio.
- Web search, vision on attachments.
- No slash-command roster required for the core chat/media path.

## House rules (this fork)

- Identity is **Meepo**. Do not name the underlying model in chat.
- **No NSFW** images or videos. Reply: `No NSFW allowed.`
- Creator (unlimited video, no roasts): Discord user id `253869773421674498` (`f0rest` / `forest`).
- Guild is WWM / Everest — do not dunk on the game or the guild.
- Video: **5 clips per user per UTC day**. Creator uncapped. Only **one video at a time** for the whole bot.
- Delivered files are named `meepo_image.png` / `meepo_video.mp4`.

## Talk to it

```
@Meepo roast this take
@Meepo generate_image 9:16 Skyward Bond poster
@Meepo edit_image  (attach up to 3 refs)
@Meepo generate_video 720p 6 seconds  (attach 1 image to animate)
```

Image-to-video uses the first attachment only.

## Run (Railway)

This fork is deployed as a Railway service from GitHub.

Required / useful variables:

```
DISCORD_BOT_TOKEN=
GROK_AUTH_MODE=oauth
ENABLE_VIDEO_GENERATION=true
GROKSITO_DATA_DIR=/app/data
GROKSITO_OWNER_ID=253869773421674498
VIDEO_UNLIMITED_USER_IDS=253869773421674498
VIDEO_DAILY_LIMIT=5
```

OAuth tokens must sit on the **volume**, not the ephemeral container disk, or they vanish on redeploy.

Video usage log:

```
/app/data/video_usage.json
```

Auth check inside the container:

```
groksito --test-auth
```

## Local (optional)

```bash
git clone https://github.com/f0rrest-wwm/groksito-discord-bot.git
cd groksito-discord-bot
cp .env.example .env
python -m pip install -e .
groksito --check
groksito --login-oauth
groksito
```

## Layout that matters here

| Path | Why |
|---|---|
| `src/groksito_discord/llm/prompt_builder.py` | Meepo identity, NSFW lock, creator lock |
| `src/groksito_discord/media/image_handler.py` | Imagine stills / edits |
| `src/groksito_discord/media/video_handler.py` | Video gen |
| `src/groksito_discord/media/video_quota.py` | Daily cap + in-flight lock + NSFW gate |
| `src/groksito_discord/media/delivery.py` | Attachment filenames (`meepo_*`) |

## Upstream

Based on Groksito by [lupintic](https://github.com/lupintic). MIT. This README describes the Everest fork, not the upstream defaults.
