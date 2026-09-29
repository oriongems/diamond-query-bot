# Diamond Inventory Telegram Bot

A Telegram bot that lets you query your diamond inventory (published as a
Google Sheets CSV) using plain English. Natural-language parsing is done via
an NVIDIA NIM LLM (`meta/llama-3.1-8b-instruct`).

## How it works

1. You send the bot a message, e.g. `show IGI pcs over 2 cts in H & I colour`.
2. The bot sends your message to NVIDIA NIM, which returns a JSON filter
   (e.g. `{"Lab": "IGI", "Weight": ">2", "Color": ["H", "I"]}`).
3. The bot downloads the latest inventory CSV and applies the filter.
4. **Single stock ID queries** (e.g. `stock 251147` or just `251147`) return
   a formatted card with the diamond's details, plus the video (embedded if
   possible) and certificate.
5. **All other filter queries** return a CSV file of every matching row, in
   the same column layout as your source sheet.

Column headers are matched case-insensitively, so it doesn't matter whether
your sheet says `Weight`, `weight`, or `WEIGHT`.

## 1. Setup

### Requirements
- Python 3.10+
- A Telegram bot token (already provided, from @BotFather)
- An NVIDIA NIM API key (already provided, from build.nvidia.com)

### Install

```bash
cd diamond-bot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

All secrets and config live in `.env` (already filled in for you):

```
TELEGRAM_BOT_TOKEN=...
NVIDIA_NIM_API_KEY=...
NVIDIA_NIM_MODEL=meta/llama-3.1-8b-instruct
INVENTORY_CSV_URL=...
```

**Important security note:** the token and API key were shared in plain text
in our conversation. Treat both as compromised — anyone with them can run a
bot as "you" or spend your NIM credits. I'd strongly recommend:
- Regenerating the Telegram token via @BotFather (`/revoke`), and
- Rotating the NVIDIA NIM key in the NVIDIA build portal,

then updating `.env` with the new values before you rely on this in
production. Never commit `.env` to a public repo — it's already listed in
`.gitignore` below.

### Run

```bash
python bot.py
```

The bot starts in polling mode — no public URL or webhook needed, so it's
fine to run on your own machine. Leave the terminal window open (or run it
under `pm2`, `screen`, `tmux`, or as a background service) for the bot to
stay online.

## 2. Using the bot

Open a chat with your bot on Telegram and try things like:

| You type | What happens |
|---|---|
| `251147` or `stock 251147` | Returns the formatted card for that stock, with video/certificate |
| `show IGI pcs over 2 cts` | Returns a CSV of all IGI diamonds over 2 carats |
| `pcs over 3 cts in H & I colour` | Returns a CSV filtered by weight and color |
| `no bgm diamonds under 1.5 ct` | Returns a CSV of diamonds with no shade, under 1.5 ct |
| `all heart shape pcs in I colour` | Returns a CSV of heart-shaped I-color diamonds |

Commands:
- `/start` — shows a quick usage hint
- `/refresh` — re-downloads the inventory CSV (the bot caches it after the
  first load / bot startup for speed; use this if you've just updated the
  sheet and want fresh results immediately)

## 3. Notes & limitations

- **Video/certificate embedding**: Telegram can only embed media it can
  fetch directly (a direct `.mp4`/`.mov` link for video, a direct
  `.pdf`/`.jpg`/`.png` link for the certificate). If your sheet's links are
  viewer pages (e.g. a Google Drive "view" link, a Dropbox share page) rather
  than direct file URLs, Telegram can't render them inline — the bot detects
  this automatically and falls back to sending the plain link instead of
  failing.
- **Weight filters**: "2 ct" / "2 carats" (no comparator) is treated as
  "**over** 2 ct" (`>2`), per your filtering rules — not "exactly 2 ct".
- **The LLM only extracts filters** — it never sees or invents diamond data;
  all actual data comes straight from your CSV, so results can't be
  hallucinated.
- **Cache**: the inventory is loaded once at startup and cached in memory.
  Run `/refresh` after editing the Google Sheet to pick up changes without
  restarting the bot.
- If NVIDIA NIM returns something that isn't valid JSON (rare, but LLMs can
  misfire), the bot will report a parsing error rather than guessing.

## 4. Files

```
diamond-bot/
├── bot.py             # main bot logic
├── requirements.txt   # Python dependencies
├── .env               # secrets & config (do not share/commit)
└── README.md          # this file
```
