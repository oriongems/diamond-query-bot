"""
Diamond Inventory Telegram Bot
--------------------------------
Queries a Google Sheets-published CSV of diamond inventory using
natural language, powered by an NVIDIA NIM LLM for intent parsing.

Run locally with:  python bot.py
"""

import os
import io
import json
import logging
import re
import threading
import time
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
NVIDIA_NIM_API_KEY = os.environ["NVIDIA_NIM_API_KEY"]
NVIDIA_NIM_MODEL = os.environ.get("NVIDIA_NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
INVENTORY_CSV_URL = os.environ["INVENTORY_CSV_URL"]
NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
# How often (seconds) the background thread re-downloads the inventory CSV.
AUTO_REFRESH_SECONDS = int(os.environ.get("AUTO_REFRESH_SECONDS", "300"))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("diamond-bot")

# ---------------------------------------------------------------------------
# System prompt used to turn a natural-language message into a JSON filter.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an intelligent data filter for a diamond inventory system.
The user will send natural language requests. Extract the filter criteria and return it as a valid JSON object.
The available columns in the dataset are: S. No., Stock #, Shape, Weight, Color, Clarity, Cut, Polish, Symmetry, Fluorescence, Lab, Report #, Measurements, Depth %, Table %, Shade, Discount, Price Per Carat, Final Price, Inscription, Black Inclusion, CertFile, Diamond Video, Diamond Image.

Mapping rules:
- 'pcs', 'diamonds', 'stones', 'inventory' all refer to the inventory.
- 'ct', 'cts', 'carat', 'carats' ALWAYS refer to the 'Weight' column (diamond weight). Any query containing these terms with a number (e.g., "2 ct", "1.5 cts", "3 carat") must filter by the Weight column.
- "over X ct", "above X ct", "more than X ct", "greater than X ct" -> use ">" (greater than)
- "under X ct", "below X ct", "less than X ct" -> use "<" (less than)
- "X ct", "X carat" (bare number) -> use ">" (minimum weight filter)
- 'BGM' / 'No BGM' refers to the 'Shade' column (No BGM = Shade is "None" or empty).
- 'Lab' refers to 'Lab'.
- 'Color' or 'Colour' refers to the 'Color' column.
- If the user mentions specific colors like 'H & I colour', extract them into a list.
- If the user provides a specific Stock # (e.g., "251147", "stock 220398"), return a special key "stock_id" with the value.

Return ONLY a JSON object. Do not add any extra text or markdown formatting.
Example 1: "fetch pcs over 2 cts" -> {"Weight": ">2"}
Example 2: "get pcs over 3 cts. in H & I colour" -> {"Weight": ">3", "Color": ["H", "I"]}
Example 3: "fetch pcs over 1.5 cts. with VVS clarity" -> {"Weight": ">1.5", "Clarity": "VVS"}
Example 4: "show all IGI pcs" -> {"Lab": "IGI"}
Example 5: "fetch all No BGM diamonds" -> {"Shade": "None"}
Example 6: "show all Heart pcs in I colour" -> {"Shape": "Heart", "Color": "I"}
Example 7: "find 2 ct diamonds" -> {"Weight": ">2"}
Example 8: "get 1 carat stones" -> {"Weight": ">1"}
Example 9: "fetch pcs below 2 cts" -> {"Weight": "<2"}
Example 10: "show diamonds under 1.5 ct" -> {"Weight": "<1.5"}
Example 11: "5 ct diamonds" -> {"Weight": ">5"}
Example 12: "stock 251147" -> {"stock_id": "251147"}
Example 13: "220398" -> {"stock_id": "220398"}
"""

# ---------------------------------------------------------------------------
# Inventory loading
# ---------------------------------------------------------------------------
_inventory_cache: Optional[pd.DataFrame] = None
_col_lookup: dict[str, str] = {}  # lowercased header -> actual header
_inventory_lock = threading.Lock()  # guards the two globals above, read by handlers + written by the background thread


def load_inventory(force: bool = False) -> pd.DataFrame:
    """Fetch and cache the inventory CSV. Headers are matched case-insensitively."""
    global _inventory_cache, _col_lookup

    with _inventory_lock:
        if _inventory_cache is not None and not force:
            return _inventory_cache

    resp = requests.get(INVENTORY_CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    with _inventory_lock:
        _col_lookup = {c.lower().strip(): c for c in df.columns}
        _inventory_cache = df
    logger.info("Loaded %d inventory rows, columns: %s", len(df), list(df.columns))
    return df


def auto_refresh_loop(interval_seconds: int) -> None:
    """Background thread: periodically re-downloads the inventory CSV so the
    cache stays fresh without users needing to run /refresh manually."""
    while True:
        time.sleep(interval_seconds)
        try:
            load_inventory(force=True)
            logger.info("Auto-refresh: inventory reloaded.")
        except Exception:
            logger.exception("Auto-refresh: failed to reload inventory (will retry next cycle).")


# Alternate header names some sheets use, keyed by the canonical name the
# NIM system prompt / card formatter refer to.
COLUMN_ALIASES: dict[str, list[str]] = {
    "fluorescence": ["fluorescence intensity"],
}


def col(name: str) -> Optional[str]:
    """Resolve a column name case-insensitively to the actual dataframe column,
    also trying known alternate header spellings."""
    key = name.lower().strip()
    if key in _col_lookup:
        return _col_lookup[key]
    for alias in COLUMN_ALIASES.get(key, []):
        if alias in _col_lookup:
            return _col_lookup[alias]
    return None


# ---------------------------------------------------------------------------
# NIM natural-language -> filter parsing
# ---------------------------------------------------------------------------
def parse_query_with_nim(user_text: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_NIM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NVIDIA_NIM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    resp = requests.post(NIM_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences, just in case.
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    return json.loads(raw)


KNOWN_LABS = {"igi", "gia", "hrd", "agsl", "ags", "egl", "gsi", "ngtc"}
KNOWN_SHAPES = {
    "round", "princess", "cushion", "oval", "emerald", "pear", "marquise",
    "heart", "radiant", "asscher",
}
KNOWN_CLARITIES = {"fl", "if", "vvs1", "vvs2", "vvs", "vs1", "vs2", "vs", "si1", "si2", "si", "i1", "i2", "i3"}


def parse_query_locally(text: str) -> dict[str, Any]:
    """Regex/keyword based fallback parser, used when the NIM API is unreachable.
    Covers the common cases from the mapping rules; anything it can't confidently
    handle is simply left out of the returned filter dict."""
    t = text.strip()
    t_lower = t.lower()
    filters: dict[str, Any] = {}

    # Stock ID: "stock 251147", "stock# 251147", "stock no 251147", or a bare number.
    m = re.search(r"stock\s*(?:#|no\.?|number)?\s*[:\-]?\s*(\w+)", t_lower)
    if m:
        filters["stock_id"] = m.group(1).upper()
        return filters
    if re.fullmatch(r"\d+", t.strip()):
        filters["stock_id"] = t.strip()
        return filters

    # Weight / carat.
    m = re.search(
        r"(over|above|more than|greater than|under|below|less than)?\s*([\d.]+)\s*(ct\.?s?|carats?)\b",
        t_lower,
    )
    if m:
        qualifier, num = m.group(1), m.group(2)
        if qualifier in ("under", "below", "less than"):
            filters["Weight"] = f"<{num}"
        else:
            filters["Weight"] = f">{num}"

    # Lab.
    for word in re.findall(r"[a-zA-Z]+", t):
        if word.lower() in KNOWN_LABS:
            filters["Lab"] = word.upper()
            break

    # Shape.
    for shape in KNOWN_SHAPES:
        if re.search(rf"\b{shape}\b", t_lower):
            filters["Shape"] = shape.capitalize()
            break

    # Clarity.
    for clarity in sorted(KNOWN_CLARITIES, key=len, reverse=True):
        if re.search(rf"\b{clarity}\b", t_lower):
            filters["Clarity"] = clarity.upper()
            break

    # No BGM -> empty Shade.
    if "no bgm" in t_lower:
        filters["Shade"] = "None"

    # Color(s): single letters D-M near the word colour/color, e.g. "H & I colour", "I colour".
    if "colour" in t_lower or "color" in t_lower:
        letters = re.findall(r"\b([D-M])\b", t)
        if len(letters) == 1:
            filters["Color"] = letters[0]
        elif len(letters) > 1:
            filters["Color"] = letters

    return filters


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def apply_weight_filter(df: pd.DataFrame, weight_col: str, expr: str) -> pd.DataFrame:
    m = re.match(r"^([<>]=?)\s*([\d.]+)$", expr.strip())
    if not m:
        return df
    op, val = m.group(1), float(m.group(2))
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    if op == ">":
        mask = weights > val
    elif op == ">=":
        mask = weights >= val
    elif op == "<":
        mask = weights < val
    elif op == "<=":
        mask = weights <= val
    else:
        return df
    return df[mask]


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()

    for key, value in filters.items():
        if key == "stock_id":
            continue  # handled separately

        actual_col = col(key)
        if not actual_col:
            logger.warning("Unknown filter column: %s", key)
            continue

        if key.lower() == "weight" and isinstance(value, str):
            result = apply_weight_filter(result, actual_col, value)
            continue

        if key.lower() == "shade" and (
            (isinstance(value, str) and value.strip().lower() == "none")
        ):
            result = result[result[actual_col].str.strip().eq("") | result[actual_col].str.lower().eq("none")]
            continue

        if isinstance(value, list):
            values_lower = [str(v).strip().lower() for v in value]
            result = result[result[actual_col].str.strip().str.lower().isin(values_lower)]
        else:
            result = result[result[actual_col].str.strip().str.lower() == str(value).strip().lower()]

    return result


def find_by_stock_id(df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    stock_col = col("Stock #") or col("Stock#") or col("Stock")
    if not stock_col:
        return df.iloc[0:0]
    return df[df[stock_col].str.strip().str.lower() == str(stock_id).strip().lower()]


# ---------------------------------------------------------------------------
# Formatting a single stock card
# ---------------------------------------------------------------------------
def escape_md(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    if text is None:
        return ""
    text = str(text)
    for ch in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def get_val(row: pd.Series, *names: str) -> str:
    for n in names:
        c = col(n)
        if c and c in row and str(row[c]).strip():
            return str(row[c]).strip()
    return ""


def format_stock_card(row: pd.Series) -> str:
    stock_no = get_val(row, "Stock #", "Stock#", "Stock")
    carat_wt = get_val(row, "Weight")
    colour = get_val(row, "Color", "Colour")
    clarity = get_val(row, "Clarity")
    shape = get_val(row, "Shape")
    lab_name = get_val(row, "Lab")
    cut = get_val(row, "Cut")
    polish = get_val(row, "Polish")
    symmetry = get_val(row, "Symmetry")
    fluorescence = get_val(row, "Fluorescence", "Fluorescence Intensity")
    shade = get_val(row, "Shade") or "None"
    video_url = get_val(row, "Diamond Video", "Video") or "N/A"
    cert_url = get_val(row, "CertFile", "Certificate") or "N/A"

    text = (
        f"*Stock ID: {escape_md(stock_no)}*\n"
        f"{escape_md(carat_wt)} ct\\. {escape_md(colour)} "
        f"{escape_md(clarity)} {escape_md(shape)}\n"
        f"{escape_md(lab_name)} Certified \\- {escape_md(cut)} {escape_md(polish)} "
        f"{escape_md(symmetry)} {escape_md(fluorescence)}\n"
        f"*Shade:* {escape_md(shade)}\n\n"
        f"*Video:*\n{escape_md(video_url)}\n\n"
        f"*Certificate*\n{escape_md(cert_url)}"
    )
    return text


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Ask me about the diamond inventory in plain English, e.g.\n"
        "\u2022 \"show IGI pcs over 2 cts in H & I colour\"\n"
        "\u2022 \"stock 251147\"\n"
        "\u2022 \"no bgm diamonds under 1.5 ct\"\n\n"
        "Use /refresh to reload the inventory sheet."
    )


async def refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        df = load_inventory(force=True)
        await update.message.reply_text(f"Inventory reloaded: {len(df)} rows.")
    except Exception as e:
        logger.exception("Refresh failed")
        await update.message.reply_text(f"Failed to reload inventory: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        df = load_inventory()
    except Exception as e:
        logger.exception("Failed to load inventory")
        await update.message.reply_text(f"Couldn't load the inventory sheet: {e}")
        return

    # Fast path: a bare number or "stock X" is very likely a stock id;
    # still let NIM parse it so wording variations work, but this avoids
    # depending on the LLM alone for the most common query.
    try:
        filters = parse_query_with_nim(user_text)
    except Exception as e:
        logger.warning("NIM parsing failed (%s), falling back to local parser", e)
        filters = parse_query_locally(user_text)
        if not filters:
            await update.message.reply_text(
                "The AI query service is currently unavailable, and I couldn't "
                "figure out that request with basic matching either. Try something "
                "simpler, e.g. a stock number, \"over 2 ct\", or \"IGI pcs\"."
            )
            return

    if "stock_id" in filters:
        matches = find_by_stock_id(df, filters["stock_id"])
        if matches.empty:
            await update.message.reply_text(f"No stock found with ID {filters['stock_id']}.")
            return
        row = matches.iloc[0]
        card_text = format_stock_card(row)
        await update.message.reply_text(
            card_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
        )
        return

    # Otherwise: filter query -> return matching rows as a CSV file,
    # in the exact same column format as the source sheet.
    try:
        results = apply_filters(df, filters)
    except Exception as e:
        logger.exception("Filtering failed")
        await update.message.reply_text(f"Couldn't apply that filter: {e}")
        return

    if results.empty:
        await update.message.reply_text("No matching diamonds found.")
        return

    buf = io.StringIO()
    results.to_csv(buf, index=False)
    buf.seek(0)
    file_bytes = io.BytesIO(buf.getvalue().encode("utf-8"))
    file_bytes.name = "results.csv"

    await update.message.reply_document(
        document=file_bytes,
        filename="results.csv",
        caption=f"{len(results)} matching diamond(s).",
    )


def main() -> None:
    load_inventory()  # fail fast at startup if the sheet URL is bad

    refresh_thread = threading.Thread(
        target=auto_refresh_loop, args=(AUTO_REFRESH_SECONDS,), daemon=True
    )
    refresh_thread.start()
    logger.info("Background auto-refresh started (every %ds).", AUTO_REFRESH_SECONDS)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("refresh", refresh))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
