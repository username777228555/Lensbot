import os
import logging
import threading
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
from openai import OpenAI
import httpx
from bs4 import BeautifulSoup
from telegram import Update, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """Ты опытный фотограф и знаток оптики, общаешься в фото-чате как свой среди своих.

Стиль общения:
- Пиши как живой человек из фотосообщества, не как справочник
- Коротко, 2-4 предложения максимум
- Используй сленг: ФФ (полный кадр), кроп (APS-C), Микра (Micro 4/3), гелик (Гелиос), сапог/кэнон (Canon), никон, сонька (Sony), фуджик (Fujifilm), сф (среднеформатная), стекло (объектив), светлое стекло, мыльница, беззер, зеркалка, боке, ГРИП, кит, телевик, ширик, макро, портретник, фикс, зум
- Никакого форматирования: без **, *, #, _, без списков — только обычный текст
- Если не знаешь — скажи честно
- Если вопрос не по теме фото/оптики — вежливо скажи об этом
- Отвечай на языке собеседника
"""

MISTAKE_PROMPT = """Ты опытный фотограф и знаток оптики. Читаешь сообщение из фото-чата.

Есть ли в сообщении фактическая ошибка по теме фотографии, объективов или оптики?

Вмешайся если человек путает технические факты: кроп-факторы, принцип работы диафрагмы/ISO/выдержки, физику оптики, байонеты, совместимость объективов, характеристики конкретных моделей.
Не вмешивайся если это мнение, вопрос, или сообщение не про фото/оптику, или ты не уверен.

Если ошибки нет — ответь: SKIP
Если ошибка есть — поправь коротко и по-дружески, 1-2 предложения, без форматирования.
"""

private_histories: dict[int, deque] = {}
group_histories: dict[int, deque] = {}

MAX_HISTORY = 30
MAX_TOKENS = 350

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


# ─── lens-club.ru через DuckDuckGo ───────────────────────────────────────────

def search_lens_club(query: str) -> str | None:
    """
    Ищет по lens-club.ru через DuckDuckGo HTML (не требует API-ключей).
    Возвращает текст со сниппетами найденных страниц.
    """
    try:
        ddg_url = "https://html.duckduckgo.com/html/"
        params = {"q": f"site:lens-club.ru {query}", "kl": "ru-ru"}

        r = httpx.post(ddg_url, data=params, headers=HEADERS, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")

        results = []
        for result in soup.select(".result")[:4]:
            title_el = result.select_one(".result__title")
            snippet_el = result.select_one(".result__snippet")
            url_el = result.select_one(".result__url")

            title = title_el.get_text(strip=True) if title_el else ""
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""
            url = url_el.get_text(strip=True) if url_el else ""

            if title or snippet:
                results.append(f"{title}\n{snippet}\n{url}".strip())

        if not results:
            return None

        return "\n\n".join(results)

    except Exception as e:
        logger.warning(f"lens-club search error: {e}")
        return None


def should_search_lens_club(text: str) -> bool:
    """Определяет, стоит ли искать на lens-club — вопрос про конкретный объектив."""
    keywords = [
        "объектив", "стекло", "линза", r"\d+mm", r"\d+мм", r"f/\d", r"f\d\.",
        "canon", "nikon", "sony", "sigma", "tamron", "zeiss", "цейсс", "voigtlander",
        "гелиос", "гелик", "юпитер", "индустар", "мир-", "зенитар", "ломо",
        "характеристики", "резкость", "боке", "автофокус", "светосил",
        "что скажешь", "как стекло", "стоит брать", "посоветуй", "обзор",
        "мтф", "mtf", "кроп-фактор", "байонет",
    ]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in keywords)


# ─── История ─────────────────────────────────────────────────────────────────

def get_private_history(user_id: int) -> deque:
    if user_id not in private_histories:
        private_histories[user_id] = deque(maxlen=MAX_HISTORY)
    return private_histories[user_id]


def get_group_history(chat_id: int) -> deque:
    if chat_id not in group_histories:
        group_histories[chat_id] = deque(maxlen=MAX_HISTORY)
    return group_histories[chat_id]


def is_mentioned(message: Message, bot_username: str) -> bool:
    if message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mention = message.text[entity.offset: entity.offset + entity.length]
                if mention.lower() == f"@{bot_username.lower()}":
                    return True
    return False


def is_reply_to_bot(message: Message, bot_id: int) -> bool:
    return (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == bot_id
    )


# ─── Health check ─────────────────────────────────────────────────────────────

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", 8000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server on port {port}")
    server.serve_forever()


# ─── Построение сообщений с данными с lens-club ───────────────────────────────

def build_messages(system: str, history: list, user_text: str) -> list:
    messages = [{"role": "system", "content": system}] + list(history)

    if should_search_lens_club(user_text):
        lens_data = search_lens_club(user_text)
        if lens_data:
            messages.append({
                "role": "system",
                "content": (
                    "Вот что нашлось на lens-club.ru по этому вопросу:\n\n"
                    f"{lens_data}\n\n"
                    "Используй эти данные, но отвечай в своём обычном живом стиле."
                )
            })
            logger.info("lens-club data injected")
        else:
            logger.info("lens-club: no results")

    messages.append({"role": "user", "content": user_text})
    return messages


# ─── Handlers ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Спрашивай про стёкла, камеры и всё такое.\n/reset — сбросить историю"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    private_histories.pop(user_id, None)
    group_histories.pop(chat_id, None)
    await update.message.reply_text("Сброшено.")


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    history = get_private_history(user_id)
    messages = build_messages(SYSTEM_PROMPT, list(history), user_text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.8,
        )
        answer = response.choices[0].message.content.strip()
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        answer = "Что-то сломалось, попробуй ещё раз."

    await update.message.reply_text(answer)


async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    user_name = message.from_user.first_name or "Кто-то"
    user_text = message.text
    bot_username = context.bot.username
    bot_id = context.bot.id

    history = get_group_history(chat_id)
    history.append({"name": user_name, "text": user_text})

    mentioned = is_mentioned(message, bot_username)
    replied = is_reply_to_bot(message, bot_id)

    if mentioned or replied:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        context_text = "\n".join(
            f"{m['name']}: {m['text']}" for m in list(history)[-15:]
        )
        full_query = f"Переписка в чате:\n{context_text}\n\nОтветь на последнее обращение к тебе."
        messages = build_messages(SYSTEM_PROMPT, [], full_query)

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.8,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            answer = "Что-то сломалось, попробуй ещё раз."

        await message.reply_text(answer)

    else:
        # Проверка на ошибку
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": MISTAKE_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=120,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            if not answer.upper().startswith("SKIP"):
                await message.reply_text(answer)
        except Exception as e:
            logger.error(f"DeepSeek mistake check error: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))

    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        handle_private,
    ))

    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
        handle_group,
    ))

    logger.info("Starting polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
        
