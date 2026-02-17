import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import deque
from openai import OpenAI
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

# Системный промпт — стиль ответов
SYSTEM_PROMPT = """Ты эксперт по фотографии, объективам, камерам и оптике.

Стиль:
- Пиши как человек, коротко и по делу
- Максимум 3-4 предложения, если не просят подробнее
- Никакого форматирования: никаких **, *, #, _, никаких списков с тире или цифрами — только обычный текст
- Если не знаешь точно — так и скажи, не выдумывай
- Если вопрос не по теме фото/оптики — вежливо откажи
- Отвечай на языке собеседника
"""

# Промпт для проверки ошибок в чате
MISTAKE_PROMPT = """Ты эксперт по фотографии, объективам, камерам и оптике.

Тебе покажут сообщение из чата. Твоя задача: определить, содержит ли оно очевидную фактическую ошибку по теме фотографии или оптики.

Если ошибка есть — ответь коротко и вежливо, поправь. Без форматирования, обычным текстом, 1-2 предложения.
Если ошибки нет, или сообщение не по теме фото/оптики, или ты не уверен — ответь ровно одним словом: SKIP
"""

# Личные диалоги: user_id -> deque of messages
private_histories: dict[int, deque] = {}

# Групповые чаты: chat_id -> deque of {role, content, user_name}
group_histories: dict[int, deque] = {}

MAX_HISTORY = 30
MAX_TOKENS = 350  # ~2000 символов, хватит для любого ответа


def get_private_history(user_id: int) -> deque:
    if user_id not in private_histories:
        private_histories[user_id] = deque(maxlen=MAX_HISTORY)
    return private_histories[user_id]


def get_group_history(chat_id: int) -> deque:
    if chat_id not in group_histories:
        group_histories[chat_id] = deque(maxlen=MAX_HISTORY)
    return group_histories[chat_id]


def ask_deepseek(messages: list, max_tokens: int = MAX_TOKENS) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def is_mentioned(message: Message, bot_username: str) -> bool:
    """Проверяет, упомянут ли бот в сообщении."""
    if message.entities:
        for entity in message.entities:
            if entity.type == "mention":
                mention = message.text[entity.offset: entity.offset + entity.length]
                if mention.lower() == f"@{bot_username.lower()}":
                    return True
    return False


def is_reply_to_bot(message: Message, bot_id: int) -> bool:
    """Проверяет, является ли сообщение ответом на сообщение бота."""
    return (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == bot_id
    )


# ─── Health check ────────────────────────────────────────────────────────────

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


# ─── Handlers ────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Спрашивай про объективы, камеры и оптику.\n/reset — сбросить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    private_histories.pop(user_id, None)
    group_histories.pop(chat_id, None)
    await update.message.reply_text("История сброшена.")


async def handle_private(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Личные сообщения — всегда отвечаем."""
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    history = get_private_history(user_id)
    history.append({"role": "user", "content": user_text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    try:
        answer = ask_deepseek(messages)
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        answer = "Ошибка запроса к AI, попробуй ещё раз."

    await update.message.reply_text(answer)


async def handle_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Групповые сообщения — сложная логика."""
    message = update.message
    if not message or not message.text:
        return

    chat_id = message.chat_id
    user_name = message.from_user.first_name or "Пользователь"
    user_text = message.text
    bot_username = context.bot.username
    bot_id = context.bot.id

    # Сохраняем сообщение в историю чата
    history = get_group_history(chat_id)
    history.append({"name": user_name, "text": user_text})

    mentioned = is_mentioned(message, bot_username)
    replied = is_reply_to_bot(message, bot_id)

    if mentioned or replied:
        # Отвечаем на вопрос/обращение
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # Собираем контекст чата для модели
        context_text = "\n".join(
            f"{m['name']}: {m['text']}" for m in list(history)[-15:]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Переписка в чате:\n{context_text}\n\nОтветь на последнее обращение к тебе.",
            },
        ]

        try:
            answer = ask_deepseek(messages)
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")
            answer = "Ошибка запроса, попробуй ещё раз."

        await message.reply_text(answer)

    else:
        # Проверяем, нет ли очевидной ошибки
        messages = [
            {"role": "system", "content": MISTAKE_PROMPT},
            {"role": "user", "content": user_text},
        ]

        try:
            answer = ask_deepseek(messages, max_tokens=150)
            if answer.strip().upper() != "SKIP" and len(answer) > 5:
                await message.reply_text(answer)
        except Exception as e:
            logger.error(f"DeepSeek error (mistake check): {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))

    # Личные сообщения
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        handle_private,
    ))

    # Групповые сообщения (группы и супергруппы)
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUP | filters.ChatType.SUPERGROUP),
        handle_group,
    ))

    logger.info("Starting polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
    
