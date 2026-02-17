import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from openai import OpenAI
from telegram import Update
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

SYSTEM_PROMPT = """Ты — эксперт по фотографии, фотоаппаратам, объективам и оптическим системам.

Правила:
- Отвечай коротко и по делу, без воды
- Если не знаешь — честно скажи «Не знаю» или «Нет точных данных»
- Не придумывай характеристики, цифры, спецификации
- Используй технически точные термины, но объясняй их если нужно
- Если вопрос не по теме — вежливо скажи, что отвечаешь только на вопросы по фотографии и оптике
- Отвечай на том языке, на котором задан вопрос
"""

# История диалогов: user_id -> list of messages
user_histories: dict[int, list] = {}


def get_history(user_id: int) -> list:
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]


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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👁 Привет! Я эксперт по объективам, оптике, камерам и фотографии.\n"
        "Задавай вопросы — отвечу чётко и без лишней воды.\n\n"
        "/reset — сбросить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_histories.pop(user_id, None)
    await update.message.reply_text("История сброшена.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        history = get_history(user_id)
        history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})

        # Ограничиваем историю последними 20 сообщениями
        if len(history) > 20:
            user_histories[user_id] = history[-20:]

    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        answer = f"Ошибка: {e}"

    await update.message.reply_text(answer)


def main() -> None:
    t = threading.Thread(target=start_health_server, daemon=True)
    t.start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
    
