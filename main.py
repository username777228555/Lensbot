import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import google.generativeai as genai

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """Ты — эксперт по фотографии, фотоаппаратам, объективам и оптическим системам.

Правила:
- Отвечай коротко и по делу, без воды
- Если не знаешь — честно скажи «Не знаю» или «Нет точных данных»
- Не придумывай характеристики, цифры, спецификации
- Используй технически точные термины, но объясняй их если нужно
- Если вопрос не по теме фотографии/оптики — вежливо скажи, что отвечаешь только на вопросы по этой теме
- Отвечай на том языке, на котором задан вопрос
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT,
)

# Store chat sessions per user
user_sessions: dict[int, genai.ChatSession] = {}


def get_session(user_id: int) -> genai.ChatSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = model.start_chat(history=[])
    return user_sessions[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👁 Привет! Я эксперт по объективам, оптике, камерам и фотографии.\n"
        "Задавай вопросы — отвечу чётко и без лишней воды.\n\n"
        "/reset — сбросить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_sessions.pop(user_id, None)
    await update.message.reply_text("История сброшена. Начнём заново.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        session = get_session(user_id)
        response = session.send_message(user_text)
        answer = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        answer = "Ошибка при обращении к AI. Попробуй ещё раз."

    await update.message.reply_text(answer)


def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    port = int(os.environ.get("PORT", 8443))
    webhook_url = os.environ.get("WEBHOOK_URL", "")

    if webhook_url:
        # Koyeb webhook mode
        logger.info(f"Starting webhook on port {port}")
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            webhook_url=webhook_url,
            secret_token=os.environ.get("WEBHOOK_SECRET", ""),
        )
    else:
        # Local polling mode
        logger.info("Starting polling")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
