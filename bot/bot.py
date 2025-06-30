import json
import os
import aiohttp
import asyncio
import logging
import tempfile
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from elastic import check_user_index, clear_index, add_document, search_documents_vector, \
    search_documents_text

BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # можно заменить на WARNING, если нужно убрать даже свои INFO-логи
)
logger = logging.getLogger(__name__)

# Подавить избыточные логи библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.bot").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.dispatcher").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.application").setLevel(logging.WARNING)
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)

keyboard = [
    ["Спросить вопрос"],
    ["Добавить документ"],
    ["Поиск по документам"],
    ["Очистить индекс"]
]
main_keyboard = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    logger.info(f"/start от пользователя {user_id}")
    check_user_index(user_id)
    await update.message.reply_text(
        "Привет! Я Telegram-бот 🔧\nВыбери, что хочешь сделать:",
        reply_markup=main_keyboard
    )


async def get_embedding(text: str) -> list[float]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://llm:8000/embed_single", json={"text": text}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Длина полученного эмбеддинга: {len(data['embedding'])}")
                    return data["embedding"]
                else:
                    text_error = await resp.text()
                    raise Exception(f"Ошибка при запросе эмбеддинга: {resp.status}, тело: {text_error}")
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return []

async def get_answer(context_docs: list[str], question: str) -> str:
    payload = {
        "question": question,
        "documents": context_docs
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post("http://llm:8000/ask_question", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["answer"]
                else:
                    return f"Ошибка API модели: {resp.status}"
        except Exception as e:
            logger.error(f"Ошибка при вызове LLM API: {e}")
            return f"❌ Ошибка при вызове LLM API: {e}"



async def handle_text_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    document = update.message.document

    # Проверка, что это .txt
    if not document.file_name.endswith(".txt"):
        await update.message.reply_text("Пожалуйста, отправьте .txt файл.")
        return

    # Скачиваем файл во временную папку
    file = await document.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        await file.download_to_drive(f.name)
        file_path = f.name

    # Загружаем документ по чанкам
    try:
        from elastic import add_text_file
        await add_text_file(user_id, file_path, chunk_size=100)
        await update.message.reply_text("✅ Файл успешно загружен и разбит на документы.", reply_markup=main_keyboard)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при обработке файла: {e}", reply_markup=main_keyboard)
    context.user_data["awaiting_document"] = False


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = str(update.message.from_user.id)
    logger.info(f"Сообщение от пользователя {user_id}: {text}")
    check_user_index(user_id)

    if context.user_data.get("awaiting_question"):
        if text == "Отмена":
            context.user_data["awaiting_question"] = False
            logger.info(f"Пользователь {user_id} отменил вопрос")
            await update.message.reply_text("Вопрос отменен.", reply_markup=main_keyboard)
        else:
            embedding = await get_embedding(text)

            docs_vector = search_documents_vector(user_id, embedding, top_k=7)
            docs_text = search_documents_text(user_id, text, top_k=2)

            # Объединение с удалением дубликатов
            combined = list(dict.fromkeys(docs_vector + docs_text))  # сохраняет порядок и удаляет повторы

            logger.info(f"Формируем запрос к LLM с {len(combined)} документами и вопросом: {text[:100]}...")
            answer = await get_answer(combined, text)

            await update.message.reply_text(f"Ответ модели:\n{answer}", reply_markup=main_keyboard)
            context.user_data["awaiting_question"] = False
        return

    if context.user_data.get("awaiting_document"):
        if text == "Отмена":
            context.user_data["awaiting_document"] = False
            logger.info(f"Пользователь {user_id} отменил добавление документа")
            await update.message.reply_text("Добавление документа отменено.", reply_markup=main_keyboard)
        else:
            embedding = await get_embedding(text)
            add_document(user_id, text, embedding=embedding)
            logger.info(f"Пользователь {user_id} добавил документ в индекс")
            await update.message.reply_text(f"Документ добавлен в индекс {user_id}.", reply_markup=main_keyboard)
            context.user_data["awaiting_document"] = False
        return

    if context.user_data.get("awaiting_content"):
        if text == "Отмена":
            logger.info(f"Пользователь {user_id} отменил поиск по документам")
            await update.message.reply_text("Поиск по документам отменен.", reply_markup=main_keyboard)
            context.user_data["awaiting_content"] = False
        else:
            try:
                embedding = await get_embedding(text)
                logger.info(f"Длина полученного эмбеддинга: {len(embedding)}")

                docs_vector = search_documents_vector(user_id, embedding, top_k=3)  # список строк
                docs_text = search_documents_text(user_id, text, top_k=2)  # список строк

                combined = ["🔍 *Векторный поиск:*"] + docs_vector + ["\n🔤 *Полнотекстовый поиск:*"] + docs_text
                combined_text = "\n\n".join(combined)

                logger.info(f"Пользователь {user_id} выполнил комбинированный поиск.")
                await update.message.reply_text(combined_text, reply_markup=main_keyboard, parse_mode="Markdown")
                context.user_data["awaiting_content"] = False
            except Exception as e:
                logger.error(f"Ошибка при комбинированном поиске: {e}")
                await update.message.reply_text("Произошла ошибка при поиске.", reply_markup=main_keyboard)
                context.user_data["awaiting_content"] = False
        return

    if text == "Спросить вопрос":
        context.user_data["awaiting_question"] = True
        await update.message.reply_text("Напиши вопрос. Для отмены команды введите \"Отмена\"",
                                        reply_markup=ReplyKeyboardRemove())
    elif text == "Добавить документ":
        context.user_data["awaiting_document"] = True
        await update.message.reply_text("Отправь текст документа для добавления. Для отмены команды введите \"Отмена\"",
                                        reply_markup=ReplyKeyboardRemove())
    elif text == "Поиск по документам":
        context.user_data["awaiting_content"] = True
        await update.message.reply_text("Отправь текст по которому нужно искать. Для отмены команды введите \"Отмена\"",
                                        reply_markup=ReplyKeyboardRemove())
    elif text == "Очистить индекс":
        clear_index(user_id)
        logger.info(f"Пользователь {user_id} очистил индекс")
        await update.message.reply_text(f"Индекс {user_id} очищен.", reply_markup=main_keyboard)
    else:
        logger.warning(f"Пользователь {user_id} отправил неизвестную команду: {text}")
        await update.message.reply_text("Такой команды не существует.", reply_markup=main_keyboard)


def main():
    logger.info("Запуск бота...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    app.add_handler(MessageHandler(filters.Document.TEXT, handle_text_file))  # <---

    logger.info("✅ Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
