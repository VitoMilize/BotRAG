import logging
from typing import List

from fastapi import FastAPI
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from pydantic import BaseModel

# Настройка логгера
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Подключение к Ollama
embeddings = OllamaEmbeddings(
    model="llama3.1:latest",  # Убедись, что эта модель загружена
    base_url="http://ollama:11434",  # Имя сервиса в docker-compose
)
llm = Ollama(
    model="llama3.1:latest",  # Название модели, обязательно должна быть загружена
    base_url="http://ollama:11434"  # Или адрес контейнера, например, http://ollama:11434
)

app = FastAPI()


# Модели запроса
class EmbedRequestSingle(BaseModel):
    text: str


class EmbedRequestList(BaseModel):
    texts: List[str]


class QuestionRequest(BaseModel):
    question: str
    documents: List[str]


# Эндпоинт для одного текста
@app.post("/embed_single")
def embed_single(request: EmbedRequestSingle):
    try:
        vector = embeddings.embed_query(request.text)
        logger.info(f"Готов embedding длиной {len(vector)}")
        return {"embedding": vector}
    except Exception as e:
        logger.error(f"Ошибка в embed_single: {e}")
        return {"error": str(e)}


# Эндпоинт для списка текстов
@app.post("/embed_list")
def embed_list(request: EmbedRequestList):
    try:
        vectors = embeddings.embed_documents(request.texts)
        logger.info(f"Готово {len(vectors)} эмбеддингов")
        return {"embeddings": vectors}
    except Exception as e:
        logger.error(f"Ошибка в embed_list: {e}")
        return {"error": str(e)}


@app.post("/ask_question")
def ask_question(request: QuestionRequest):
    try:
        docs_text = "\n\n".join(request.documents)

        prompt = (
            "Ты — помощник, который отвечает на вопросы на основе предоставленных документов.\n"
            "Используй только факты из документов. Если ответа нет — скажи, что не знаешь.\n"
            "Отвечай только на русском языке.\n\n"
            f"Документы:\n{docs_text}\n\n"
            f"Вопрос:\n{request.question.strip()}"
        )

        result = llm.invoke(prompt)
        logger.info(result)
        return {"answer": str(result)}
    except Exception as e:
        logger.error(f"Ошибка в ask_question: {e}")
        return {"error": str(e)}
