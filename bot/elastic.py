import json
import logging

import aiohttp

from elasticsearch import Elasticsearch

HOST = "http://elasticsearch:9200"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # можно заменить на WARNING, если нужно убрать даже свои INFO-логи
)
logger = logging.getLogger(__name__)


def check_user_index(user_id):
    es = Elasticsearch(HOST)

    if es.indices.exists(index=user_id):
        logger.info(f"Индекс '{user_id}' существует.")
    else:
        body = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 4096
                    }
                }
            }
        }
        es.indices.create(index=user_id, body=body)
        logger.info(f"Индекс '{user_id}' создан с полем embedding типа dense_vector.")


def clear_index(user_id):
    es = Elasticsearch(
        HOST
    )

    es.indices.delete(index=user_id, ignore=[400, 404])
    logger.info(f"Индекс '{user_id}' удален.")


def add_document(user_id: str, content: str, embedding: list[float]):
    es = Elasticsearch(HOST)

    es.index(
        index=user_id,
        body={
            "content": content,
            "embedding": embedding
        }
    )


async def get_embedding(text: str) -> list[float]:
    async with aiohttp.ClientSession() as session:
        async with session.post("http://llm:8000/embed_single", json={"text": text}) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["embedding"]
            else:
                raise Exception(f"Ошибка при запросе эмбеддинга: {resp.status}")


async def add_text_file(user_id: str, file_path: str, chunk_size: int = 100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    logger.info(f"Загружено {len(chunks)} чанков из файла '{file_path}'.")

    for chunk in chunks:
        embedding = await get_embedding(chunk)
        add_document(user_id, chunk, embedding)


def search_documents_text(user_id: str, query: str, top_k: int = 5) -> list[str]:
    es = Elasticsearch(HOST)
    try:
        response = es.search(
            index=user_id,
            body={
                "size": top_k,
                "query": {
                    "match": {
                        "content": query
                    }
                }
            }
        )
    except Exception as e:
        logger.error(f"Ошибка при поиске по тексту: {e}")
        return []

    hits = response["hits"]["hits"]
    return [hit["_source"]["content"] for hit in hits]


def search_documents_vector(user_id: str, embedding: list[float], top_k: int = 5) -> list[str]:
    es = Elasticsearch(HOST)
    try:
        response = es.search(
            index=user_id,
            body={
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": embedding}
                        }
                    }
                }
            }
        )
    except Exception as e:
        logger.error(f"Ошибка при векторном поиске: {e}")
        return []

    hits = response["hits"]["hits"]
    return [hit["_source"]["content"] for hit in hits]

