FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn langchain_ollama

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]