# === CyclIA - Dockerfile para producción ===
# Base: Python 3.11 slim (más ligera que la imagen completa)
FROM python:3.11-slim

# Metadata
LABEL maintainer="Yasmine Kharbach"
LABEL project="CyclIA - TFM Agentes IA"
LABEL description="Chatbot multi-agente de salud hormonal femenina"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
# (se hace en una capa separada para aprovechar la cache de Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-descargar modelo de embeddings (evita descarga en runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copiar el código de la aplicación
COPY . .

# Puerto expuesto (Render lo sobrescribe vía $PORT, pero es buena práctica)
EXPOSE 5000

# Comando de arranque: Gunicorn con la config personalizada
CMD ["gunicorn", "--config", "gunicorn_config.py", "api:app"]