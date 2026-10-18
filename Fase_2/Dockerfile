# Imagen para ejecutar train.py, predict.py y prepare_data.py (Fase 2).
# Python 3.12: misma versión usada en el entorno local (ver README).
FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código de la aplicación (los datos y modelos se montan como volúmenes, NO se copian)
COPY src/ ./src/
COPY tests/ ./tests/
COPY train.py predict.py prepare_data.py ./

# No hay ENTRYPOINT: el comando se indica en cada "docker run".
# No se entrena nada durante el build. Por defecto solo muestra la ayuda.
CMD ["python", "train.py", "--help"]
