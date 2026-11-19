# Imagen de la API (Fase 3). Reutiliza el código de la Fase 2 (src/, train.py, predict.py).
FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Solo el código: datos y modelos se montan como volúmenes (-v), no se copian a la imagen.
COPY src/ ./src/
COPY tests/ ./tests/
COPY api.py client.py train.py predict.py ./

# Variables de la API (se pueden sobrescribir con -e). Las rutas son relativas a /app.
ENV MODEL_VERSION=v1 \
    MODELS_DIR=models \
    DATA_PATH=data/dataset_optimizado.csv

EXPOSE 8000

# Servidor en 0.0.0.0 para que sea accesible desde fuera del contenedor.
# No hay ENTRYPOINT: train.py / predict.py / pytest también pueden ejecutarse con "docker run ... <comando>".
CMD ["python", "-m", "flask", "--app", "api", "run", "--host", "0.0.0.0", "--port", "8000"]
