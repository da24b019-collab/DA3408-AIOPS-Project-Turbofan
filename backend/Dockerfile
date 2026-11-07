# Backend image: one image, three containers (see docker-compose.yml):
#   mlflow        - MLflow tracking server and model registry
#   model-server  - MLflow model serving of the "champion" model
#   backend       - FastAPI service used by the dashboard
# Build context is the project root, so the paths below start from there.

FROM python:3.14-slim

# PYTHONUNBUFFERED: print logs immediately (useful with `docker compose logs`)
# PYTHONDONTWRITEBYTECODE: no .pyc files inside the image
# HOME=/tmp: the containers run as the host user (see compose), who has no home
#            folder inside the image; libraries that cache files need somewhere to write
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

# libgomp1: parallel processing library XGBoost needs (not in the slim image)
# git: DVC uses it to read the repository for `dvc status`
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install libraries first: this layer is cached and only rebuilt when
# requirements.txt changes, not on every code change.
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Then the application code and settings
COPY src/ /app/src/
COPY params.yaml /app/params.yaml

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
