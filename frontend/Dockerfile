# Frontend image: the Streamlit dashboard. It contains only the frontend code
# and its own small set of libraries; it talks to the backend over REST at
# the address in BACKEND_URL (set in docker-compose.yml).
# Build context is the project root.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SAMPLES_DIR=/app/samples

WORKDIR /app

COPY frontend/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

COPY frontend/ /app/
# The example file behind the "Try with example data" button
COPY samples/fd001_mixed_engines.csv /app/samples/fd001_mixed_engines.csv

# Run as an ordinary user, not root: if the dashboard were ever compromised,
# the attacker would not have administrator rights inside the container.
RUN useradd --create-home dashboard
USER dashboard

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
