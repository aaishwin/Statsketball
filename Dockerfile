# Statsketball API — serving-only image
# Heavy ML stack (sklearn/umap/hdbscan/scrapy) and the raw CSV data are
# excluded; index rebuilds and scraping happen locally, artifacts are
# baked in at build time and swapped in on deploy.
FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FAISS_OUTPUT_DIR=/app/faiss_output \
    PLAYER_OUTPUT_DIR=/app/output_players \
    HEADSHOT_JSON_PATH=/app/data/nba_player_headshots.json \
    NBA_DATA_DIR=/app/data/csvs \
    ENABLE_DOCS=0

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY backend/app ./app
COPY faiss_output ./faiss_output
COPY output_players/players_with_archetypes.csv ./output_players/
COPY output_players/cluster_profiles.json ./output_players/
COPY backend/data/nba_player_headshots.json ./data/

EXPOSE 8000
# Shell form so ${PORT} expands (Render sets PORT=10000 for Docker services;
# defaults to 8000 for local runs / Fly).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
