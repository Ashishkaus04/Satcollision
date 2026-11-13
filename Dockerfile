# Full-stack app image: builds the React frontend, then runs the FastAPI
# backend (which serves that build) on port 8000.
#
#   docker compose up --build        # see docker-compose.yml
#
# The real-catalogue world needs data/*.tle in the build context (data/ is
# git-ignored, so download the catalogues first; without them the image
# still builds and runs the two synthetic worlds). data/historical (Space-Track
# history, for real_leo_history) is left out of the image on purpose and
# mounted at run time instead (docker-compose.yml).

FROM node:20.18.0-alpine AS frontend
WORKDIR /build
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend/ ./
RUN npm run build

FROM python:3.12.7-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
COPY src/ src/
COPY app/__init__.py app/__init__.py
COPY app/backend/ app/backend/
COPY worlds/ worlds/
COPY data/ data/
COPY --from=frontend /build/dist app/frontend/dist
ENV APP_HOST=0.0.0.0 APP_PORT=8000 APP_RUNS_DIR=/srv/runs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "app.backend"]
