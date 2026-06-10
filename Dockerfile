FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg yt-dlp supervisor build-essential && rm -rf /var/lib/apt/lists/*
ARG INSTALL_BASIC_PITCH=false

WORKDIR /app

COPY api/requirements.txt /tmp/api-requirements.txt
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt -r /tmp/worker-requirements.txt
RUN if [ "$INSTALL_BASIC_PITCH" = "true" ]; then pip install --no-cache-dir basic-pitch; fi

COPY api /app/api
COPY mt3 /app/mt3
COPY mt3_config.py /app/mt3_config.py
COPY scripts /app/scripts
COPY transcription /app/transcription
COPY worker /app/worker
COPY services /app/services
COPY docker /app/docker

RUN chmod +x /app/docker/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 CMD python -c "import urllib.request,json,sys; urllib.request.urlopen('http://127.0.0.1:8080/openapi.json',timeout=5); d=json.load(urllib.request.urlopen('http://127.0.0.1:8080/worker/status',timeout=5)); sys.exit(0 if d.get('status')=='online' else 1)"

CMD ["/app/docker/start.sh"]
