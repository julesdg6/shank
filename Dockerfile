FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg yt-dlp supervisor build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt /tmp/api-requirements.txt
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt -r /tmp/worker-requirements.txt

COPY api /app/api
COPY worker /app/worker
COPY services /app/services
COPY docker /app/docker

RUN chmod +x /app/docker/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/openapi.json', timeout=5)"

CMD ["/app/docker/start.sh"]
