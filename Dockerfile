FROM python:3.12-slim

# FFmpeg for merging streams and audio conversion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# deno: JavaScript runtime yt-dlp needs for full YouTube support.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

RUN useradd --create-home appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

USER appuser
ENV MEDIA_WORK_DIR=/tmp
EXPOSE 8000

# One worker is required (job state is kept in memory); threads handle load.
# Hosts such as Render/Railway/Fly set $PORT; default to 8000 locally.
ENV TRUST_PROXY=1
CMD exec gunicorn -w 1 --threads 8 --timeout 300 -b 0.0.0.0:${PORT:-8000} app:app
