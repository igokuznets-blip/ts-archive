FROM python:3.12-slim AS app

# ffmpeg из репозитория дистрибутива; для изолированной сети укажите зеркало
# через --build-arg APT_MIRROR=http://nexus.local/repository/debian
ARG APT_MIRROR=
RUN if [ -n "$APT_MIRROR" ]; then \
      sed -i "s|http://deb.debian.org/debian|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY app/tsarchive ./tsarchive
COPY db/migrations ./db/migrations
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app MIGRATIONS_DIR=/app/db/migrations ARCHIVE_ROOT=/archive

RUN useradd -u 1000 -m tsa && mkdir -p /archive && chown tsa /archive
USER tsa
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "tsarchive"]
CMD ["worker"]

# ---------------------------------------------------------------- образ воркера разметки
# docker compose build worker-detect   (target detect: + numpy, OpenVINO, NNCF для INT8)
FROM app AS detect
USER root
COPY app/requirements-detect.txt .
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" -r requirements-detect.txt
USER tsa
