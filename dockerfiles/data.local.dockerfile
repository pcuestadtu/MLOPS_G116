# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src src/
COPY data/raw data/raw
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY .dvc/config .dvc/config
COPY dockerfiles/data_local_entrypoint.sh /usr/local/bin/data_entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --verbose
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . --no-deps --verbose
RUN dvc config core.no_scm true

RUN chmod +x /usr/local/bin/data_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/data_entrypoint.sh"]
