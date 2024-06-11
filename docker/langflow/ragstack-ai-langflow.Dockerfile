# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit


################################
# BUILDER-BASE
# Used to build deps + create our virtual environment
################################

# force platform to the current architecture to increase build speed time on multi-platform builds
FROM --platform=$BUILDPLATFORM python:3.12-slim as builder-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.8.2 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential npm \
    # gcc
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY libs/langflow/pyproject.toml libs/langflow/poetry.lock libs/langflow/README.md ./
COPY libs/test-utils ./test-utils
RUN $POETRY_HOME/bin/poetry lock --no-update \
      && $POETRY_HOME/bin/poetry build -f wheel \
      && $POETRY_HOME/bin/poetry run pip install dist/*.whl --force-reinstall

################################
# RUNTIME
# Setup user, utilities and copy the virtual environment only
################################
FROM python:3.12-slim as runtime

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title=ragstack-ai-langflow
LABEL org.opencontainers.image.authors=['RAGStack']
LABEL org.opencontainers.image.licenses=BUSL-1.1
LABEL org.opencontainers.image.url=https://github.com/datastax/ragstack-ai
LABEL org.opencontainers.image.source=https://github.com/datastax/ragstack-ai

RUN useradd user -u 1000 -g 0 --no-create-home --home-dir /app/data
COPY --from=builder-base --chown=1000 /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

USER user
WORKDIR /app

ENTRYPOINT ["python", "-m", "langflow", "run"]
CMD ["--host", "0.0.0.0", "--port", "7860"]