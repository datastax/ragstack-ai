# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit

################################
# BUILDER-BASE
################################

FROM python:3.12.3-slim as builder-base

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
COPY libs/ ./libs
RUN cd libs/langflow && $POETRY_HOME/bin/poetry lock --no-update && $POETRY_HOME/bin/poetry install --no-root
RUN pip show langflow | grep Location | cut -d ' ' -f 2 | xargs -I {} cp -r {}/langflow/frontend /tmp/frontend


################################
# RUNTIME
################################
FROM nginxinc/nginx-unprivileged:stable-bookworm-perl as runtime

LABEL org.opencontainers.image.title=langflow-frontend
LABEL org.opencontainers.image.authors=['Langflow']
LABEL org.opencontainers.image.licenses=MIT
LABEL org.opencontainers.image.url=https://github.com/langflow-ai/langflow
LABEL org.opencontainers.image.source=https://github.com/langflow-ai/langflow

COPY --from=builder-base --chown=nginx /tmp/frontend /usr/share/nginx/html
COPY --chown=nginx ./docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --chown=nginx ./docker/frontend/start-nginx.sh /start-nginx.sh
RUN chmod +x /start-nginx.sh
ENTRYPOINT ["/start-nginx.sh"]