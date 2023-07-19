# Environment stage
FROM python:3.11.4-slim-bookworm AS env-base

ARG APP_ENV

ENV APP_ENV=${APP_ENV} \
    # Python
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    # pip
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry
    POETRY_VERSION=1.5.1 \
    POETRY_HOME='/opt/poetry' \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    # Paths
    APP_PATH="/opt/app" \
    VENV_PATH="/opt/app/.venv"

# Attach Poetry and virtual environment to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y graphviz


# Build stage
FROM env-base as build-base

RUN apt-get install --no-install-recommends -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR $APP_PATH

COPY poetry.lock pyproject.toml ./

RUN poetry install --no-root --no-directory \
    $(if [ "$APP_ENV" = "PROD" ]; then echo '--only main'; \
    elif [ "$APP_ENV" = "DEV" ]; then echo '--with docs,test'; fi)


# Development stage
FROM env-base as dev

WORKDIR $APP_PATH

COPY --from=build-base $POETRY_HOME $POETRY_HOME
COPY --from=build-base $APP_PATH ./
COPY . ./

RUN poetry install --only-root

EXPOSE 8501

ENTRYPOINT streamlit run src/mlui/🏠_Home.py --server.port=8501 --server.address=0.0.0.0
