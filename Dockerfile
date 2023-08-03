# Environment stage
FROM python:3.11.4-slim-bullseye AS env-base

ARG APP_ENV \
    USER_ID=1000 \
    GROUP_ID=1000

ENV APP_ENV=$APP_ENV \
    USER_ID=$USER_ID \
    GROUP_ID=$GROUP_ID \
    # Python
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    # Pip
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry
    POETRY_VERSION=1.5.1 \
    POETRY_HOME='/opt/poetry' \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_INSTALLER_MAX_WORKERS=10 \
    # Streamlit
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE="poll" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Paths
    APP_PATH="/usr/src/app" \
    VENV_PATH="/usr/src/app/.venv"

# Attach Poetry and Virtual Environment to Path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Add a non-root user with permissions to Workdir
RUN groupadd --gid $GROUP_ID user && \
    adduser user --ingroup user --gecos '' --disabled-password --uid $USER_ID && \
    mkdir -p /usr/src/app && \
    chown -R user:user /usr/src/app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y graphviz


# Build stage
FROM env-base as build-base

RUN apt-get install --no-install-recommends -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR $APP_PATH

COPY --chown=user:user poetry.lock pyproject.toml ./

RUN poetry install --no-root --no-directory \
    $(if [ "$APP_ENV" = "PROD" ]; then echo '--only main'; \
    elif [ "$APP_ENV" = "DEV" ]; then echo '--with docs,tests'; fi)


# Development stage
FROM env-base as dev

WORKDIR $APP_PATH

USER user

COPY --chown=user:user --from=build-base $POETRY_HOME $POETRY_HOME
COPY --chown=user:user --from=build-base $APP_PATH ./
COPY . ./

RUN poetry install --only-root

EXPOSE 8501

ENTRYPOINT streamlit run src/mlui/🏠_Home.py
