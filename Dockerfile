# ENVIRONMENT STAGE
FROM python:3.11.4-slim-bullseye AS env-base

# Build arguments to process
ARG APP_ENV \
    USER_ID=1000 \
    GROUP_ID=1000

# Environment variables
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
    APP_PATH="/app" \
    VENV_PATH="/app/.venv"

# Attach Poetry and Venv to Path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Add a non-root user with permissions to a working directory
RUN groupadd --gid $GROUP_ID -r user && \
    useradd -d '/app' -g user -l -r -u $USER_ID user && \
    mkdir -p /app && \
    mkdir -p /opt/poetry && \
    chown -R user:user /app && \
    chown -R user:user /opt/poetry

# Update system packages information and install GraphViz library for 
# correct in-app graph visualizations 
RUN apt-get update && \
    apt-get install --no-install-recommends -y graphviz


# BUILD STAGE
FROM env-base as build-base

# Upgrade system packages and install Poetry with curl
RUN apt-get upgrade -y && \
    apt-get install --no-install-recommends -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

WORKDIR $APP_PATH

COPY --chown=user:user poetry.lock pyproject.toml README.md LICENSE ./

# Install only 3rd party dependencies for a proper Docker cache handling
RUN poetry self add setuptools@~68.0.0 && \
    poetry install --no-root --no-directory \
    $(if [ "$APP_ENV" = "PROD" ]; then echo '--only main'; \
    elif [ "$APP_ENV" = "DEV" ]; then echo '--with docs,tests'; fi) \
    --no-interaction --no-ansi

# Copy only the root package and install it
COPY --chown=user:user ./src ./src/

RUN poetry install --only-root


# APP STAGE
FROM env-base as app-base

# Remove unnecessary cache
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR $APP_PATH

USER user

COPY --chown=user:user --from=build-base $APP_PATH ./

# Streamlit's port and entrypoint
EXPOSE 8501

ENTRYPOINT streamlit run src/mlui/🏠_Home.py


# DEVELOPMENT STAGE
FROM app-base as dev

# Copy Poetry executable and all the package folders
COPY --chown=user:user --from=build-base $POETRY_HOME $POETRY_HOME

COPY --chown=user:user . ./


# PRODUCTION STAGE
FROM app-base as prod

# Remove Poetry files
RUN rm -f poetry.lock pyproject.toml
