# stage 1: builder — installs dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# copy everything needed to build the package
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ ./src/

# install dependencies into /app/.venv
RUN uv sync --frozen --no-dev

# stage 2: runtime — lean final image
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uvicorn", "raglab.main:app", "--host", "0.0.0.0", "--port", "8000"]
