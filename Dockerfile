FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app
ENV KERAS_BACKEND=jax
ENV UV_LINK_MODE=copy

COPY . .
RUN uv sync --locked --no-dev --extra ml --extra web

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "mllol.model.api:app", "--host", "0.0.0.0", "--port", "8000"]