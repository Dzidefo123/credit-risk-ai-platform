# Credit risk platform: local demo image (SYNTHETIC data only; no Freddie Mac data, no credentials).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir uv==0.10.11

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# runtime deps + analyst UI; no dev tools in the image
RUN uv sync --frozen --no-dev --group ui
COPY configs ./configs
COPY docs ./docs
COPY docker/demo-init.sh ./docker/demo-init.sh
RUN chmod +x docker/demo-init.sh && useradd --create-home --uid 10001 app \
    && mkdir -p data artifacts reports && chown -R app:app /app
USER app
EXPOSE 8000 8501
CMD ["crisk", "serve", "--dataset", "synthetic_fixture", "--host", "0.0.0.0", "--port", "8000"]
