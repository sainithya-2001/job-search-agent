# Job-search agent, headless commands only (run, search, score, tailor, digest, queue, notion-sync, ...).
# `review` opens a visible browser for you to submit, so it runs on the host, not here.
#
# Nothing personal is in the image: data/ (master resume, answer bank, jobs.db), config.local.yaml and .env are
# mounted at runtime.
# Only the code paths listed below are copied, and .dockerignore excludes data/, .env, output/ and logs/ as well.
#
#   docker build -t job-agent .
#   docker run --rm --env-file .env \
#     -v "$PWD/data:/app/data" -v "$PWD/output:/app/output" -v "$PWD/logs:/app/logs" \
#     -v "$PWD/config.local.yaml:/app/config.local.yaml:ro" \
#     job-agent run --dry-run
FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    TZ=America/New_York

WORKDIR /app

# tzdata: the digest is dated by local day (override with -e TZ=...).
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY main.py config.yaml ./
COPY src/ src/
COPY templates/ templates/
COPY evals/*.py evals/
COPY data/answer_bank.example.yaml data/

# Run as an unprivileged user. Mounted data/, output/ and logs/ must be writable by it.
RUN useradd --create-home --uid 1000 agent \
    && mkdir -p data output logs \
    && chown -R agent /app /ms-playwright
USER agent

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
