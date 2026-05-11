# Titan-IBKR — Champion Portfolio Live Runner
# Builds a runtime image for scripts/watchdog_portfolio.py
# Companion: docker-compose.yml (alongside ib-gateway service)

FROM python:3.11-slim

# System packages: build-essential for native wheels (numba/numpy/etc),
# tzdata so TZ=America/New_York actually resolves, ca-certs for HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# uv (pinned to a recent stable). Copying the binary from the official
# distroless image is the documented pattern.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer — cached as long as pyproject + lockfile don't change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Application code. Bind-mount data/, models/, config/ at runtime via compose,
# so they're host-editable without a rebuild.
COPY titan/ ./titan/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1 \
    TZ=America/New_York \
    PATH="/app/.venv/bin:${PATH}"

# Sanity check at build time — fails the build if titan can't import.
RUN python -c "import titan; print('titan import OK')"

# Default to the portfolio watchdog. Override with `docker compose run` for
# debug shells or one-shot scripts (kill_switch, verify_connection).
CMD ["python", "scripts/watchdog_portfolio.py", "--strategies", "champion_portfolio"]
