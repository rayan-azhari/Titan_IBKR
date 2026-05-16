# Titan-IBKR — V3.7 Multi-Strategy Portfolio Runner (paper/live production)
# Builds a runtime image for scripts/watchdog_portfolio.py.
# Companion: docker-compose.yml (alongside ib-gateway service).
#
# Switched 2026-05-15 from the V1 champion_portfolio bundle to the
# V2.0 framework-audited GEM cell C12 (see directives/Pre-Reg GEM
# Dual Momentum 2026-05-14.md §4.2).
#
# Evolved 2026-05-17 from single-strategy watchdog_gem.py to V3.7
# multi-strategy watchdog_portfolio.py. The 'v37_live' STRATEGY_SETS
# entry runs GEM J5 + turtle CAT in the same TradingNode with shared
# PRM + Allocator. See directives/V3.7 Multi-Strategy Live
# Architecture 2026-05-17.md.
#
# To roll back to the previous GEM-only entrypoint, set CMD back to
# ["python", "scripts/watchdog_gem.py"] in this Dockerfile or override
# via docker-compose.yml `command:` block.

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
#
# research/ is included because the GEM live class delegates to the research
# function gem_returns / gem_target_weights -- this is the parity-by-
# construction design from V3.6 A10. Strategy mechanics live ONCE in
# research/ and the live class imports them directly.
COPY titan/ ./titan/
COPY scripts/ ./scripts/
COPY research/ ./research/

ENV PYTHONUNBUFFERED=1 \
    TZ=America/New_York \
    PATH="/app/.venv/bin:${PATH}"

# Sanity check at build time — fails the build if titan or any V3.7 LIVE
# strategy module can't import.
RUN python -c "import titan; print('titan import OK')" \
 && python -c "from titan.strategies.gem.strategy import GemStrategy; print('gem strategy import OK')" \
 && python -c "from titan.strategies.turtle.strategy import TurtleStrategy; print('turtle strategy import OK')"

# Default to the V3.7 multi-strategy portfolio watchdog. The watchdog
# forwards --strategies to scripts/run_portfolio.py, which initialises a
# single NautilusTrader TradingNode with all selected strategies sharing
# PRM + Allocator. Default set 'v37_live' = GEM J5 + turtle CAT.
#
# Override with docker-compose `command:` block to run a different set
# (e.g., samir_validation, champion_portfolio) or to roll back to
# scripts/watchdog_gem.py for the GEM-only path.
CMD ["python", "scripts/watchdog_portfolio.py", "--strategies", "v37_live"]
