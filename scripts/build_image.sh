#!/usr/bin/env bash
# Build, tag, and optionally save the titan-portfolio Docker image.
#
# Usage:
#   scripts/build_image.sh                    # build + tag :latest
#   scripts/build_image.sh 1.0.0              # build + tag :1.0.0 + :latest
#   scripts/build_image.sh 1.0.0 --save       # also save to dist/titan-portfolio-1.0.0.tar.gz
#   scripts/build_image.sh --save             # build :latest + save dist/titan-portfolio-latest.tar.gz
#
# After build: `docker compose up -d` will use the tagged image without
# rebuilding. To force a rebuild on the next start, pass --build to
# docker compose, or run this script again.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="titan-portfolio"
DIST_DIR="dist"

# ── Parse args ──────────────────────────────────────────────────────────────
VERSION="latest"
SAVE_TAR=false

for arg in "$@"; do
    case "$arg" in
        --save)
            SAVE_TAR=true
            ;;
        --help|-h)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            echo "Unknown flag: $arg" >&2
            exit 1
            ;;
        *)
            VERSION="$arg"
            ;;
    esac
done

# ── Build ───────────────────────────────────────────────────────────────────
echo "==> Building $IMAGE_NAME:$VERSION from $REPO_ROOT"
docker build -t "$IMAGE_NAME:$VERSION" -f Dockerfile .

# Always also tag :latest (so docker-compose.yml default keeps working)
if [ "$VERSION" != "latest" ]; then
    docker tag "$IMAGE_NAME:$VERSION" "$IMAGE_NAME:latest"
    echo "==> Also tagged as $IMAGE_NAME:latest"
fi

# Show image size for context
SIZE=$(docker image inspect "$IMAGE_NAME:$VERSION" --format='{{.Size}}' \
       | awk '{printf "%.1f MB\n", $1/1024/1024}')
echo "==> Image size: $SIZE"

# ── Optional save to tar ────────────────────────────────────────────────────
if [ "$SAVE_TAR" = true ]; then
    mkdir -p "$DIST_DIR"
    TAR_PATH="$DIST_DIR/${IMAGE_NAME}-${VERSION}.tar.gz"
    echo "==> Saving image to $TAR_PATH (this can take a minute)..."
    docker save "$IMAGE_NAME:$VERSION" | gzip > "$TAR_PATH"
    SAVED_SIZE=$(du -h "$TAR_PATH" | cut -f1)
    echo "==> Saved $TAR_PATH ($SAVED_SIZE)"
    echo ""
    echo "    To restore on this or another machine:"
    echo "      gunzip -c $TAR_PATH | docker load"
fi

echo ""
echo "==> Done. Run \`docker compose --env-file .env.docker up -d\` to start."
echo "    (No --build needed — compose will use the tagged image.)"
