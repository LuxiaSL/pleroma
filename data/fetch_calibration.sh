#!/usr/bin/env bash
# Fetch positional_means.npz (172MB — too large for the git tree; ships as a
# GitHub release asset). Everything else the loom needs is already in data/.
set -euo pipefail

DEST="$(dirname "$0")/calibration/3b/positional_means.npz"
URL="https://github.com/LuxiaSL/pleroma/releases/download/data-v1/positional_means_3b.npz"
SHA="f5c238aa085140fc9a44783a98a5d7fa820687d8d23b934322736f3bb5053893"

if [ -f "$DEST" ] && echo "$SHA  $DEST" | sha256sum -c --quiet 2>/dev/null; then
    echo "already present and verified: $DEST"
    exit 0
fi

echo "fetching positional_means_3b.npz (172MB)..."
curl -L --fail --progress-bar -o "$DEST.part" "$URL"
echo "$SHA  $DEST.part" | sha256sum -c --quiet
mv "$DEST.part" "$DEST"
echo "verified -> $DEST"
