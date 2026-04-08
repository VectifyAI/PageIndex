#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install runpodctl if not present
if ! command -v runpodctl &> /dev/null; then
  mkdir -p ~/.local/bin
  curl -sL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64.tar.gz | tar xz -C ~/.local/bin
  chmod +x ~/.local/bin/runpodctl
fi

# Ensure ~/.local/bin is on PATH for the session
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"

# Configure runpodctl API key from .env if present and not already configured
ENV_FILE="$CLAUDE_PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  RUNPOD_KEY=$(grep -E '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d'=' -f2- || true)
  if [ -n "$RUNPOD_KEY" ]; then
    CONFIG_DIR="$HOME/.runpod"
    mkdir -p "$CONFIG_DIR"
    if [ ! -f "$CONFIG_DIR/config.toml" ] || ! grep -q "apikey" "$CONFIG_DIR/config.toml" 2>/dev/null; then
      cat > "$CONFIG_DIR/config.toml" << TOML
apikey = '$RUNPOD_KEY'
apiurl = 'https://api.runpod.io/graphql'
TOML
    fi
  fi
fi
