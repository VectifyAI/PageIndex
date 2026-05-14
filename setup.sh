#!/bin/bash
# PageIndex Azure VM setup script
# Run as root on a fresh Ubuntu 22.04 VM
set -e

echo "=== Installing Docker ==="
apt-get update -qq
apt-get install -y docker.io docker-compose-plugin git curl

systemctl enable docker
systemctl start docker

echo "=== Cloning PageIndex ==="
git clone https://github.com/clapointe-carbonleo/PageIndex.git /opt/pageindex
cd /opt/pageindex
git checkout feat/fastapi-server

echo "=== Creating .env file ==="
cat > /opt/pageindex/.env << 'ENVEOF'
# Required — Anthropic API key for Claude
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME

# Allowed origins (comma-separated) — set to your mike-legal Vercel URL
ALLOWED_ORIGINS=https://mike-legal-three.vercel.app

# Optional — secret token to protect the API (add to PAGEINDEX_SECRET in Vercel)
API_SECRET=REPLACE_WITH_RANDOM_SECRET
ENVEOF

echo "=== Building and starting service ==="
cd /opt/pageindex
docker compose up -d --build

echo ""
echo "=== Done! ==="
echo "PageIndex service is running on port 8000."
echo "Edit /opt/pageindex/.env to set your real API keys, then run: docker compose restart"
