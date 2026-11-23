#!/usr/bin/env bash
# Jalankan Cloudflare Tunnel (cloudflared) menuju FastAPI lokal (127.0.0.1:8000).
# Mendukung 3 mode (urutan prioritas):
#   1. Remotely-Managed Tunnel (disarankan): set CLOUDFLARE_TUNNEL_TOKEN di .env
#   2. Locally-Managed Tunnel (config.yml): set CLOUDFLARE_CONFIG_PATH di .env
#   3. Quick Tunnel (tes sementara trycloudflare.com): set CLOUDFLARE_QUICK_TUNNEL=true di .env
set -euo pipefail

CLOUDFLARE_TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:-}"
CLOUDFLARE_CONFIG_PATH="${CLOUDFLARE_CONFIG_PATH:-}"
CLOUDFLARE_QUICK_TUNNEL="${CLOUDFLARE_QUICK_TUNNEL:-false}"
CLOUDFLARE_ORIGIN_URL="${CLOUDFLARE_ORIGIN_URL:-http://127.0.0.1:8000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared tidak ditemukan di PATH." >&2
  exit 1
fi

if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN}" ]]; then
  # Gunakan env TUNNEL_TOKEN (otomatis di-mask oleh cloudflared) dan unset CLOUDFLARE_TUNNEL_TOKEN agar tidak masuk log.
  export TUNNEL_TOKEN="${CLOUDFLARE_TUNNEL_TOKEN}"
  unset CLOUDFLARE_TUNNEL_TOKEN
  exec cloudflared tunnel --no-autoupdate run
fi

if [[ -n "${CLOUDFLARE_CONFIG_PATH}" ]]; then
  if [[ ! -f "${CLOUDFLARE_CONFIG_PATH}" ]]; then
    echo "ERROR: File config Cloudflare tidak ditemukan: ${CLOUDFLARE_CONFIG_PATH}" >&2
    exit 1
  fi
  exec cloudflared tunnel --no-autoupdate --config "${CLOUDFLARE_CONFIG_PATH}" run
fi

if [[ "${CLOUDFLARE_QUICK_TUNNEL,,}" == "true" || "${CLOUDFLARE_QUICK_TUNNEL}" == "1" ]]; then
  exec cloudflared tunnel --no-autoupdate \
    --proxy-connect-timeout 30s \
    --proxy-tcp-keepalive 30s \
    --proxy-keepalive-connections 100 \
    --proxy-keepalive-timeout 90s \
    --url "${CLOUDFLARE_ORIGIN_URL}"
fi

echo "Cloudflare Tunnel belum dikonfigurasi." >&2
echo "Isi salah satu di /home/ubuntu/kka_lab/.env:" >&2
echo "  - CLOUDFLARE_TUNNEL_TOKEN=<token dari Cloudflare Zero Trust Dashboard>" >&2
echo "  - CLOUDFLARE_CONFIG_PATH=/home/ubuntu/.cloudflared/config.yml" >&2
echo "  - CLOUDFLARE_QUICK_TUNNEL=true (khusus tes sementara trycloudflare.com)" >&2
exit 1
