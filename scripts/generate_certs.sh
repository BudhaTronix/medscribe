#!/usr/bin/env bash
# Generate certs/cert.pem + certs/key.pem for the Gradio UI's HTTPS launch.
# Prefers mkcert (browser-trusted, no security warning) and falls back to a
# plain self-signed certificate (browser will show a one-time warning) if
# mkcert isn't installed.
set -euo pipefail

mkdir -p certs

# mkcert may live outside the non-interactive PATH (e.g. ~/.local/bin).
export PATH="$HOME/.local/bin:$PATH"

# Cover every address the UI might be opened from, so the microphone works
# from other devices on the network too, not only from localhost.
SAN_NAMES=(localhost 127.0.0.1 ::1 "$(hostname)")
while read -r ip; do
  case "$ip" in
    172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) ;; # skip docker bridge networks
    *) SAN_NAMES+=("$ip") ;;
  esac
done < <(ip -4 addr show scope global 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)

if command -v mkcert >/dev/null 2>&1; then
  echo "Using mkcert for a certificate your browser already trusts."
  # Installing into the OS trust store needs sudo; the NSS store (Chrome,
  # Chromium, Firefox) is user-writable, which is all the browser needs.
  mkcert -install 2>/dev/null || TRUST_STORES=nss mkcert -install
  mkcert -key-file certs/key.pem -cert-file certs/cert.pem "${SAN_NAMES[@]}"
  echo "Restart the browser once so it picks up the new trust store."
else
  echo "mkcert not found; generating a plain self-signed certificate instead."
  echo "Your browser will show a one-time 'connection is not private' warning for it."
  echo "Install mkcert (e.g. 'sudo apt install mkcert' or 'brew install mkcert'), then"
  echo "run 'make certs' again to remove that warning."
  SAN=""
  for name in "${SAN_NAMES[@]}"; do
    case "$name" in
      *[a-zA-Z]*) SAN+="DNS:$name," ;;
      *) SAN+="IP:$name," ;;
    esac
  done
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout certs/key.pem -out certs/cert.pem \
    -subj "/CN=localhost" \
    -addext "subjectAltName=${SAN%,}"
fi
