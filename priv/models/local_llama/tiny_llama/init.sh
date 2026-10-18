#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

model=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
revision=52e7645ba7c309695bec7ac98f4f005b139cf465
sha256=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0

verify() {
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$1")
  else
    actual=$(shasum -a 256 "$1")
  fi
  if [[ ${actual%% *} != "$sha256" ]]; then
    echo "TinyLlama fixture checksum mismatch: $1" >&2
    return 1
  fi
}

if [[ -f "$model" ]]; then
  verify "$model"
  echo "Verified cached TinyLlama fixture ($revision)"
else
  temporary=$(mktemp "${model}.download.XXXXXX")
  trap 'rm -f "$temporary"' EXIT
  curl --fail --location --retry 3 --output "$temporary" \
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/$revision/$model"
  verify "$temporary"
  mv "$temporary" "$model"
  echo "Downloaded and verified TinyLlama fixture ($revision)"
fi
