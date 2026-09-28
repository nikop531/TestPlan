#!/bin/bash
# Builds the .app and .dmg with Tauri. Run scripts/setup_mac.sh first.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "安裝 Rust 編譯工具..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

cd "$REPO"
npm install --no-audit --no-fund
npm run tauri build

echo ""
echo "完成。安裝檔位於："
ls "$REPO"/src-tauri/target/release/bundle/dmg/*.dmg
