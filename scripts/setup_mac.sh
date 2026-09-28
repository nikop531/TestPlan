#!/bin/bash
# One-time setup for MacBook Air (Apple Silicon).
# Installs Python 3.11 (through uv), the sidecar dependencies and the models.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP_SUPPORT="$HOME/Library/Application Support/DualTrackDiarization"
VENV="$APP_SUPPORT/venv"

echo "== 雙軌說話者辨識：安裝程式 =="

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "注意：這台電腦不是 Apple Silicon，將改用 CPU 執行，速度會比較慢。"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "安裝 uv（Python 管理工具）..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$APP_SUPPORT"
echo "建立 Python 3.11 環境於 $VENV ..."
uv venv --python 3.11 "$VENV"
echo "安裝 Python 套件（第一次約需 5 到 10 分鐘）..."
uv pip install --python "$VENV/bin/python" -r "$REPO/python-sidecar/requirements.txt"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "提示：沒有安裝 ffmpeg。WAV 檔可以直接使用；若要匯入 MP3 或 M4A，請執行 brew install ffmpeg。"
fi

echo "下載模型（之後本機模式可離線使用）..."
(cd "$REPO/python-sidecar" && "$VENV/bin/python" prefetch_models.py)

if command -v npm >/dev/null 2>&1; then
  echo "安裝介面套件..."
  (cd "$REPO" && npm install --no-audit --no-fund)
else
  echo "提示：沒有安裝 Node.js。請到 https://nodejs.org 下載 LTS 版本後，再執行一次本程式。"
fi

echo ""
echo "安裝完成。"
echo "  打包成 App：scripts/build_app.sh"
echo "  直接用瀏覽器試用：scripts/start_browser.sh"
