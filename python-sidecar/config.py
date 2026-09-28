"""Central configuration for the diarization sidecar.

Every value can be overridden with an environment variable so the Tauri shell,
the setup script, and tests can all adjust behaviour without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# NeMo uses a few ops that MPS does not implement yet; let torch fall back to CPU
# for those instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

APP_NAME = "DualTrackDiarization"

MODEL_ID = os.environ.get("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.08  # one Sortformer output frame

HOST = os.environ.get("SIDECAR_HOST", "127.0.0.1")
PORT = int(os.environ.get("SIDECAR_PORT", "8765"))

# Demo mode: fake diarizer and ASR, no model download. Used by tests and for
# trying the UI on a machine that has not installed the ML stack yet.
MOCK_MODE = os.environ.get("DIARIZATION_MOCK", "0") == "1"


def _default_data_dir() -> Path:
    home = Path.home()
    mac_dir = home / "Library" / "Application Support" / APP_NAME
    if mac_dir.parent.exists():
        return mac_dir
    return home / f".{APP_NAME.lower()}"


DATA_DIR = Path(os.environ.get("DIARIZATION_DATA_DIR", str(_default_data_dir())))
RECORDINGS_DIR = DATA_DIR / "recordings"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
SETTINGS_FILE = DATA_DIR / "settings.json"

# Hugging Face backend. The default URL targets the serverless router; most
# users will instead deploy hf_endpoint/handler.py as a dedicated Inference
# Endpoint and paste that URL in the settings.
HF_DEFAULT_ENDPOINT = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HF_TIMEOUT_SECONDS = float(os.environ.get("HF_TIMEOUT_SECONDS", "900"))

# ASR (faster-whisper). "small" keeps a 1 hour meeting within a reasonable time
# on a MacBook Air CPU; "large-v3-turbo" is more accurate but slower.
ASR_MODEL = os.environ.get("ASR_MODEL", "small")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "zh")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")
# Nudges Whisper towards Traditional Chinese output.
ASR_INITIAL_PROMPT = os.environ.get("ASR_INITIAL_PROMPT", "以下是繁體中文的會議逐字稿。")

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4"}


@dataclass(frozen=True)
class StreamingPreset:
    """Streaming Sortformer parameters, in 0.08 s frames.

    Latency is (chunk_len + chunk_right_context) * 0.08 s.
    """

    name: str
    chunk_len: int
    chunk_right_context: int
    fifo_len: int
    spkcache_update_period: int
    spkcache_len: int
    chunk_left_context: int = 1

    @property
    def latency_seconds(self) -> float:
        return round((self.chunk_len + self.chunk_right_context) * FRAME_SECONDS, 2)


# Track A: live display.
REALTIME_PRESET = StreamingPreset(
    name="realtime_0.32s",
    chunk_len=3,
    chunk_right_context=1,
    fifo_len=188,
    spkcache_update_period=144,
    spkcache_len=188,
)

# Track B: offline refinement.
OFFLINE_PRESET = StreamingPreset(
    name="offline_30s",
    chunk_len=340,
    chunk_right_context=40,
    fifo_len=40,
    spkcache_update_period=300,
    spkcache_len=188,
)


def pick_device() -> str:
    """Return the torch device to use: env override, then MPS, then CPU."""
    override = os.environ.get("DIARIZATION_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RECORDINGS_DIR, UPLOADS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
