"""Download every model once, so Local mode works offline afterwards.

Run by scripts/setup_mac.sh. Safe to run again; cached files are reused.
"""

from __future__ import annotations

import config


def main() -> None:
    from nemo_loader import load_model

    print(f"下載說話者辨識模型 {config.MODEL_ID} ...")
    load_model(config.OFFLINE_PRESET, "cpu")
    print(f"下載語音辨識模型 faster-whisper {config.ASR_MODEL} ...")
    from faster_whisper import WhisperModel

    WhisperModel(config.ASR_MODEL, device="cpu", compute_type=config.ASR_COMPUTE_TYPE)
    print("模型已下載完成。")


if __name__ == "__main__":
    main()
