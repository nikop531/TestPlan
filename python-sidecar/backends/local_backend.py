"""Local backend: Nemotron diarization on this Mac (MPS, CPU fallback)."""

from __future__ import annotations

import logging

from alignment import parse_diarization_lines
from config import OFFLINE_PRESET, pick_device

from .base import BackendResult, EstimatedProgress, ProgressFn

log = logging.getLogger(__name__)

# Rough real time factors, used only for the progress estimate.
_RTF = {"mps": 0.08, "cuda": 0.02, "cpu": 0.3}


class LocalBackend:
    name = "local"

    def diarize(self, wav_path: str, duration: float, progress: ProgressFn) -> BackendResult:
        from nemo_loader import load_model

        device = pick_device()
        progress(0.0, "載入模型中")
        model = load_model(OFFLINE_PRESET, device)
        expected = duration * _RTF.get(device, 0.3)
        with EstimatedProgress(progress, expected, "辨識說話者中"):
            outputs = model.diarize(audio=[wav_path], batch_size=1, verbose=False)
        lines = outputs[0] if outputs else []
        segments = parse_diarization_lines(lines)
        progress(1.0, "說話者辨識完成")
        return BackendResult(segments=segments, info={"device": device, "preset": OFFLINE_PRESET.name})
