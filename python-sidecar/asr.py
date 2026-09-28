"""Speech recognition with word timestamps (faster-whisper).

Output is converted to Traditional Chinese (Taiwan) with OpenCC, because
Whisper often answers in Simplified characters even for Taiwanese speakers.
"""

from __future__ import annotations

import logging
import threading

from alignment import Segment, Word
from config import ASR_COMPUTE_TYPE, ASR_INITIAL_PROMPT, ASR_LANGUAGE, ASR_MODEL, MOCK_MODE

from backends.base import ProgressFn

log = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()
_converter = None


def _to_traditional(text: str) -> str:
    global _converter
    if _converter is None:
        try:
            from opencc import OpenCC

            _converter = OpenCC("s2twp")
        except Exception:
            log.warning("OpenCC unavailable, keeping ASR text as is")
            _converter = False
    return _converter.convert(text) if _converter else text


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            # CTranslate2 has no MPS backend; int8 on CPU is the fastest option on Apple Silicon.
            _model = WhisperModel(ASR_MODEL, device="cpu", compute_type=ASR_COMPUTE_TYPE)
        return _model


def transcribe(wav_path: str, duration: float, progress: ProgressFn) -> list[Word]:
    if MOCK_MODE:
        return []
    progress(0.0, "載入語音辨識模型中")
    model = _load_model()
    segments, _info = model.transcribe(
        wav_path,
        language=ASR_LANGUAGE or None,
        word_timestamps=True,
        vad_filter=True,
        initial_prompt=ASR_INITIAL_PROMPT or None,
    )
    words: list[Word] = []
    # ``segments`` is a lazy generator: the file is decoded window by window,
    # which keeps memory flat for long meetings and gives real progress.
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(float(w.start), float(w.end), _to_traditional(w.word)))
        progress(min(0.99, seg.end / max(duration, 0.01)), "語音轉文字中")
    progress(1.0, "語音轉文字完成")
    return words


def mock_words(segments: list[Segment]) -> list[Word]:
    """Placeholder text for demo mode so the refined view shows something."""
    words = []
    for i, seg in enumerate(segments, start=1):
        words.append(Word(seg.start, seg.end, f"（示範模式第 {i} 段，未執行語音辨識）"))
    return words
