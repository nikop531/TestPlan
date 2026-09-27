from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from alignment import Segment, Word

ProgressFn = Callable[[float, str], None]


@dataclass
class BackendResult:
    segments: list[Segment]
    # Filled when the backend also ran ASR (for example a custom HF endpoint).
    words: list[Word] | None = None
    info: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def diarize(self, wav_path: str, duration: float, progress: ProgressFn) -> BackendResult: ...


class EstimatedProgress:
    """Ticks progress while a call without callbacks is running.

    Diarization APIs give no progress events, so progress is estimated from
    the audio duration and a real time factor, and never reaches 100% until
    the call actually returns.
    """

    def __init__(self, progress: ProgressFn, expected_seconds: float, message: str):
        self.progress = progress
        self.expected = max(expected_seconds, 1.0)
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        start = time.monotonic()
        while not self._stop.wait(0.5):
            ratio = (time.monotonic() - start) / self.expected
            self.progress(min(0.95, ratio), self.message)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False
