"""Demo backend: runs the energy based mock engine over the whole file."""

from __future__ import annotations

import time

import soundfile as sf

from alignment import Segment
from realtime_diarizer import MockStreamingEngine, RealtimeSession

from .base import BackendResult, ProgressFn


class MockBackend:
    def __init__(self, label: str = "local"):
        self.name = label

    def diarize(self, wav_path: str, duration: float, progress: ProgressFn) -> BackendResult:
        session = RealtimeSession(MockStreamingEngine())
        done = 0
        for block in sf.blocks(wav_path, blocksize=16000 * 10, dtype="float32", always_2d=True):
            session.feed(block.mean(axis=1))
            done += len(block)
            progress(min(0.99, done / 16000 / max(duration, 0.01)), "辨識說話者中（示範模式）")
            time.sleep(0.01)
        segments = [Segment(s["start"], s["end"], s["speaker"]) for s in session.all_segments()]
        return BackendResult(segments=segments, info={"mock": True})
