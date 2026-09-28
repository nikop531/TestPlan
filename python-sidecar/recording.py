"""Recording sessions: one audio source feeding both tracks.

Audio (from the microphone or from a WebSocket client) goes into a queue. A
worker thread writes it to the session WAV (Track B input) and feeds the
realtime diarizer (Track A), then broadcasts segment updates to listeners.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from audio_capture import MicRecorder, WavWriter
from config import RECORDINGS_DIR, SAMPLE_RATE
from realtime_diarizer import RealtimeSession, create_engine

log = logging.getLogger(__name__)


class Listener:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue[dict] = asyncio.Queue()

    def send(self, message: dict) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, message)


class RecordingSession:
    def __init__(self, source: str = "mic", device: int | None = None, engine_factory=create_engine):
        self.id = uuid.uuid4().hex[:12]
        self.source = source
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.wav_path: Path = RECORDINGS_DIR / f"meeting_{stamp}_{self.id}.wav"
        self.writer = WavWriter(self.wav_path)
        self.state = "loading_model"
        self.error: str | None = None
        self._engine_factory = engine_factory
        self._rt: RealtimeSession | None = None
        self._inbox: queue.Queue[np.ndarray | None] = queue.Queue()
        self._listeners: set[Listener] = set()
        self._listeners_lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name=f"rec-{self.id}", daemon=True)
        self._mic = MicRecorder(self.push, device=device) if source == "mic" else None

    # lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._worker.start()
        if self._mic is not None:
            self._mic.start()

    def stop(self) -> dict:
        if self._mic is not None:
            self._mic.stop()
        self._inbox.put(None)
        # Let Track A finish what is queued so the live view is complete.
        self._worker.join(timeout=120)
        self.writer.close()
        self.state = "stopped"
        info = {
            "session_id": self.id,
            "wav_path": str(self.wav_path),
            "duration": round(self.writer.frames_written / SAMPLE_RATE, 2),
        }
        self._broadcast({"type": "stopped", **info})
        return info

    # audio in ------------------------------------------------------------
    def push(self, samples: np.ndarray) -> None:
        self._inbox.put(samples)

    def _run(self) -> None:
        try:
            self._rt = RealtimeSession(self._engine_factory())
            self.state = "live"
        except Exception as exc:
            log.exception("realtime engine failed to load")
            self.error = f"即時辨識模型載入失敗：{exc}"
            self.state = "recording_only"
        self._broadcast(self.status())

        last_status = time.monotonic()
        while True:
            block = self._inbox.get()
            if block is None:
                return
            self.writer.write(block)
            if self._rt is None:
                continue
            updates = self._rt.feed(block)
            if updates:
                self._broadcast({"type": "segments", "segments": updates})
            if time.monotonic() - last_status > 1.0:
                last_status = time.monotonic()
                self._broadcast(self.status())

    # listeners -----------------------------------------------------------
    def status(self) -> dict:
        recorded = self.writer.frames_written / SAMPLE_RATE
        processed = self._rt.seconds_processed if self._rt else 0.0
        return {
            "type": "status",
            "state": self.state,
            "recorded_seconds": round(recorded, 2),
            "lag_seconds": round(max(0.0, recorded - processed), 2),
            "error": self.error,
        }

    def snapshot(self) -> list[dict]:
        return self._rt.all_segments() if self._rt else []

    def add_listener(self, listener: Listener) -> None:
        with self._listeners_lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._listeners_lock:
            self._listeners.discard(listener)

    def _broadcast(self, message: dict) -> None:
        with self._listeners_lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.send(message)


class RecordingManager:
    def __init__(self):
        self.sessions: dict[str, RecordingSession] = {}
        self.active: RecordingSession | None = None
        self._lock = threading.Lock()

    def start(self, source: str, device: int | None) -> RecordingSession:
        with self._lock:
            if self.active is not None:
                raise RuntimeError("已經在錄音中。")
            session = RecordingSession(source=source, device=device)
            try:
                session.start()
            except Exception as exc:
                session._inbox.put(None)
                session.writer.close()
                session.wav_path.unlink(missing_ok=True)
                log.exception("could not open microphone")
                raise RuntimeError("無法開啟麥克風。請到「系統設定 > 隱私權與安全性 > 麥克風」允許本程式使用。") from exc
            self.sessions[session.id] = session
            self.active = session
            return session

    def stop(self, session_id: str | None) -> dict:
        with self._lock:
            session = self.sessions.get(session_id) if session_id else self.active
            if session is None:
                raise KeyError("找不到錄音工作階段。")
            if session is self.active:
                self.active = None
        return session.stop()
