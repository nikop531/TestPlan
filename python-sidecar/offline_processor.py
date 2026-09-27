"""Track B: offline refinement job queue.

Jobs run one at a time on a worker thread so a MacBook Air never holds two
large models busy at once. Each job:

1. converts the input to 16 kHz mono WAV (block by block),
2. diarizes it with the chosen backend (30 s latency preset),
3. transcribes it with faster-whisper unless the backend already did,
4. aligns words to speakers and writes final_transcript.json and .srt.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import asr
from alignment import align_words, relabel_by_first_appearance, segments_without_text, to_srt
from audio_capture import audio_duration, convert_to_16k_mono
from backends import get_backend
from config import MOCK_MODE, RESULTS_DIR
from settings_store import load_settings

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    file_path: str
    backend: str
    state: str = "queued"  # queued, running, done, error
    progress: float = 0.0
    stage: str = "排隊中"
    result_path: str | None = None
    srt_path: str | None = None
    error: str | None = None
    duration: float | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def public(self) -> dict:
        return asdict(self)


class JobManager:
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="offline-worker", daemon=True)
        self._worker.start()

    def submit(self, file_path: str, backend: str) -> Job:
        if backend not in ("local", "hf"):
            raise ValueError("backend must be 'local' or 'hf'")
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(file_path)
        job = Job(id=uuid.uuid4().hex[:12], file_path=str(path), backend=backend)
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                self._process(job)
            except Exception as exc:
                log.exception("job %s failed", job.id)
                job.state = "error"
                job.error = str(exc) or exc.__class__.__name__
                job.stage = "處理失敗"
            finally:
                job.finished_at = time.time()

    def _set(self, job: Job, lo: float, hi: float, fraction: float, stage: str) -> None:
        job.progress = round(lo + (hi - lo) * max(0.0, min(1.0, fraction)), 3)
        job.stage = stage

    def _process(self, job: Job) -> None:
        job.state = "running"
        out_dir = self.results_dir / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        settings = load_settings()

        self._set(job, 0, 0.05, 0, "轉換音訊格式中")
        wav = convert_to_16k_mono(job.file_path, out_dir / "audio_16k.wav")
        job.duration = round(audio_duration(wav), 2)

        try:
            run_asr = settings.get("run_asr", True)
            diar_hi = 0.5 if run_asr else 0.95
            backend = get_backend(job.backend)
            result = backend.diarize(
                str(wav), job.duration, lambda f, s: self._set(job, 0.05, diar_hi, f, s)
            )
            segments = relabel_by_first_appearance(result.segments)

            words = result.words
            if words is None and run_asr:
                if MOCK_MODE:
                    words = asr.mock_words(segments)
                else:
                    words = asr.transcribe(str(wav), job.duration, lambda f, s: self._set(job, 0.5, 0.95, f, s))
        finally:
            # The converted copy is only needed while processing.
            wav.unlink(missing_ok=True)

        self._set(job, 0.95, 1.0, 0.5, "整理逐字稿中")
        rows = align_words(words, segments) if words else segments_without_text(segments)

        result_path = out_dir / "final_transcript.json"
        result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
        srt_path = out_dir / "final_transcript.srt"
        srt_path.write_text(to_srt(rows), "utf-8")
        meta = {
            "source": job.file_path,
            "backend": job.backend,
            "duration": job.duration,
            "speakers": sorted({r["speaker"] for r in rows}),
            **result.info,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

        job.result_path = str(result_path)
        job.srt_path = str(srt_path)
        job.progress = 1.0
        job.stage = "完成"
        job.state = "done"
