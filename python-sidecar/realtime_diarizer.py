"""Track A: realtime streaming diarization (0.32 s latency preset).

Audio arrives in blocks of any size. ``RealtimeSession`` slices it into model
chunks, runs one streaming step per chunk, and turns frame level speaker
probabilities into growing segments that the UI can render immediately.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from alignment import speaker_label
from config import FRAME_SECONDS, MOCK_MODE, REALTIME_PRESET, SAMPLE_RATE, StreamingPreset

log = logging.getLogger(__name__)

ACTIVITY_THRESHOLD = 0.5
MAX_SILENCE_JOIN = 0.5  # seconds of silence that still count as the same turn


@dataclass
class LiveSegment:
    id: int
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
        }


@dataclass
class SegmentTracker:
    """Turns per frame probabilities into speaker turns.

    ``update`` returns only the segments that changed, so the UI can replace
    them by id instead of re-rendering everything.
    """

    frame_seconds: float = FRAME_SECONDS
    threshold: float = ACTIVITY_THRESHOLD
    segments: list[LiveSegment] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=itertools.count)

    def update(self, probs: np.ndarray, start_time: float) -> list[dict]:
        changed: dict[int, LiveSegment] = {}
        for i, frame in enumerate(np.asarray(probs)):
            if frame.size == 0:
                continue
            t0 = start_time + i * self.frame_seconds
            t1 = t0 + self.frame_seconds
            idx = int(np.argmax(frame))
            if frame[idx] < self.threshold:
                continue
            speaker = speaker_label(idx)
            last = self.segments[-1] if self.segments else None
            if last and last.speaker == speaker and t0 - last.end <= MAX_SILENCE_JOIN:
                last.end = t1
                changed[last.id] = last
            else:
                seg = LiveSegment(next(self._ids), t0, t1, speaker)
                self.segments.append(seg)
                changed[seg.id] = seg
        return [s.to_dict() for s in changed.values()]


class _FeatureBuffer:
    """Rolling log-mel buffer for streaming steps.

    Adapted from NeMo's voice agent ``CacheFeatureBufferer`` (Apache 2.0),
    which is not importable on its own because its package pulls in pipecat.
    """

    LOG_MEL_ZERO = -16.635

    def __init__(self, model, buffer_seconds: float, chunk_seconds: float, device: str):
        import torch

        self.torch = torch
        self.device = device
        pre_cfg = model.cfg.preprocessor
        self.preprocessor = model.preprocessor
        self.step = float(pre_cfg.window_stride)
        self.look_back = int(self.step * SAMPLE_RATE)
        self.chunk_samples = int(round(chunk_seconds * SAMPLE_RATE))
        self.sample_buffer = torch.zeros(int(round(buffer_seconds * SAMPLE_RATE)), dtype=torch.float32)
        self.feature_chunk_len = int(round(chunk_seconds / self.step))
        n_feat = int(pre_cfg.features)
        self.features = torch.full(
            [n_feat, int(round(buffer_seconds / self.step))], self.LOG_MEL_ZERO, dtype=torch.float32, device=device
        )

    def update(self, audio: np.ndarray) -> None:
        torch = self.torch
        chunk = torch.from_numpy(audio.astype(np.float32))
        n = chunk.shape[0]
        self.sample_buffer[:-n] = self.sample_buffer[n:].clone()
        self.sample_buffer[-n:] = chunk
        samples = self.sample_buffer[-(self.look_back + self.chunk_samples) :].unsqueeze(0).to(self.device)
        length = torch.tensor([samples.shape[1]], device=self.device)
        feats, _ = self.preprocessor(input_signal=samples, length=length)
        feats = feats.squeeze(0)
        extra = feats.shape[1] - self.feature_chunk_len - 1
        if extra > 0:
            feats = feats[:, :-extra]
        k = self.feature_chunk_len
        self.features[:, :-k] = self.features[:, k:].clone()
        self.features[:, -k:] = feats[:, -k:]

    def batch(self):
        feats = self.features.clone().unsqueeze(0).transpose(1, 2)  # [1, time, feat]
        return feats, self.torch.tensor([feats.shape[1]], device=self.device)


class NemoStreamingEngine:
    """One streaming state over the Nemotron model, following NeMo's reference loop."""

    LEFT_OFFSET = 8  # 10 ms feature frames of context on each side
    RIGHT_OFFSET = 8

    def __init__(self, preset: StreamingPreset = REALTIME_PRESET):
        import torch

        from config import pick_device
        from nemo_loader import load_model

        self.torch = torch
        self.device = pick_device()
        self.model = load_model(preset, self.device)
        self.chunk_frames = preset.chunk_len
        chunk_seconds = preset.chunk_len * FRAME_SECONDS
        buffer_seconds = chunk_seconds + (self.LEFT_OFFSET + self.RIGHT_OFFSET) * 0.01
        self.chunk_samples = int(round(chunk_seconds * SAMPLE_RATE))
        self.features = _FeatureBuffer(self.model, buffer_seconds, chunk_seconds, self.device)
        self.state = self.model.sortformer_modules.init_streaming_state(
            batch_size=1, async_streaming=self.model.async_streaming, device=self.device
        )
        n_spk = int(self.model.cfg.get("max_num_of_spks", 4))
        self.total_preds = torch.zeros((1, 0, n_spk), device=self.device)

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """Process exactly ``chunk_samples`` samples, return [chunk_frames, n_spk] probabilities."""
        torch = self.torch
        self.features.update(chunk)
        feats, feat_len = self.features.batch()
        with torch.inference_mode():
            self.state, self.total_preds = self.model.forward_streaming_step(
                processed_signal=feats,
                processed_signal_length=feat_len,
                streaming_state=self.state,
                total_preds=self.total_preds,
                left_offset=self.LEFT_OFFSET,
                right_offset=self.RIGHT_OFFSET,
            )
        return self.total_preds[0, -self.chunk_frames :, :].float().cpu().numpy()


class MockStreamingEngine:
    """Demo engine: energy based voice activity, speaker switches after each pause.

    Lets the UI and the tests run end to end without downloading a model.
    """

    chunk_frames = REALTIME_PRESET.chunk_len

    def __init__(self, n_speakers: int = 2):
        self.chunk_samples = int(round(self.chunk_frames * FRAME_SECONDS * SAMPLE_RATE))
        self.n_speakers = n_speakers
        self.speaker = 0
        self.silent_frames = 0
        self.spoke = False

    def step(self, chunk: np.ndarray) -> np.ndarray:
        frame_len = len(chunk) // self.chunk_frames
        out = np.zeros((self.chunk_frames, self.n_speakers), dtype=np.float32)
        for i in range(self.chunk_frames):
            frame = chunk[i * frame_len : (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
            if rms > 0.01:
                if self.spoke and self.silent_frames * FRAME_SECONDS >= 0.6:
                    self.speaker = (self.speaker + 1) % self.n_speakers
                self.silent_frames = 0
                self.spoke = True
                out[i, self.speaker] = 0.9
            else:
                self.silent_frames += 1
        return out


def create_engine():
    return MockStreamingEngine() if MOCK_MODE else NemoStreamingEngine()


class RealtimeSession:
    """Buffers arbitrary sized audio and emits segment updates per model chunk."""

    def __init__(self, engine=None):
        self.engine = engine or create_engine()
        self.tracker = SegmentTracker()
        self._pending = np.zeros(0, dtype=np.float32)
        self.samples_processed = 0

    @property
    def seconds_processed(self) -> float:
        return self.samples_processed / SAMPLE_RATE

    def feed(self, samples: np.ndarray) -> list[dict]:
        """Add audio; return changed segments (possibly empty)."""
        self._pending = np.concatenate([self._pending, samples.astype(np.float32)])
        n = self.engine.chunk_samples
        updates: dict[int, dict] = {}
        while len(self._pending) >= n:
            chunk, self._pending = self._pending[:n], self._pending[n:]
            chunk_start = self.samples_processed / SAMPLE_RATE
            try:
                probs = self.engine.step(chunk)
            except Exception:
                log.exception("streaming step failed")
                probs = np.zeros((0, 0))
            self.samples_processed += n
            if probs.size:
                # The last ``chunk_frames`` predictions cover the chunk just fed.
                offset = chunk_start + (n / SAMPLE_RATE) - len(probs) * FRAME_SECONDS
                for seg in self.tracker.update(probs, max(0.0, offset)):
                    updates[seg["id"]] = seg
        return list(updates.values())

    def all_segments(self) -> list[dict]:
        return [s.to_dict() for s in self.tracker.segments]


def expected_latency_seconds() -> float:
    return REALTIME_PRESET.latency_seconds
