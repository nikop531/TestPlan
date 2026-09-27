"""Audio capture module: microphone recording, WAV writing, and file import.

Everything that reaches the models is 16 kHz mono. Imported files are
converted block by block so a 1 hour 48 kHz stereo file never has to sit in
memory as a whole.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from math import gcd
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE

log = logging.getLogger(__name__)

BLOCK_SECONDS = 30


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def _is_target_format(path: Path) -> bool:
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return info.samplerate == SAMPLE_RATE and info.channels == 1 and info.format == "WAV"


def _resample_block(block: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return block
    from scipy.signal import resample_poly

    g = gcd(src_rate, SAMPLE_RATE)
    return resample_poly(block, SAMPLE_RATE // g, src_rate // g).astype(np.float32)


def _ffmpeg_to_wav(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("無法讀取這個音訊格式。請改用 WAV 檔，或先安裝 ffmpeg。")
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)],
        check=True,
    )


def convert_to_16k_mono(src: str | Path, dst: str | Path) -> Path:
    """Write ``src`` as a 16 kHz mono PCM WAV at ``dst`` and return ``dst``.

    If ``src`` already has the right format it is copied unchanged.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_target_format(src):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst
    try:
        info = sf.info(str(src))
    except Exception:
        _ffmpeg_to_wav(src, dst)
        return dst

    block_frames = BLOCK_SECONDS * info.samplerate
    with sf.SoundFile(str(dst), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as out:
        for block in sf.blocks(str(src), blocksize=block_frames, dtype="float32", always_2d=True):
            mono = block.mean(axis=1)
            out.write(_resample_block(mono, info.samplerate))
    return dst


def load_audio(path: str | Path) -> np.ndarray:
    """Load a file as float32 16 kHz mono. Meant for short clips and tests."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _resample_block(data.mean(axis=1), rate)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def list_input_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except Exception as exc:  # PortAudio missing
        log.warning("sounddevice unavailable: %s", exc)
        return []
    devices = []
    try:
        default_in = sd.default.device[0]
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({"id": idx, "name": dev["name"], "default": idx == default_in})
    except Exception as exc:
        log.warning("could not list audio devices: %s", exc)
    return devices


class WavWriter:
    """Thread safe 16 kHz mono PCM16 WAV writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(str(self.path), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16")
        self._lock = threading.Lock()
        self.frames_written = 0

    def write(self, samples: np.ndarray) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._file.write(samples)
            self.frames_written += len(samples)

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


class MicRecorder:
    """Records the microphone at 16 kHz mono and hands blocks to a callback.

    The PortAudio callback only copies data into a queue; a worker thread does
    the rest so the audio thread never blocks.
    """

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        device: int | None = None,
        block_seconds: float = 0.08,
    ):
        self.on_audio = on_audio
        self.device = device
        self.blocksize = int(SAMPLE_RATE * block_seconds)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream = None
        self._worker: threading.Thread | None = None

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status:
            log.debug("audio status: %s", status)
        self._queue.put(indata[:, 0].copy())

    def _drain(self) -> None:
        while True:
            block = self._queue.get()
            if block is None:
                return
            try:
                self.on_audio(block)
            except Exception:
                log.exception("audio consumer failed")

    def start(self) -> None:
        import sounddevice as sd

        self._worker = threading.Thread(target=self._drain, name="mic-drain", daemon=True)
        self._worker.start()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=10)
            self._worker = None
