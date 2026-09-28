"""Custom handler for a Hugging Face Inference Endpoint.

Deploy this folder (handler.py + requirements.txt) as a custom Inference
Endpoint to give the app's "Hugging Face" backend a guaranteed contract:

    POST <endpoint url>   body: raw audio bytes (WAV)   header: Content-Type: audio/wav
    -> {"segments": [{"start", "end", "speaker"}], "words": [{"start", "end", "text"}]}

The endpoint runs Nemotron-3-Diarization with the 30 s offline preset and
faster-whisper for the transcript, so the Mac only uploads and waits.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

MODEL_ID = os.environ.get("DIARIZATION_MODEL", "nvidia/Nemotron-3-Diarization")
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")

# Same values as OFFLINE_PRESET in python-sidecar/config.py.
OFFLINE_PRESET = dict(
    chunk_len=340, chunk_left_context=1, chunk_right_context=40, fifo_len=40, spkcache_update_period=300, spkcache_len=188
)


class EndpointHandler:
    def __init__(self, path: str = ""):
        import torch
        from faster_whisper import WhisperModel
        from nemo.collections.asr.models import SortformerEncLabelModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diar = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location=device)
        for key, value in OFFLINE_PRESET.items():
            setattr(self.diar.sortformer_modules, key, value)
        self.diar.streaming_mode = True
        self.diar.eval()
        self.asr = WhisperModel(ASR_MODEL, device=device, compute_type="float16" if device == "cuda" else "int8")

    def _to_wav16k(self, raw: bytes) -> str:
        src = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
        src.write(raw)
        src.close()
        dst = src.name + ".wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src.name, "-ac", "1", "-ar", "16000", dst], check=True
        )
        os.unlink(src.name)
        return dst

    def __call__(self, data: dict[str, Any]) -> dict:
        raw = data.get("inputs", data)
        params = data.get("parameters") or {}
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("send the audio file as the raw request body")
        wav = self._to_wav16k(bytes(raw))
        try:
            lines = self.diar.diarize(audio=[wav], batch_size=1, verbose=False)[0]
            segments = []
            for line in lines:
                start, end, speaker = line.split()[:3]
                segments.append({"start": float(start), "end": float(end), "speaker": speaker})
            words = []
            if params.get("asr", True):
                result, _ = self.asr.transcribe(
                    wav, language=params.get("language", "zh"), word_timestamps=True, vad_filter=True
                )
                for seg in result:
                    for w in seg.words or []:
                        words.append({"start": w.start, "end": w.end, "text": w.word})
            return {"segments": segments, "words": words}
        finally:
            os.unlink(wav)
