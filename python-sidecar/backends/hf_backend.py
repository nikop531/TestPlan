"""Hugging Face backend: uploads the recording and runs diarization remotely.

The request goes to ``settings.hf_endpoint``. By default that is the model's
serverless URL; if Hugging Face does not serve this model there, deploy
``hf_endpoint/handler.py`` as an Inference Endpoint and paste its URL in the
app settings. That handler also runs ASR, so the Mac does no heavy work.

The call is a plain HTTPS POST of the WAV bytes (the same request
``huggingface_hub.InferenceClient`` sends for audio tasks), done with httpx so
the file is streamed from disk and large uploads get a long timeout.
"""

from __future__ import annotations

import logging
import time

import httpx

from alignment import Word, parse_diarization_lines, parse_segment_dicts
from config import HF_TIMEOUT_SECONDS
from settings_store import get_hf_token, load_settings

from .base import BackendResult, EstimatedProgress, ProgressFn

log = logging.getLogger(__name__)

MAX_LOADING_RETRIES = 6


class HFBackendError(RuntimeError):
    pass


def parse_response(payload) -> BackendResult:
    """Accept the shapes remote diarization services commonly return."""
    words = None
    if isinstance(payload, dict):
        raw_words = payload.get("words")
        if isinstance(raw_words, list):
            words = [
                Word(float(w["start"]), float(w["end"]), str(w.get("text", w.get("word", ""))))
                for w in raw_words
                if "start" in w and "end" in w
            ]
        for key in ("segments", "diarization", "output", "result"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return BackendResult(segments=parse_diarization_lines(payload), words=words)
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        # NeMo style [[lines for file 1]].
        return parse_response(payload[0])
    if isinstance(payload, list):
        return BackendResult(segments=parse_segment_dicts(payload), words=words)
    raise HFBackendError("無法解析 Hugging Face 回傳的結果格式。")


class HFBackend:
    name = "hf"

    def diarize(self, wav_path: str, duration: float, progress: ProgressFn) -> BackendResult:
        settings = load_settings()
        token = get_hf_token()
        if not token:
            raise HFBackendError("尚未設定 Hugging Face 金鑰，請先在設定中輸入。")
        if not settings.get("hf_upload_consent"):
            raise HFBackendError("使用 Hugging Face 前，需先同意將音訊上傳到雲端。")
        url = settings["hf_endpoint"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "audio/wav", "Accept": "application/json"}

        # Upload plus remote inference; estimate roughly 0.15x real time.
        with EstimatedProgress(progress, duration * 0.15 + 10, "上傳並於雲端辨識中"):
            response = self._post_with_retry(url, headers, wav_path)
        progress(1.0, "雲端辨識完成")
        result = parse_response(response.json())
        result.info.update({"endpoint": url})
        return result

    def _post_with_retry(self, url: str, headers: dict, wav_path: str) -> httpx.Response:
        timeout = httpx.Timeout(HF_TIMEOUT_SECONDS, connect=30)
        for attempt in range(MAX_LOADING_RETRIES):
            with open(wav_path, "rb") as fh:
                try:
                    response = httpx.post(url, content=fh, headers=headers, timeout=timeout)
                except httpx.HTTPError as exc:
                    raise HFBackendError(f"無法連線到 Hugging Face：{exc.__class__.__name__}") from exc
            if response.status_code == 503 and attempt < MAX_LOADING_RETRIES - 1:
                # Model or endpoint is still starting up.
                wait = 20
                try:
                    wait = min(60, float(response.json().get("estimated_time", wait)))
                except Exception:
                    pass
                log.info("HF endpoint loading, retrying in %.0fs", wait)
                time.sleep(wait)
                continue
            if response.status_code in (401, 403):
                raise HFBackendError("Hugging Face 金鑰無效或沒有權限。")
            if response.status_code == 404:
                raise HFBackendError("Hugging Face 找不到這個端點。請在設定中填入您的 Inference Endpoint 網址。")
            if response.status_code >= 400:
                raise HFBackendError(f"Hugging Face 回傳錯誤（{response.status_code}）。")
            return response
        raise HFBackendError("Hugging Face 端點啟動逾時，請稍後再試。")
