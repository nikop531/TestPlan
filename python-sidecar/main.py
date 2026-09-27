"""FastAPI sidecar entry point.

Run with ``python main.py``. Listens on 127.0.0.1 only; nothing is exposed to
the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import platform
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from audio_capture import list_input_devices, pcm16_to_float
from offline_processor import JobManager
from recording import Listener, RecordingManager
from settings_store import get_hf_token, load_settings, save_settings, set_hf_token

VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sidecar")

config.ensure_dirs()


def _preload_realtime_model() -> None:
    if config.MOCK_MODE:
        return
    try:
        from nemo_loader import load_model

        load_model(config.REALTIME_PRESET)
    except Exception:
        log.exception("realtime model preload failed; it will be retried on record")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_preload_realtime_model, name="preload", daemon=True).start()
    yield


app = FastAPI(title="Dual-Track Diarization Sidecar", version=VERSION, lifespan=lifespan)
recordings = RecordingManager()
jobs = JobManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
        "http://localhost:1420",
        "http://127.0.0.1:1420",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- system ---------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    ready = True
    if not config.MOCK_MODE:
        from nemo_loader import is_loaded

        ready = is_loaded(config.REALTIME_PRESET)
    return {
        "ok": True,
        "version": VERSION,
        "mock": config.MOCK_MODE,
        "model": config.MODEL_ID,
        "device": "mock" if config.MOCK_MODE else config.pick_device(),
        "realtime_model_ready": ready,
        "realtime_latency": config.REALTIME_PRESET.latency_seconds,
        "offline_latency": config.OFFLINE_PRESET.latency_seconds,
    }


@app.get("/system")
def system() -> dict:
    on_battery = None
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is not None:
            on_battery = not battery.power_plugged
    except Exception:
        pass
    return {"on_battery": on_battery, "recommended_backend": "hf" if on_battery else "local"}


@app.get("/devices")
def devices() -> list[dict]:
    return list_input_devices()


# --- settings -------------------------------------------------------------


class SettingsIn(BaseModel):
    backend: str | None = None
    hf_endpoint: str | None = None
    hf_upload_consent: bool | None = None
    run_asr: bool | None = None


class TokenIn(BaseModel):
    token: str | None = None


def _settings_out() -> dict:
    return {**load_settings(), "has_hf_token": bool(get_hf_token())}


@app.get("/settings")
def get_settings() -> dict:
    return _settings_out()


@app.post("/settings")
def post_settings(body: SettingsIn) -> dict:
    if body.backend is not None and body.backend not in ("local", "hf"):
        raise HTTPException(400, "backend must be 'local' or 'hf'")
    save_settings(body.model_dump())
    return _settings_out()


@app.post("/settings/hf_token")
def post_token(body: TokenIn) -> dict:
    try:
        set_hf_token(body.token)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return _settings_out()


# --- Track A: recording and live stream -------------------------------------


class StartIn(BaseModel):
    source: str = "mic"  # "mic" records here; "client" expects PCM16 over the WebSocket
    device: int | None = None


class StopIn(BaseModel):
    session_id: str | None = None


@app.post("/start")
def start(body: StartIn | None = None) -> dict:
    body = body or StartIn()
    if body.source not in ("mic", "client"):
        raise HTTPException(400, "source must be 'mic' or 'client'")
    try:
        session = recordings.start(body.source, body.device)
    except RuntimeError as exc:
        raise HTTPException(409 if "錄音中" in str(exc) else 500, str(exc)) from exc
    return {"session_id": session.id, "wav_path": str(session.wav_path)}


@app.post("/stop")
def stop(body: StopIn | None = None) -> dict:
    body = body or StopIn()
    try:
        return recordings.stop(body.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.websocket("/stream")
async def stream(websocket: WebSocket, session_id: str = Query(...)) -> None:
    session = recordings.sessions.get(session_id)
    await websocket.accept()
    if session is None:
        await websocket.send_json({"type": "error", "message": "找不到錄音工作階段。"})
        await websocket.close(code=4404)
        return

    listener = Listener(asyncio.get_running_loop())
    session.add_listener(listener)
    await websocket.send_json({"type": "snapshot", "segments": session.snapshot()})
    await websocket.send_json(session.status())

    async def pump_out() -> None:
        while True:
            message = await listener.queue.get()
            await websocket.send_json(message)
            if message.get("type") == "stopped":
                return

    async def pump_in() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data and session.source == "client":
                session.push(pcm16_to_float(data))

    tasks = [asyncio.create_task(pump_out()), asyncio.create_task(pump_in())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for task in tasks:
            task.cancel()
        session.remove_listener(listener)
        with contextlib.suppress(Exception):
            await websocket.close()


# --- Track B: offline refinement ---------------------------------------------


class OfflineIn(BaseModel):
    file_path: str
    backend: str | None = None


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(400, "不支援這個檔案格式，請使用 WAV 檔。")
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{Path(file.filename or 'audio').name}"
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    return {"file_path": str(dest), "name": file.filename}


@app.post("/diarize/offline")
def diarize_offline(body: OfflineIn) -> dict:
    backend = body.backend or load_settings()["backend"]
    path = Path(body.file_path).expanduser()
    if path.suffix.lower() not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(400, "不支援這個檔案格式，請使用 WAV 檔。")
    if backend == "hf" and not config.MOCK_MODE:
        settings = load_settings()
        if not get_hf_token():
            raise HTTPException(400, "尚未設定 Hugging Face 金鑰。")
        if not settings.get("hf_upload_consent"):
            raise HTTPException(400, "使用 Hugging Face 前，需先同意將音訊上傳到雲端。")
    try:
        job = jobs.submit(str(path), backend)
    except FileNotFoundError as exc:
        raise HTTPException(404, "找不到這個檔案。") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job.id}


def _job_or_404(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "找不到這個工作。")
    return job


@app.get("/status")
def status(job_id: str) -> dict:
    return _job_or_404(job_id).public()


@app.get("/result")
def result(job_id: str) -> list[dict]:
    job = _job_or_404(job_id)
    if job.state != "done" or not job.result_path:
        raise HTTPException(409, "工作尚未完成。")
    return json.loads(Path(job.result_path).read_text("utf-8"))


@app.get("/export")
def export(job_id: str, format: str = "json") -> FileResponse:
    job = _job_or_404(job_id)
    if job.state != "done":
        raise HTTPException(409, "工作尚未完成。")
    if format == "srt":
        return FileResponse(job.srt_path, media_type="application/x-subrip", filename="final_transcript.srt")
    return FileResponse(job.result_path, media_type="application/json", filename="final_transcript.json")


@app.post("/reveal")
def reveal(job_id: str) -> dict:
    """Show the result file in Finder."""
    job = _job_or_404(job_id)
    if not job.result_path:
        raise HTTPException(409, "工作尚未完成。")
    if platform.system() == "Darwin" and shutil.which("open"):
        subprocess.run(["open", "-R", job.result_path], check=False)
    return {"path": job.result_path}


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    run()
