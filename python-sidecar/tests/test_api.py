import time

from fastapi.testclient import TestClient

from audio_capture import float_to_pcm16
from conftest import TWO_SPEAKER_PATTERN, make_bursts
from main import app

client = TestClient(app)


def wait_for_job(job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/status", params={"job_id": job_id}).json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_health_reports_mock():
    body = client.get("/health").json()
    assert body["mock"] is True and body["realtime_latency"] == 0.32 and body["offline_latency"] == 30.4


def test_upload_then_offline_job(two_speaker_wav):
    with two_speaker_wav.open("rb") as fh:
        up = client.post("/upload", files={"file": ("meeting.wav", fh, "audio/wav")}).json()
    job_id = client.post("/diarize/offline", json={"file_path": up["file_path"], "backend": "local"}).json()["job_id"]
    status = wait_for_job(job_id)
    assert status["state"] == "done", status
    rows = client.get("/result", params={"job_id": job_id}).json()
    assert [r["speaker"] for r in rows] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert all(set(r) == {"start", "end", "speaker", "text"} for r in rows)
    srt = client.get("/export", params={"job_id": job_id, "format": "srt"}).text
    assert "[SPEAKER_01]" in srt


def test_rejects_unknown_extension_and_missing_file(tmp_path):
    assert client.post("/diarize/offline", json={"file_path": str(tmp_path / "x.txt")}).status_code == 400
    assert client.post("/diarize/offline", json={"file_path": str(tmp_path / "nope.wav")}).status_code == 404


def test_realtime_stream_with_client_audio():
    session_id = client.post("/start", json={"source": "client"}).json()["session_id"]
    assert client.post("/start", json={"source": "client"}).status_code == 409

    audio = make_bursts(TWO_SPEAKER_PATTERN)
    speakers = set()
    with client.websocket_connect(f"/stream?session_id={session_id}") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        for i in range(0, len(audio), 3200):  # 0.2 s blocks, like a live mic
            ws.send_bytes(float_to_pcm16(audio[i : i + 3200]))
        deadline = time.time() + 10
        while len(speakers) < 2 and time.time() < deadline:
            msg = ws.receive_json()
            if msg["type"] == "segments":
                speakers.update(s["speaker"] for s in msg["segments"])
        stopped = client.post("/stop", json={"session_id": session_id}).json()
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}
    assert abs(stopped["duration"] - 6.5) < 0.05


def test_hf_requires_token_and_consent(monkeypatch):
    import main

    monkeypatch.setattr(main.config, "MOCK_MODE", False)
    r = client.post("/diarize/offline", json={"file_path": "/tmp/a.wav", "backend": "hf"})
    assert r.status_code == 400


def test_settings_roundtrip():
    body = client.post("/settings", json={"backend": "hf", "hf_upload_consent": True}).json()
    assert body["backend"] == "hf" and body["hf_upload_consent"] is True and body["has_hf_token"] is False
    assert client.post("/settings", json={"backend": "cloud"}).status_code == 400
    client.post("/settings", json={"backend": "local"})
