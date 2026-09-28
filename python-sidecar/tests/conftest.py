import os
import sys
import tempfile
from pathlib import Path

# Must run before any sidecar module is imported.
os.environ["DIARIZATION_MOCK"] = "1"
os.environ["DIARIZATION_DATA_DIR"] = tempfile.mkdtemp(prefix="diar-test-")
os.environ.pop("HF_TOKEN", None)
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402


def make_bursts(pattern, rate=16000):
    """pattern: list of (seconds, is_speech). Speech is a noisy tone."""
    rng = np.random.default_rng(0)
    parts = []
    for seconds, speech in pattern:
        n = int(seconds * rate)
        if speech:
            t = np.arange(n) / rate
            parts.append(0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n))
        else:
            parts.append(np.zeros(n))
    return np.concatenate(parts).astype(np.float32)


TWO_SPEAKER_PATTERN = [(1.5, True), (1.0, False), (1.5, True), (1.0, False), (1.5, True)]


@pytest.fixture
def two_speaker_wav(tmp_path):
    path = tmp_path / "meeting.wav"
    sf.write(path, make_bursts(TWO_SPEAKER_PATTERN), 16000, subtype="PCM_16")
    return path
