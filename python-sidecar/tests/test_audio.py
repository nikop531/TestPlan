import numpy as np
import soundfile as sf

from audio_capture import audio_duration, convert_to_16k_mono, float_to_pcm16, pcm16_to_float


def test_convert_resamples_stereo_44k(tmp_path):
    src = tmp_path / "in.wav"
    stereo = np.zeros((44100 * 3, 2), dtype=np.float32)
    stereo[:, 0] = 0.1
    sf.write(src, stereo, 44100)
    out = convert_to_16k_mono(src, tmp_path / "out.wav")
    info = sf.info(str(out))
    assert info.samplerate == 16000 and info.channels == 1
    assert abs(audio_duration(out) - 3.0) < 0.01


def test_convert_copies_when_already_16k_mono(two_speaker_wav, tmp_path):
    out = convert_to_16k_mono(two_speaker_wav, tmp_path / "copy.wav")
    assert out.read_bytes() == two_speaker_wav.read_bytes()


def test_pcm_roundtrip():
    x = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    assert np.allclose(pcm16_to_float(float_to_pcm16(x)), x, atol=1e-4)
