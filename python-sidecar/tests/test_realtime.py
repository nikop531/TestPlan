import numpy as np

from conftest import TWO_SPEAKER_PATTERN, make_bursts
from realtime_diarizer import MockStreamingEngine, RealtimeSession, SegmentTracker


def test_tracker_extends_and_switches():
    tracker = SegmentTracker(frame_seconds=0.08)
    probs = np.array([[0.9, 0.1], [0.8, 0.1], [0.1, 0.1], [0.1, 0.9]])
    updates = tracker.update(probs, 10.0)
    assert [(u["speaker"], u["start"], u["end"]) for u in updates] == [
        ("SPEAKER_00", 10.0, 10.16),
        ("SPEAKER_01", 10.24, 10.32),
    ]
    # A later chunk from the same speaker extends the same segment id.
    more = tracker.update(np.array([[0.1, 0.9]]), 10.32)
    assert more[0]["id"] == updates[1]["id"] and more[0]["end"] == 10.4


def test_session_accepts_odd_block_sizes_and_alternates_speakers():
    audio = make_bursts(TWO_SPEAKER_PATTERN)
    session = RealtimeSession(MockStreamingEngine())
    for i in range(0, len(audio), 1234):
        session.feed(audio[i : i + 1234])
    segs = session.all_segments()
    assert [s["speaker"] for s in segs] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert abs(segs[1]["start"] - 2.5) < 0.2
