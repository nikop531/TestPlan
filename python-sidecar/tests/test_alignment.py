from alignment import (
    Segment,
    Word,
    align_words,
    normalize_speaker,
    parse_diarization_lines,
    relabel_by_first_appearance,
    segments_without_text,
    to_srt,
)
from backends.hf_backend import parse_response


def test_parse_nemo_lines():
    segs = parse_diarization_lines(["3.20 5.00 speaker_1", "0.00 3.10 speaker_0", "bad line", "4 4 speaker_0"])
    assert [(s.start, s.end, s.speaker) for s in segs] == [(0.0, 3.1, "SPEAKER_00"), (3.2, 5.0, "SPEAKER_01")]


def test_normalize_speaker_variants():
    assert normalize_speaker("speaker_3") == "SPEAKER_03"
    assert normalize_speaker("SPEAKER_01") == "SPEAKER_01"
    assert normalize_speaker(2) == "SPEAKER_02"
    assert normalize_speaker("spk1") == "SPEAKER_01"


def test_relabel_first_speaker_is_zero():
    segs = relabel_by_first_appearance([Segment(5, 6, "SPEAKER_00"), Segment(0, 2, "SPEAKER_02")])
    assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_01"]


def test_align_chinese_words_to_speakers():
    segs = [Segment(192.0, 198.5, "SPEAKER_00"), Segment(198.5, 202.1, "SPEAKER_01")]
    words = [
        Word(192.1, 193.0, "我們"),
        Word(193.0, 194.0, "先確認"),
        Word(194.0, 195.0, "預算"),
        Word(198.6, 199.5, "我這邊"),
        Word(199.5, 200.4, "沒問題"),
    ]
    rows = align_words(words, segs)
    assert rows == [
        {"start": 192.1, "end": 195.0, "speaker": "SPEAKER_00", "text": "我們先確認預算"},
        {"start": 198.6, "end": 200.4, "speaker": "SPEAKER_01", "text": "我這邊沒問題"},
    ]


def test_align_english_keeps_spaces_and_splits_on_long_gap():
    segs = [Segment(0, 20, "SPEAKER_00")]
    words = [Word(0, 0.5, " Hello"), Word(0.5, 1, " world"), Word(5, 5.5, " again")]
    rows = align_words(words, segs, max_gap=1.5)
    assert [r["text"] for r in rows] == ["Hello world", "again"]


def test_word_outside_segments_goes_to_nearest():
    segs = [Segment(0, 1, "SPEAKER_00"), Segment(10, 11, "SPEAKER_01")]
    rows = align_words([Word(9.0, 9.5, "嗨")], segs)
    assert rows[0]["speaker"] == "SPEAKER_01"


def test_segments_without_text_merges_short_gaps():
    rows = segments_without_text(
        [Segment(0, 1, "SPEAKER_00"), Segment(1.1, 2, "SPEAKER_00"), Segment(2, 3, "SPEAKER_01")]
    )
    assert [(r["start"], r["end"], r["speaker"]) for r in rows] == [(0, 2, "SPEAKER_00"), (2, 3, "SPEAKER_01")]


def test_srt_format():
    srt = to_srt([{"start": 3661.5, "end": 3662.25, "speaker": "SPEAKER_00", "text": "你好"}])
    assert srt == "1\n01:01:01,500 --> 01:01:02,250\n[SPEAKER_00] 你好\n"


def test_hf_response_shapes():
    assert len(parse_response([{"start": 0, "end": 1, "speaker": "speaker_0"}]).segments) == 1
    assert len(parse_response(["0 1 speaker_0", "1 2 speaker_1"]).segments) == 2
    assert len(parse_response([["0 1 speaker_0"]]).segments) == 1
    res = parse_response(
        {"segments": [{"start": 0, "end": 1, "speaker": 0}], "words": [{"start": 0, "end": 0.5, "text": "好"}]}
    )
    assert res.segments[0].speaker == "SPEAKER_00"
    assert res.words[0].text == "好"
