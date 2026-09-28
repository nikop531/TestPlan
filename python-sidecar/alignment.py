"""Pure functions that turn diarization segments and ASR words into a transcript.

No ML imports here, so everything is unit tested without models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


@dataclass
class Word:
    start: float
    end: float
    text: str


def speaker_label(index: int) -> str:
    return f"SPEAKER_{index:02d}"


_SPEAKER_RE = re.compile(r"(?:speaker|spk)[_\s-]*(\d+)", re.IGNORECASE)


def normalize_speaker(raw: object) -> str:
    """Map "speaker_1", "SPEAKER_01", 1, "spk1" and similar to "SPEAKER_01"."""
    if isinstance(raw, int):
        return speaker_label(raw)
    text = str(raw).strip()
    if text.isdigit():
        return speaker_label(int(text))
    match = _SPEAKER_RE.search(text)
    if match:
        return speaker_label(int(match.group(1)))
    return text


def parse_diarization_lines(lines: Iterable[str]) -> list[Segment]:
    """Parse NeMo Sortformer output lines "begin end speaker_N"."""
    segments = []
    for line in lines:
        parts = str(line).split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if end <= start:
            continue
        segments.append(Segment(start, end, normalize_speaker(parts[2])))
    return sort_segments(segments)


def parse_segment_dicts(items: Iterable[dict]) -> list[Segment]:
    """Parse [{start, end, speaker}, ...] as returned by remote backends."""
    segments = []
    for item in items:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        segments.append(Segment(start, end, normalize_speaker(item.get("speaker", "SPEAKER_00"))))
    return sort_segments(segments)


def sort_segments(segments: list[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: (s.start, s.end))


def relabel_by_first_appearance(segments: Sequence[Segment]) -> list[Segment]:
    """Renumber speakers so the first person to talk is SPEAKER_00."""
    mapping: dict[str, str] = {}
    out = []
    for seg in sort_segments(list(segments)):
        if seg.speaker not in mapping:
            mapping[seg.speaker] = speaker_label(len(mapping))
        out.append(Segment(seg.start, seg.end, mapping[seg.speaker]))
    return out


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(word: Word, segments: Sequence[Segment]) -> str | None:
    """Speaker whose segments overlap the word the most, else the nearest one."""
    if not segments:
        return None
    totals: dict[str, float] = {}
    for seg in segments:
        if seg.start > word.end:
            break
        ov = _overlap(word.start, word.end, seg.start, seg.end)
        if ov > 0:
            totals[seg.speaker] = totals.get(seg.speaker, 0.0) + ov
    if totals:
        return max(totals.items(), key=lambda kv: kv[1])[0]
    mid = (word.start + word.end) / 2
    nearest = min(segments, key=lambda s: min(abs(mid - s.start), abs(mid - s.end)))
    return nearest.speaker


def _join(texts: Sequence[str]) -> str:
    # Whisper words keep their own leading spaces (English) or have none (CJK),
    # so plain concatenation is correct for both.
    return "".join(texts).strip()


def align_words(
    words: Sequence[Word],
    segments: Sequence[Segment],
    max_gap: float = 1.5,
) -> list[dict]:
    """Group words into utterances, one speaker each.

    A new utterance starts when the speaker changes or when the silence
    between two words is longer than ``max_gap`` seconds.
    """
    segs = sort_segments(list(segments))
    utterances: list[dict] = []
    current: dict | None = None
    current_texts: list[str] = []

    def flush() -> None:
        if current is not None:
            text = _join(current_texts)
            if text:
                current["text"] = text
                utterances.append(current)

    for word in sorted(words, key=lambda w: w.start):
        speaker = assign_speaker(word, segs) or "SPEAKER_00"
        new_turn = (
            current is None
            or speaker != current["speaker"]
            or word.start - current["end"] > max_gap
        )
        if new_turn:
            flush()
            current = {"start": round(word.start, 2), "end": round(word.end, 2), "speaker": speaker, "text": ""}
            current_texts = [word.text]
        else:
            current["end"] = round(max(current["end"], word.end), 2)
            current_texts.append(word.text)
    flush()
    return utterances


def segments_without_text(segments: Sequence[Segment], merge_gap: float = 0.3) -> list[dict]:
    """Transcript rows when no ASR ran: merged diarization turns, empty text."""
    rows: list[dict] = []
    for seg in sort_segments(list(segments)):
        if rows and rows[-1]["speaker"] == seg.speaker and seg.start - rows[-1]["end"] <= merge_gap:
            rows[-1]["end"] = round(max(rows[-1]["end"], seg.end), 2)
            continue
        rows.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "speaker": seg.speaker, "text": ""})
    return rows


def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(rows: Sequence[dict]) -> str:
    blocks = []
    for i, row in enumerate(rows, start=1):
        text = row.get("text") or ""
        line = f"[{row['speaker']}] {text}".rstrip()
        blocks.append(f"{i}\n{_srt_time(row['start'])} --> {_srt_time(row['end'])}\n{line}\n")
    return "\n".join(blocks)
