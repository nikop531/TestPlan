"""Runs one offline job on a long file and reports wall time and peak memory.

    python benchmark_long_audio.py                 # generates a 1 hour test file
    python benchmark_long_audio.py meeting.wav     # use your own recording
    python benchmark_long_audio.py --backend hf

The generated file is 44.1 kHz stereo on purpose, to exercise the block by
block conversion. Target on a MacBook Air: peak memory under 2 GB.
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def make_test_file(path: Path, minutes: int) -> None:
    rate = 44100
    rng = np.random.default_rng(0)
    with sf.SoundFile(str(path), "w", samplerate=rate, channels=2, subtype="PCM_16") as out:
        for i in range(minutes * 6):  # 10 second blocks: 7 s of "speech", 3 s of silence
            t = np.arange(rate * 7) / rate
            tone = 0.2 * np.sin(2 * np.pi * (180 + 60 * (i % 2)) * t) + 0.02 * rng.standard_normal(t.size)
            block = np.concatenate([tone, np.zeros(rate * 3)]).astype(np.float32)
            out.write(np.stack([block, block], axis=1))


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    parser.add_argument("--backend", default="local", choices=["local", "hf"])
    parser.add_argument("--minutes", type=int, default=60)
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        path = Path(tempfile.mkdtemp()) / "benchmark.wav"
        print(f"產生 {args.minutes} 分鐘測試音檔：{path}")
        make_test_file(path, args.minutes)

    from offline_processor import JobManager

    manager = JobManager()
    start = time.time()
    job = manager.submit(str(path), args.backend)
    last = ""
    while job.state not in ("done", "error"):
        line = f"{job.progress * 100:5.1f}%  {job.stage}"
        if line != last:
            print(line, flush=True)
            last = line
        time.sleep(1)
    elapsed = time.time() - start

    print(f"\n狀態：{job.state} {job.error or ''}")
    print(f"音檔長度：{job.duration} 秒")
    print(f"處理時間：{elapsed:.1f} 秒")
    print(f"記憶體峰值：{peak_rss_mb():.0f} MB（目標 2048 MB 以下）")
    print(f"結果：{job.result_path}")
    sys.exit(0 if job.state == "done" else 1)


if __name__ == "__main__":
    main()
