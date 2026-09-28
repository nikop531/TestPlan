import { useEffect, useRef } from "react";
import { formatTime, type LiveSegment } from "../api";
import type { Phase } from "../appState";
import { SpeakerChip } from "./SpeakerChip";

interface Props {
  phase: Phase;
  segments: LiveSegment[];
  recordedSeconds: number;
  lagSeconds: number;
  note: string | null;
  latency: number;
}

export function LivePanel({ phase, segments, recordedSeconds, lagSeconds, note, latency }: Props) {
  const listRef = useRef<HTMLOListElement>(null);
  const recording = phase === "RECORDING_REALTIME";

  useEffect(() => {
    const el = listRef.current;
    if (el && recording) el.scrollTop = el.scrollHeight;
  }, [segments.length, recording]);

  return (
    <section className="panel" aria-labelledby="live-title">
      <header className="panel-head">
        <h2 id="live-title">即時辨識</h2>
        <span className="panel-sub">延遲約 {latency} 秒</span>
      </header>
      {recording && (
        <div className="live-meta">
          <span className="rec-dot" aria-hidden="true" />
          錄音中 {formatTime(recordedSeconds)}
          {lagSeconds > 2 && <span className="lag">處理落後 {lagSeconds.toFixed(0)} 秒</span>}
        </div>
      )}
      {note && <p className="notice">{note}</p>}
      {segments.length === 0 ? (
        <p className="empty">
          {recording ? "請開始說話，說話者會即時出現在這裡。" : "按下「開始錄音」後，這裡會即時顯示誰在說話。"}
        </p>
      ) : (
        <ol className="rows" ref={listRef}>
          {segments.map((s) => (
            <li key={s.id} className="row">
              <SpeakerChip speaker={s.speaker} />
              <span className="time">
                {formatTime(s.start)} 到 {formatTime(s.end)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
