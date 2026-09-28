import { api, formatTime, isTauri, type TranscriptRow } from "../api";
import type { Phase } from "../appState";
import { SpeakerChip } from "./SpeakerChip";

interface Props {
  phase: Phase;
  rows: TranscriptRow[];
  jobId: string | null;
  sourceName: string | null;
  stage: string;
}

export function RefinedPanel({ phase, rows, jobId, sourceName, stage }: Props) {
  const speakers = new Set(rows.map((r) => r.speaker)).size;

  return (
    <section className="panel" aria-labelledby="refined-title">
      <header className="panel-head">
        <h2 id="refined-title">精修逐字稿</h2>
        {phase === "DONE" && jobId && (
          <div className="actions">
            {isTauri() ? (
              <button className="btn small" onClick={() => api.reveal(jobId)}>
                在 Finder 中顯示
              </button>
            ) : (
              <>
                <a className="btn small" href={api.exportUrl(jobId, "json")} download>
                  下載 JSON
                </a>
                <a className="btn small" href={api.exportUrl(jobId, "srt")} download>
                  下載 SRT
                </a>
              </>
            )}
          </div>
        )}
      </header>

      {phase === "PROCESSING_OFFLINE" && (
        <div className="processing">
          <p>
            正在精修 {sourceName ?? "錄音"}：{stage}
          </p>
          <p className="hint">完成後會自動顯示，並校正左側即時結果的誤差。</p>
        </div>
      )}

      {phase === "DONE" && (
        <p className="summary">
          {sourceName}，共 {speakers} 位說話者，{rows.length} 段。
        </p>
      )}

      {rows.length > 0 ? (
        <ol className="rows transcript">
          {rows.map((r, i) => (
            <li key={i} className="row">
              <div className="row-head">
                <SpeakerChip speaker={r.speaker} />
                <span className="time">
                  {formatTime(r.start)} 到 {formatTime(r.end)}
                </span>
              </div>
              {r.text && <p className="text">{r.text}</p>}
            </li>
          ))}
        </ol>
      ) : (
        phase !== "PROCESSING_OFFLINE" && (
          <p className="empty">
            停止錄音或拖入錄音檔後，這裡會顯示高準確度的逐字稿。
          </p>
        )
      )}
    </section>
  );
}
