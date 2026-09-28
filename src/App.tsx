import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { api, type Backend, type Health, type Settings, type StreamMessage } from "./api";
import { initialState, reducer, type Phase } from "./appState";
import { BackendToggle } from "./components/BackendToggle";
import { FileDrop } from "./components/FileDrop";
import { LivePanel } from "./components/LivePanel";
import { RefinedPanel } from "./components/RefinedPanel";
import { SettingsDialog } from "./components/SettingsDialog";

const PHASE_LABEL: Record<Phase, string> = {
  IDLE: "待命",
  RECORDING_REALTIME: "錄音中",
  PROCESSING_OFFLINE: "精修中",
  DONE: "完成",
  ERROR: "發生錯誤",
};

function liveNote(state: string, error: string | null): string | null {
  if (state === "loading_model") return "即時辨識模型載入中，錄音已經開始，請繼續說話。";
  if (state === "recording_only")
    return `${error ?? "即時辨識無法使用"}。錄音仍在進行，停止後會自動精修。`;
  return null;
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [health, setHealth] = useState<Health | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [onBattery, setOnBattery] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Wait for the sidecar, then load settings.
  useEffect(() => {
    let stopped = false;
    let loaded = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const h = await api.health();
        if (stopped) return;
        setHealth(h);
        if (!loaded) {
          setSettings(await api.settings());
          loaded = true;
          const sys = await api.system();
          setOnBattery(sys.on_battery === true);
        }
        if (!h.realtime_model_ready) timer = window.setTimeout(poll, 3000);
      } catch {
        if (!stopped) timer = window.setTimeout(poll, 1000);
      }
    };
    poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, []);

  // Poll the offline job while it runs.
  useEffect(() => {
    if (state.phase !== "PROCESSING_OFFLINE" || !state.jobId) return;
    const jobId = state.jobId;
    let active = true;
    const tick = async () => {
      try {
        const s = await api.status(jobId);
        if (!active) return;
        if (s.state === "done") {
          const rows = await api.result(jobId);
          if (active) dispatch({ type: "OFFLINE_DONE", rows });
          return;
        }
        if (s.state === "error") {
          dispatch({ type: "FAILED", message: s.error ?? "精修失敗" });
          return;
        }
        dispatch({ type: "OFFLINE_PROGRESS", progress: s.progress, stage: s.stage });
      } catch {
        /* sidecar busy; try again */
      }
      if (active) window.setTimeout(tick, 700);
    };
    tick();
    return () => {
      active = false;
    };
  }, [state.phase, state.jobId]);

  const backend: Backend = settings?.backend ?? "local";

  const hfReady = (s: Settings | null) => !!s && s.has_hf_token && s.hf_upload_consent;

  const submitOffline = useCallback(
    async (filePath: string, sourceName: string, keepLive: boolean) => {
      if (backend === "hf" && !hfReady(settings) && !health?.mock) {
        setShowSettings(true);
        throw new Error("請先完成 Hugging Face 設定，或切換成「本機」。");
      }
      const { job_id } = await api.diarizeOffline(filePath, backend);
      dispatch({ type: "OFFLINE_STARTED", jobId: job_id, sourceName, keepLive });
    },
    [backend, settings, health],
  );

  async function startRecording() {
    setBusy(true);
    try {
      const { session_id } = await api.start();
      dispatch({ type: "RECORDING_STARTED", sessionId: session_id });
      const ws = new WebSocket(api.streamUrl(session_id));
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as StreamMessage;
        if (msg.type === "snapshot") dispatch({ type: "LIVE_SEGMENTS", segments: msg.segments, replace: true });
        else if (msg.type === "segments") dispatch({ type: "LIVE_SEGMENTS", segments: msg.segments });
        else if (msg.type === "status")
          dispatch({
            type: "LIVE_STATUS",
            recordedSeconds: msg.recorded_seconds,
            lagSeconds: msg.lag_seconds,
            note: liveNote(msg.state, msg.error),
          });
      };
    } catch (e) {
      dispatch({ type: "FAILED", message: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function stopRecording() {
    if (!state.sessionId) return;
    setBusy(true);
    try {
      const stopped = await api.stop(state.sessionId);
      wsRef.current?.close();
      wsRef.current = null;
      if (stopped.duration < 0.5) {
        dispatch({ type: "FAILED", message: "錄音太短，沒有可以精修的內容。" });
        return;
      }
      await submitOffline(stopped.wav_path, "這段錄音", true);
    } catch (e) {
      dispatch({ type: "FAILED", message: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  const handlePath = useCallback(
    async (path: string) => {
      try {
        await submitOffline(path, path.split("/").pop() ?? "錄音檔", false);
      } catch (e) {
        dispatch({ type: "FAILED", message: (e as Error).message });
      }
    },
    [submitOffline],
  );

  const handleFile = useCallback(
    async (file: File) => {
      setBusy(true);
      try {
        const { file_path } = await api.upload(file);
        await submitOffline(file_path, file.name, false);
      } catch (e) {
        dispatch({ type: "FAILED", message: (e as Error).message });
      } finally {
        setBusy(false);
      }
    },
    [submitOffline],
  );

  async function changeBackend(next: Backend) {
    try {
      const saved = await api.saveSettings({ backend: next });
      setSettings(saved);
      if (next === "hf" && !hfReady(saved)) setShowSettings(true);
    } catch (e) {
      dispatch({ type: "FAILED", message: (e as Error).message });
    }
  }

  const connected = health !== null;
  const recording = state.phase === "RECORDING_REALTIME";
  const processing = state.phase === "PROCESSING_OFFLINE";
  const canStart = connected && !recording && !processing && !busy;

  return (
    <div className="app">
      <header className="topbar">
        <h1>雙軌說話者辨識</h1>
        <div className="controls">
          <button className="btn record" onClick={startRecording} disabled={!canStart}>
            開始錄音
          </button>
          <button className="btn stop" onClick={stopRecording} disabled={!recording || busy}>
            停止
          </button>
        </div>
        <div className={`status status-${state.phase.toLowerCase()}`} role="status">
          <span className="dot" aria-hidden="true" />
          {connected ? PHASE_LABEL[state.phase] : "正在啟動辨識引擎"}
          {health?.mock && <span className="badge">示範模式</span>}
        </div>
      </header>

      {state.phase === "ERROR" && (
        <div className="banner error" role="alert">
          <span>{state.error}</span>
          <button className="btn small" onClick={() => dispatch({ type: "RESET" })}>
            知道了
          </button>
        </div>
      )}
      {onBattery && backend === "local" && state.phase === "IDLE" && (
        <div className="banner info">目前使用電池供電。長時間的會議建議改用 Hugging Face 處理，比較省電。</div>
      )}

      <main className="panels">
        <LivePanel
          phase={state.phase}
          segments={state.liveSegments}
          recordedSeconds={state.recordedSeconds}
          lagSeconds={state.lagSeconds}
          note={state.liveNote}
          latency={health?.realtime_latency ?? 0.32}
        />
        <RefinedPanel
          phase={state.phase}
          rows={state.rows}
          jobId={state.jobId}
          sourceName={state.sourceName}
          stage={state.stage}
        />
      </main>

      <footer className="bottombar">
        <BackendToggle
          value={backend}
          disabled={!settings || recording || processing}
          onChange={changeBackend}
          onOpenSettings={() => setShowSettings(true)}
        />
        <FileDrop disabled={!canStart} onPath={handlePath} onFile={handleFile} />
        <div className="progress" aria-label="精修進度">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.round(state.progress * 100)}%` }} />
          </div>
          <span className="progress-text">
            {processing ? `${Math.round(state.progress * 100)}%` : state.phase === "DONE" ? "100%" : ""}
          </span>
        </div>
      </footer>

      {showSettings && settings && (
        <SettingsDialog settings={settings} onClose={() => setShowSettings(false)} onSaved={setSettings} />
      )}
    </div>
  );
}
