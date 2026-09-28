// Thin client for the Python sidecar (python-sidecar/main.py).

export const SIDECAR_URL: string =
  (import.meta.env.VITE_SIDECAR_URL as string | undefined) ?? "http://127.0.0.1:8765";

export type Backend = "local" | "hf";

export interface LiveSegment {
  id: number;
  start: number;
  end: number;
  speaker: string;
}

export interface TranscriptRow {
  start: number;
  end: number;
  speaker: string;
  text: string;
}

export interface Health {
  ok: boolean;
  mock: boolean;
  device: string;
  realtime_model_ready: boolean;
  realtime_latency: number;
  offline_latency: number;
}

export interface Settings {
  backend: Backend;
  hf_endpoint: string;
  hf_upload_consent: boolean;
  run_asr: boolean;
  has_hf_token: boolean;
}

export interface JobStatus {
  id: string;
  state: "queued" | "running" | "done" | "error";
  progress: number;
  stage: string;
  result_path: string | null;
  srt_path: string | null;
  error: string | null;
  duration: number | null;
}

export type StreamMessage =
  | { type: "snapshot"; segments: LiveSegment[] }
  | { type: "segments"; segments: LiveSegment[] }
  | {
      type: "status";
      state: string;
      recorded_seconds: number;
      lag_seconds: number;
      error: string | null;
    }
  | { type: "stopped"; session_id: string; wav_path: string; duration: number }
  | { type: "error"; message: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${SIDECAR_URL}${path}`, init);
  } catch {
    throw new Error("無法連線到辨識引擎，請稍候再試。");
  }
  if (!res.ok) {
    let message = `發生錯誤（${res.status}）`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* keep default message */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  system: () => request<{ on_battery: boolean | null; recommended_backend: Backend }>("/system"),
  settings: () => request<Settings>("/settings"),
  saveSettings: (changes: Partial<Omit<Settings, "has_hf_token">>) =>
    request<Settings>("/settings", json(changes)),
  saveToken: (token: string | null) => request<Settings>("/settings/hf_token", json({ token })),
  start: () => request<{ session_id: string }>("/start", json({ source: "mic" })),
  stop: (sessionId: string) =>
    request<{ session_id: string; wav_path: string; duration: number }>(
      "/stop",
      json({ session_id: sessionId }),
    ),
  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ file_path: string }>("/upload", { method: "POST", body: form });
  },
  diarizeOffline: (filePath: string, backend: Backend) =>
    request<{ job_id: string }>("/diarize/offline", json({ file_path: filePath, backend })),
  status: (jobId: string) => request<JobStatus>(`/status?job_id=${jobId}`),
  result: (jobId: string) => request<TranscriptRow[]>(`/result?job_id=${jobId}`),
  reveal: (jobId: string) =>
    request<{ path: string }>(`/reveal?job_id=${jobId}`, { method: "POST" }),
  exportUrl: (jobId: string, format: "json" | "srt") =>
    `${SIDECAR_URL}/export?job_id=${jobId}&format=${format}`,
  streamUrl: (sessionId: string) =>
    `${SIDECAR_URL.replace(/^http/, "ws")}/stream?session_id=${sessionId}`,
};

export const isTauri = (): boolean => "__TAURI_INTERNALS__" in window;

export function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function speakerName(label: string): string {
  const match = /(\d+)$/.exec(label);
  return match ? `說話者 ${Number(match[1]) + 1}` : label;
}

export function speakerIndex(label: string): number {
  const match = /(\d+)$/.exec(label);
  return match ? Number(match[1]) : 0;
}
