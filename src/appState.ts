// GUI state machine: IDLE, RECORDING_REALTIME, PROCESSING_OFFLINE, DONE, ERROR.

import type { LiveSegment, TranscriptRow } from "./api";

export type Phase = "IDLE" | "RECORDING_REALTIME" | "PROCESSING_OFFLINE" | "DONE" | "ERROR";

export interface AppState {
  phase: Phase;
  sessionId: string | null;
  jobId: string | null;
  liveSegments: LiveSegment[];
  recordedSeconds: number;
  lagSeconds: number;
  liveNote: string | null;
  progress: number;
  stage: string;
  sourceName: string | null;
  rows: TranscriptRow[];
  error: string | null;
}

export const initialState: AppState = {
  phase: "IDLE",
  sessionId: null,
  jobId: null,
  liveSegments: [],
  recordedSeconds: 0,
  lagSeconds: 0,
  liveNote: null,
  progress: 0,
  stage: "",
  sourceName: null,
  rows: [],
  error: null,
};

export type Action =
  | { type: "RECORDING_STARTED"; sessionId: string }
  | { type: "LIVE_SEGMENTS"; segments: LiveSegment[]; replace?: boolean }
  | { type: "LIVE_STATUS"; recordedSeconds: number; lagSeconds: number; note: string | null }
  | { type: "OFFLINE_STARTED"; jobId: string; sourceName: string; keepLive: boolean }
  | { type: "OFFLINE_PROGRESS"; progress: number; stage: string }
  | { type: "OFFLINE_DONE"; rows: TranscriptRow[] }
  | { type: "FAILED"; message: string }
  | { type: "RESET" };

function mergeSegments(current: LiveSegment[], updates: LiveSegment[]): LiveSegment[] {
  const byId = new Map(current.map((s) => [s.id, s]));
  for (const seg of updates) byId.set(seg.id, seg);
  return Array.from(byId.values()).sort((a, b) => a.start - b.start);
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "RECORDING_STARTED":
      return {
        ...initialState,
        phase: "RECORDING_REALTIME",
        sessionId: action.sessionId,
        sourceName: "麥克風錄音",
      };
    case "LIVE_SEGMENTS":
      return {
        ...state,
        liveSegments: action.replace
          ? mergeSegments([], action.segments)
          : mergeSegments(state.liveSegments, action.segments),
      };
    case "LIVE_STATUS":
      return {
        ...state,
        recordedSeconds: action.recordedSeconds,
        lagSeconds: action.lagSeconds,
        liveNote: action.note,
      };
    case "OFFLINE_STARTED":
      return {
        ...state,
        phase: "PROCESSING_OFFLINE",
        jobId: action.jobId,
        sourceName: action.sourceName,
        liveSegments: action.keepLive ? state.liveSegments : [],
        progress: 0,
        stage: "排隊中",
        rows: [],
        error: null,
      };
    case "OFFLINE_PROGRESS":
      return { ...state, progress: action.progress, stage: action.stage };
    case "OFFLINE_DONE":
      return { ...state, phase: "DONE", rows: action.rows, progress: 1, stage: "完成" };
    case "FAILED":
      return { ...state, phase: "ERROR", error: action.message };
    case "RESET":
      return initialState;
  }
}
