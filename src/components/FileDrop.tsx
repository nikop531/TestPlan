import { useEffect, useRef, useState } from "react";
import { isTauri } from "../api";

interface Props {
  disabled: boolean;
  onPath: (path: string) => void; // Tauri: file stays where it is
  onFile: (file: File) => void; // Browser or file picker: uploaded to the sidecar
}

export function FileDrop({ disabled, onPath, onFile }: Props) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const disabledRef = useRef(disabled);
  disabledRef.current = disabled;

  // In the Tauri window, drops arrive as native file paths.
  useEffect(() => {
    if (!isTauri()) return;
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
      getCurrentWebview()
        .onDragDropEvent((event) => {
          const p = event.payload;
          if (p.type === "enter" || p.type === "over") setHover(true);
          else if (p.type === "leave") setHover(false);
          else if (p.type === "drop") {
            setHover(false);
            if (!disabledRef.current && p.paths.length > 0) onPath(p.paths[0]);
          }
        })
        .then((fn) => {
          if (cancelled) fn();
          else unlisten = fn;
        }),
    );
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [onPath]);

  return (
    <div
      className={`drop ${hover ? "hover" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        const file = e.dataTransfer.files[0];
        if (file && !disabled) onFile(file);
      }}
    >
      <span>把錄音檔拖曳到這裡，或</span>
      <button className="btn small" disabled={disabled} onClick={() => inputRef.current?.click()}>
        選擇檔案
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.mp3,.m4a,.flac,.ogg,.aac,.mp4,audio/*"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
      />
    </div>
  );
}
