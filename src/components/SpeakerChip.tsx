import { speakerIndex, speakerName } from "../api";

export function SpeakerChip({ speaker }: { speaker: string }) {
  const idx = speakerIndex(speaker) % 8;
  return <span className={`chip spk-${idx}`}>{speakerName(speaker)}</span>;
}
