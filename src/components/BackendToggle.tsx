import type { Backend } from "../api";

interface Props {
  value: Backend;
  disabled: boolean;
  onChange: (value: Backend) => void;
  onOpenSettings: () => void;
}

const OPTIONS: { value: Backend; label: string; hint: string }[] = [
  { value: "local", label: "本機", hint: "在這台 Mac 上處理，音訊不會離開電腦" },
  { value: "hf", label: "Hugging Face", hint: "上傳到雲端處理，較省電" },
];

export function BackendToggle({ value, disabled, onChange, onOpenSettings }: Props) {
  return (
    <div className="backend">
      <span className="label" id="backend-label">
        精修方式
      </span>
      <div className="segmented" role="radiogroup" aria-labelledby="backend-label">
        {OPTIONS.map((o) => (
          <button
            key={o.value}
            role="radio"
            aria-checked={value === o.value}
            title={o.hint}
            className={value === o.value ? "active" : ""}
            disabled={disabled}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
      <button className="btn small ghost" onClick={onOpenSettings} disabled={disabled}>
        設定
      </button>
    </div>
  );
}
