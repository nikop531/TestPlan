import { useState } from "react";
import { api, type Settings } from "../api";

interface Props {
  settings: Settings;
  onClose: () => void;
  onSaved: (s: Settings) => void;
}

export function SettingsDialog({ settings, onClose, onSaved }: Props) {
  const [token, setToken] = useState("");
  const [endpoint, setEndpoint] = useState(settings.hf_endpoint);
  const [consent, setConsent] = useState(settings.hf_upload_consent);
  const [runAsr, setRunAsr] = useState(settings.run_asr);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      let next = await api.saveSettings({
        hf_endpoint: endpoint.trim(),
        hf_upload_consent: consent,
        run_asr: runAsr,
      });
      if (token.trim()) next = await api.saveToken(token.trim());
      onSaved(next);
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function clearToken() {
    try {
      onSaved(await api.saveToken(null));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="dialog">
        <h2 id="settings-title">設定</h2>

        <label className="check">
          <input type="checkbox" checked={runAsr} onChange={(e) => setRunAsr(e.target.checked)} />
          產生文字逐字稿（語音轉文字）
        </label>

        <h3>Hugging Face 雲端處理</h3>
        <p className="warn">
          使用 Hugging Face 時，錄音檔會上傳到雲端伺服器處理。會議內容敏感時，請改用「本機」。
        </p>
        <label className="field">
          <span>存取金鑰（HF_TOKEN）</span>
          <input
            type="password"
            placeholder={settings.has_hf_token ? "已儲存在鑰匙圈，輸入新值可更換" : "hf_ 開頭的金鑰"}
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
          />
        </label>
        {settings.has_hf_token && (
          <button className="btn small ghost" onClick={clearToken}>
            刪除已儲存的金鑰
          </button>
        )}
        <label className="field">
          <span>端點網址</span>
          <input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} spellCheck={false} />
        </label>
        <label className="check">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          我了解並同意將錄音上傳到 Hugging Face
        </label>

        {error && <p className="error">{error}</p>}
        <div className="dialog-actions">
          <button className="btn ghost" onClick={onClose}>
            取消
          </button>
          <button className="btn primary" onClick={save} disabled={saving}>
            {saving ? "儲存中" : "儲存"}
          </button>
        </div>
      </div>
    </div>
  );
}
