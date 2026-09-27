"""User settings (JSON file) and the Hugging Face token (macOS Keychain).

The token is never written to the settings file or to logs.
"""

from __future__ import annotations

import json
import logging
import os
import threading

from config import APP_NAME, HF_DEFAULT_ENDPOINT, SETTINGS_FILE

log = logging.getLogger(__name__)

_KEYRING_USER = "hf_token"
_lock = threading.Lock()

DEFAULTS = {
    "backend": "local",
    "hf_endpoint": HF_DEFAULT_ENDPOINT,
    "hf_upload_consent": False,
    "run_asr": True,
}


def load_settings() -> dict:
    with _lock:
        data = dict(DEFAULTS)
        try:
            data.update(json.loads(SETTINGS_FILE.read_text("utf-8")))
        except FileNotFoundError:
            pass
        except Exception:
            log.warning("settings file unreadable, using defaults")
        return data


def save_settings(changes: dict) -> dict:
    allowed = {k: v for k, v in changes.items() if k in DEFAULTS and v is not None}
    current = load_settings()
    current.update(allowed)
    with _lock:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), "utf-8")
    return current


def _keyring():
    try:
        import keyring

        return keyring
    except Exception:
        return None


def get_hf_token() -> str | None:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(APP_NAME, _KEYRING_USER)
    except Exception:
        log.warning("keychain read failed")
        return None


def set_hf_token(token: str | None) -> None:
    kr = _keyring()
    if kr is None:
        raise RuntimeError("找不到系統鑰匙圈，無法安全儲存金鑰。")
    if token:
        kr.set_password(APP_NAME, _KEYRING_USER, token.strip())
    else:
        try:
            kr.delete_password(APP_NAME, _KEYRING_USER)
        except Exception:
            pass
