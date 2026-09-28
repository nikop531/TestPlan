"""Loads the Nemotron diarization checkpoint and applies a streaming preset.

Track A and Track B use the same checkpoint with different streaming
parameters. Those parameters live on the model object, so each track keeps
its own instance instead of flipping settings on a shared one.
"""

from __future__ import annotations

import logging
import threading

from config import MODEL_ID, StreamingPreset, pick_device

log = logging.getLogger(__name__)

_cache: dict[str, object] = {}
_lock = threading.Lock()


def apply_preset(model, preset: StreamingPreset) -> None:
    mods = model.sortformer_modules
    mods.chunk_len = preset.chunk_len
    mods.chunk_left_context = preset.chunk_left_context
    mods.chunk_right_context = preset.chunk_right_context
    mods.fifo_len = preset.fifo_len
    mods.spkcache_update_period = preset.spkcache_update_period
    mods.spkcache_len = preset.spkcache_len
    # Streaming mode processes the file chunk by chunk with a speaker cache,
    # which keeps memory flat on hour long recordings.
    model.streaming_mode = True
    if hasattr(mods, "_check_streaming_parameters"):
        mods._check_streaming_parameters()


def load_model(preset: StreamingPreset, device: str | None = None):
    """Return a cached model configured for ``preset``."""
    device = device or pick_device()
    key = f"{preset.name}@{device}"
    with _lock:
        if key in _cache:
            return _cache[key]
        from nemo.collections.asr.models import SortformerEncLabelModel

        log.info("loading %s for %s on %s", MODEL_ID, preset.name, device)
        if MODEL_ID.endswith(".nemo"):
            model = SortformerEncLabelModel.restore_from(MODEL_ID, map_location=device)
        else:
            model = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location=device)
        apply_preset(model, preset)
        model.eval()
        _cache[key] = model
        return model


def is_loaded(preset: StreamingPreset) -> bool:
    return any(k.startswith(f"{preset.name}@") for k in _cache)
