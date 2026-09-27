"""Track B backends. ``get_backend`` picks one by name."""

from __future__ import annotations

from config import MOCK_MODE

from .base import Backend, BackendResult, ProgressFn


def get_backend(name: str) -> Backend:
    if MOCK_MODE:
        from .mock_backend import MockBackend

        return MockBackend(label=name)
    if name == "local":
        from .local_backend import LocalBackend

        return LocalBackend()
    if name == "hf":
        from .hf_backend import HFBackend

        return HFBackend()
    raise ValueError(f"unknown backend: {name}")


__all__ = ["Backend", "BackendResult", "ProgressFn", "get_backend"]
