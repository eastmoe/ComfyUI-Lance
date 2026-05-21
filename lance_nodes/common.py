"""Shared ComfyUI/Lance environment, localization, runtime cache, and progress helpers."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO_ROOT.parent.parent
LANCE_SRC = REPO_ROOT / "Lance"
LANCE_REPO_ID = "bytedance-research/Lance"
QUANTIZATION_CACHE_DIR_NAME = "Lance-quantized-cache"
DEFAULT_LOCALE = "zh-cn"

if str(COMFY_ROOT) not in sys.path and (COMFY_ROOT / "folder_paths.py").is_file():
    sys.path.insert(0, str(COMFY_ROOT))

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from comfy import model_management
except Exception:
    model_management = None

try:
    from comfy.utils import ProgressBar as ComfyProgressBar
except Exception:
    ComfyProgressBar = None

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

try:
    from comfy_api.latest import InputImpl, Types
except Exception:
    InputImpl = None
    Types = None


_RUNTIME_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.RLock()
_RUNTIME_CACHE: dict[tuple[Any, ...], "LanceRuntime"] = {}


def _load_localization(locale: str) -> dict[str, Any]:
    locale_name = (locale or DEFAULT_LOCALE).strip().lower()
    path = REPO_ROOT / "local" / locale_name / "nodes.json"
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[ComfyUI-Lance] Failed to load localization file {path}: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


_LOCALIZATION = _load_localization(os.environ.get("COMFYUI_LANCE_LOCALE", DEFAULT_LOCALE))


def _tr(path: str, default: Any) -> Any:
    value: Any = _LOCALIZATION
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _tr_text(path: str, default: str) -> str:
    value = _tr(path, default)
    return value if isinstance(value, str) else default


def _tr_names(path: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _tr(path, list(default))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return default


def _tr_mapping(path: str, default: dict[str, str]) -> dict[str, str]:
    value = _tr(path, default)
    if isinstance(value, dict) and all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        return value
    return default


CATEGORY = _tr_text("category", "Lance/Multimodal")


def _clear_cuda_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    if model_management is not None and hasattr(model_management, "soft_empty_cache"):
        model_management.soft_empty_cache()


def _drop_runtime(runtime: "LanceRuntime") -> None:
    runtime.model = None
    runtime.vae_model = None
    runtime.tokenizer = None


def _evict_runtime_cache(predicate, *, clear_cuda_cache: bool = True) -> int:
    runtimes: list[LanceRuntime] = []
    with _RUNTIME_LOCK:
        for key in list(_RUNTIME_CACHE):
            if predicate(key):
                runtime = _RUNTIME_CACHE.pop(key, None)
                if runtime is not None:
                    runtimes.append(runtime)
    for runtime in runtimes:
        _drop_runtime(runtime)
    if runtimes and clear_cuda_cache:
        _clear_cuda_runtime_cache()
    elif runtimes:
        gc.collect()
    return len(runtimes)


def _ui(key: str, display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    text = _tr(f"ui.{key}", {})
    if not isinstance(text, dict):
        text = {}
    extra["display_name"] = text.get("display_name", display_name)
    extra["tooltip"] = text.get("tooltip", tooltip)
    return extra


def _check_interrupted() -> None:
    if model_management is not None and hasattr(model_management, "throw_exception_if_processing_interrupted"):
        model_management.throw_exception_if_processing_interrupted()


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{rem:02d}s"
    return f"{minutes}m{rem:02d}s"


class _LanceProgress:
    def __init__(self, total: int, label: str, *, log_interval: float = 5.0) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self.label = label
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self.log_interval = float(log_interval)
        self.pbar = ComfyProgressBar(self.total) if ComfyProgressBar is not None else None
        self.tqdm = (
            _tqdm(
                total=self.total,
                desc=f"[ComfyUI-Lance] {self.label}",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
            if _tqdm is not None
            else None
        )
        if self.tqdm is None:
            print(f"[ComfyUI-Lance] {self.label} 开始。", flush=True)
        self._send()

    def _send(self) -> None:
        if self.pbar is not None:
            self.pbar.update_absolute(self.current, self.total)

    def _log(self, *, force: bool = False) -> None:
        if self.tqdm is not None:
            return
        now = time.monotonic()
        if not force and now - self.last_log_time < self.log_interval:
            return
        self.last_log_time = now
        percent = (self.current / self.total) * 100 if self.total else 100.0
        elapsed = now - self.start_time
        eta = ""
        if 0 < self.current < self.total:
            eta_seconds = elapsed * (self.total - self.current) / self.current
            eta = f", 预计剩余 {_format_seconds(eta_seconds)}"
        print(
            f"[ComfyUI-Lance] {self.label}: {self.current}/{self.total} "
            f"({percent:.1f}%), 已用 {_format_seconds(elapsed)}{eta}",
            flush=True,
        )

    def update(self, amount: int = 1, label: Optional[str] = None) -> None:
        self.update_absolute(self.current + int(amount), label=label)

    def update_absolute(self, value: int, *, total: Optional[int] = None, label: Optional[str] = None) -> None:
        if total is not None:
            self.total = max(1, int(total))
        if label:
            self.label = label
            if self.tqdm is not None:
                self.tqdm.set_description_str(f"[ComfyUI-Lance] {self.label}")
        previous = self.current
        self.current = max(0, min(int(value), self.total))
        self._send()
        if self.tqdm is not None:
            self.tqdm.total = self.total
            delta = self.current - previous
            if delta > 0:
                self.tqdm.update(delta)
            else:
                self.tqdm.n = self.current
                self.tqdm.refresh()
        self._log()

    def finish(self, label: Optional[str] = None) -> None:
        if label:
            self.label = label
            if self.tqdm is not None:
                self.tqdm.set_description_str(f"[ComfyUI-Lance] {self.label}")
        self.current = self.total
        self._send()
        if self.tqdm is not None:
            self.tqdm.n = self.total
            self.tqdm.refresh()
            self.tqdm.close()
        self._log(force=True)




__all__ = [name for name in globals() if not name.startswith("__")]
