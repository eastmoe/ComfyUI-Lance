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


REPO_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = REPO_ROOT.parent.parent
LANCE_SRC = REPO_ROOT / "Lance"
LANCE_REPO_ID = "bytedance-research/Lance"
QUANTIZATION_CACHE_DIR_NAME = "Lance-quantized-cache"
CATEGORY = "Lance/多模态"

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


def _ui(display_name: str, tooltip: str, **extra: Any) -> dict[str, Any]:
    extra["display_name"] = display_name
    extra["tooltip"] = tooltip
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


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return COMFY_ROOT / "models"


def _comfy_output_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        return Path(folder_paths.get_output_directory())
    return COMFY_ROOT / "output"


def _comfy_temp_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
        return Path(folder_paths.get_temp_directory())
    return Path(tempfile.gettempdir()) / "ComfyUI"


def _resolve_model_root(model_root: str) -> Path:
    text = (model_root or "auto").strip().strip('"')
    if not text or text.lower() == "auto":
        return _comfy_models_dir() / "Lance"
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _device_choices() -> list[str]:
    choices = ["auto"]
    if torch.cuda.is_available():
        choices.append("cuda")
        choices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    return choices or ["auto"]


def _resolve_device(device: str) -> tuple[torch.device, int]:
    selected = (device or "auto").strip().lower()
    if selected == "auto":
        if model_management is not None and hasattr(model_management, "get_torch_device"):
            comfy_device = model_management.get_torch_device()
            if getattr(comfy_device, "type", None) == "cuda":
                return comfy_device, int(comfy_device.index or 0)
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device()), int(torch.cuda.current_device())
        raise RuntimeError("Lance 推理需要 CUDA 设备，但当前没有可用的 CUDA。")
    if selected == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("选择了 cuda，但当前没有可用的 CUDA。")
        return torch.device("cuda", torch.cuda.current_device()), int(torch.cuda.current_device())
    if selected.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("选择了 CUDA 设备，但当前没有可用的 CUDA。")
        index = int(selected.split(":", 1)[1])
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA 设备不存在: {selected}")
        return torch.device("cuda", index), index
    raise ValueError("Lance 当前推理路径只支持 CUDA 设备。")


def _dtype_from_name(name: str) -> torch.dtype:
    selected = (name or "bf16").strip().lower()
    if selected in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if selected in {"fp16", "float16"}:
        return torch.float16
    if selected in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"未知 compute dtype: {name}")


def _ensure_lance_import_path() -> None:
    if not LANCE_SRC.is_dir():
        raise FileNotFoundError(f"找不到 Lance 推理源码目录: {LANCE_SRC}")
    if str(LANCE_SRC) not in sys.path:
        sys.path.insert(0, str(LANCE_SRC))


@dataclass(frozen=True)
class LancePaths:
    root: Path
    image_model: Path
    video_model: Path
    vit: Path
    vae: Path


def _lance_paths(model_root: Path) -> LancePaths:
    return LancePaths(
        root=model_root,
        image_model=model_root / "Lance_3B",
        video_model=model_root / "Lance_3B_Video",
        vit=model_root / "Qwen2.5-VL-ViT",
        vae=model_root / "Wan2.2_VAE.pth",
    )


def _looks_like_lance_model(path: Path) -> bool:
    return (
        (path / "llm_config.json").is_file()
        and ((path / "model.safetensors").is_file() or (path / "ema.safetensors").is_file())
        and ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file())
    )


def _looks_like_vit(path: Path) -> bool:
    return (path / "vit.safetensors").is_file() and (
        (path / "config.json").is_file() or (path / "preprocessor_config.json").is_file()
    )


def _missing_model_items(paths: LancePaths, family: Optional[str] = None) -> list[str]:
    missing = []
    if family in {None, "image"} and not _looks_like_lance_model(paths.image_model):
        missing.append(f"图像模型 {paths.image_model}")
    if family in {None, "video"} and not _looks_like_lance_model(paths.video_model):
        missing.append(f"视频模型 {paths.video_model}")
    if not _looks_like_vit(paths.vit):
        missing.append(f"Qwen2.5-VL ViT {paths.vit}")
    if not paths.vae.is_file():
        missing.append(f"Wan VAE {paths.vae}")
    return missing


def _download_lance_snapshot(model_root: Path, source: str, revision: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 huggingface_hub，无法自动下载 Lance 模型。") from exc

    model_root.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": LANCE_REPO_ID,
        "local_dir": str(model_root),
        "allow_patterns": [
            "Lance_3B/**",
            "Lance_3B_Video/**",
            "Qwen2.5-VL-ViT/**",
            "Wan2.2_VAE.pth",
        ],
    }
    if revision.strip():
        kwargs["revision"] = revision.strip()

    if source == "hf-mirror.com":
        kwargs["endpoint"] = "https://hf-mirror.com"

    try:
        snapshot_download(**kwargs)
    except TypeError:
        endpoint = kwargs.pop("endpoint", None)
        if endpoint is None:
            raise
        previous = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            snapshot_download(**kwargs)
        finally:
            if previous is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous


def _ensure_models(paths: LancePaths, download_missing: bool, download_source: str, revision: str, family: Optional[str] = None) -> None:
    missing = _missing_model_items(paths, family)
    if missing and download_missing:
        _download_lance_snapshot(paths.root, download_source, revision)
        missing = _missing_model_items(paths, family)
    if missing:
        layout = (
            "请将 bytedance-research/Lance 权重放到 ComfyUI/models/Lance，目录应包含:\n"
            "  Lance_3B/\n"
            "  Lance_3B_Video/\n"
            "  Qwen2.5-VL-ViT/\n"
            "  Wan2.2_VAE.pth"
        )
        raise FileNotFoundError("Lance 模型文件不完整:\n- " + "\n- ".join(missing) + "\n\n" + layout)


def _install_lance_path_config(paths: LancePaths) -> None:
    _ensure_lance_import_path()
    import config.config_factory as config_factory

    config_factory._MODEL_PATH_CONFIG_CACHE = {
        "base_dir": str(paths.root),
        "lance": {
            "image": str(paths.image_model),
            "video": str(paths.video_model),
        },
        "vit": {
            "qwen2_5_vl": str(paths.vit),
        },
        "vae": {
            "wan": str(paths.vae),
        },
    }


def _patch_lance_device(device: torch.device) -> None:
    _ensure_lance_import_path()
    index = int(device.index or 0)
    os.environ["LOCAL_RANK"] = str(index)
    torch.cuda.set_device(index)
    import common.utils.distributed as distributed

    distributed.get_device = lambda: device
    try:
        import modeling.vae.wan.model as wan_model

        wan_model.get_device = lambda: device
    except Exception:
        pass


def _configure_attention_backend(attention_backend: str) -> None:
    backend = (attention_backend or "auto").strip().lower()
    _ensure_lance_import_path()
    if backend in {"auto", "flash_attention_2"}:
        return
    if backend in {"sdpa", "eager"}:
        try:
            import modeling.lance.qwen2_navit as navit

            navit.flash_attn_varlen_func = None
        except Exception:
            pass
        return
    if backend == "sage_attention":
        try:
            from sageattention import sageattn
        except Exception as exc:
            raise RuntimeError(
                "已选择 SageAttention，但没有安装 sageattention。请安装 sageattention，或把 attention_backend 改为 auto/sdpa。"
            ) from exc
        import modeling.lance.qwen2_navit as navit

        def sage_varlen_or_sdpa(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            causal: bool = False,
        ) -> torch.Tensor:
            outputs = []
            cu_q = cu_seqlens_q.to(torch.long)
            cu_k = cu_seqlens_k.to(torch.long)
            num_heads = q.shape[1]
            kv_heads = k.shape[1]
            kv_groups = num_heads // kv_heads
            for i in range(cu_q.numel() - 1):
                _check_interrupted()
                qi = q[cu_q[i] : cu_q[i + 1]]
                ki = k[cu_k[i] : cu_k[i + 1]]
                vi = v[cu_k[i] : cu_k[i + 1]]
                if kv_groups != 1:
                    ki = ki.repeat_interleave(kv_groups, dim=1)
                    vi = vi.repeat_interleave(kv_groups, dim=1)
                try:
                    out = sageattn(
                        qi.unsqueeze(0),
                        ki.unsqueeze(0),
                        vi.unsqueeze(0),
                        is_causal=causal,
                        tensor_layout="NHD",
                    ).squeeze(0)
                except Exception:
                    attn_mask = None
                    if causal:
                        query_len = qi.shape[0]
                        key_len = ki.shape[0]
                        past_len = key_len - query_len
                        query_positions = torch.arange(query_len, device=q.device) + past_len
                        key_positions = torch.arange(key_len, device=q.device)
                        attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        qi.transpose(0, 1).unsqueeze(0),
                        ki.transpose(0, 1).unsqueeze(0),
                        vi.transpose(0, 1).unsqueeze(0),
                        attn_mask=attn_mask,
                        dropout_p=0.0,
                    ).squeeze(0).transpose(0, 1)
                outputs.append(out)
            return torch.cat(outputs, dim=0)

        navit.flash_attn_varlen_or_sdpa = sage_varlen_or_sdpa
        return
    raise ValueError(f"未知 attention backend: {attention_backend}")


def _vit_attention_impl(attention_backend: str) -> str:
    backend = (attention_backend or "auto").strip().lower()
    if backend == "flash_attention_2":
        return "flash_attention_2"
    if backend in {"sdpa", "sage_attention"}:
        return "sdpa"
    return "flash_attention_2"


def _llm_attention_impl(attention_backend: str) -> str:
    # Navit uses its own packed attention path; keep Transformers init from
    # rejecting this custom architecture during backend validation.
    return "eager"


def _set_attention_impl(config: Any, implementation: str) -> None:
    config._attn_implementation = implementation
    # Newer Transformers versions keep the requested backend in this internal
    # field and validate it during PreTrainedModel.__init__.
    config._attn_implementation_internal = implementation


def _normalize_quantization_mode(quantization: str) -> Optional[str]:
    mode = (quantization or "none").strip().lower()
    if mode in {"none", "off", "false", ""}:
        return None
    if mode in {"fp4_e2m1", "fp4_e2m1fn", "fp4_e2m1fn_x2", "float4_e2m1fn_x2"}:
        return "fp4"
    if mode in {"int8", "int4", "fp4", "fp8_e4m3fn", "fp8_e5m2", "fp8"}:
        return mode
    raise ValueError(f"不支持的量化格式: {mode}")


def _lance_quantization_cache_dir() -> Path:
    return _comfy_models_dir() / QUANTIZATION_CACHE_DIR_NAME


def _checkpoint_file(path: Path) -> Path:
    model_path = path / "model.safetensors"
    if model_path.is_file():
        return model_path
    ema_path = path / "ema.safetensors"
    if ema_path.is_file():
        return ema_path
    raise FileNotFoundError(f"找不到 Lance checkpoint: {path}")


_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


class _PlainSafetensorsReader:
    """Read safetensors tensors without mmap, which can trip Windows pagefile limits."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._header: dict[str, Any] = {}
        self._data_start = 0

    def __enter__(self) -> "_PlainSafetensorsReader":
        self._file = open(self.path, "rb")
        header_len_bytes = self._file.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"无效 safetensors 文件: {self.path}")
        header_len = int.from_bytes(header_len_bytes, "little")
        header_raw = self._file.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"safetensors header 不完整: {self.path}")
        self._header = json.loads(header_raw.decode("utf-8"))
        self._data_start = 8 + header_len
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def keys(self) -> list[str]:
        return [key for key in self._header if key != "__metadata__"]

    def get_tensor(self, key: str, *, clone: bool = False) -> torch.Tensor:
        if self._file is None:
            raise RuntimeError("safetensors reader is not open")
        item = self._header[key]
        dtype_name = item["dtype"]
        dtype = _SAFETENSORS_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(f"暂不支持 safetensors dtype: {dtype_name}")
        shape = tuple(int(dim) for dim in item["shape"])
        start, end = (int(offset) for offset in item["data_offsets"])
        self._file.seek(self._data_start + start)
        raw = self._file.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"tensor {key!r} 数据不完整: {self.path}")
        tensor = torch.frombuffer(raw, dtype=dtype).reshape(shape)
        return tensor.clone() if clone else tensor


def _file_signature(path: Path, label: Optional[str] = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": label or path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _lance_checkpoint_signature(model_path: Path, vit_path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for filename in (
        "llm_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        candidate = model_path / filename
        if candidate.exists():
            files.append(_file_signature(candidate))

    checkpoint = _checkpoint_file(model_path)
    files.append(_file_signature(checkpoint))

    vit_files: list[dict[str, Any]] = []
    for filename in ("config.json", "preprocessor_config.json", "vit.safetensors"):
        candidate = vit_path / filename
        if candidate.exists():
            vit_files.append(_file_signature(candidate))

    return {
        "model_path": str(model_path.resolve()),
        "checkpoint": checkpoint.name,
        "vit_path": str(vit_path.resolve()),
        "files": sorted(files, key=lambda item: item["name"]),
        "vit_files": sorted(vit_files, key=lambda item: item["name"]),
    }


def _signature_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _quantization_cache_path(cache_dir: Path, model_path: Path, vit_path: Path, mode: str) -> tuple[Path, dict[str, Any]]:
    metadata = {
        "cache_version": 3,
        "format": "lance-weight-only-state-dict-safe-fp8",
        "mode": mode,
        "signature": _lance_checkpoint_signature(model_path, vit_path),
    }
    filename = f"{model_path.name}-{mode}-{_signature_digest(metadata)}.pt"
    return cache_dir / filename, metadata


def _torch_load_weights(path: Path, map_location: str | torch.device):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _float8_dtype_name(dtype: torch.dtype) -> Optional[str]:
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return "float8_e5m2"
    return None


def _float8_dtype_from_name(name: str) -> torch.dtype:
    if name == "float8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if name == "float8_e5m2" and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    raise RuntimeError(f"当前 PyTorch 不支持量化缓存中的 dtype: {name}")


def _pack_quantized_cache_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    packed: dict[str, torch.Tensor] = {}
    packed_float8_dtypes: dict[str, str] = {}
    for key, tensor in state_dict.items():
        dtype_name = _float8_dtype_name(tensor.dtype)
        if dtype_name is None:
            packed[key] = tensor
            continue
        packed[key] = tensor.detach().cpu().contiguous().view(torch.uint8).clone()
        packed_float8_dtypes[key] = dtype_name
    return packed, packed_float8_dtypes


def _unpack_quantized_cache_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = payload["state_dict"]
    packed_float8_dtypes = payload.get("packed_float8_dtypes", {})
    for key, dtype_name in packed_float8_dtypes.items():
        if key not in state_dict:
            raise RuntimeError(f"量化缓存缺少 fp8 tensor: {key}")
        tensor = state_dict[key]
        if tensor.dtype != torch.uint8:
            raise RuntimeError(f"量化缓存中的 fp8 tensor 未按 uint8 保存: {key} ({tensor.dtype})")
        state_dict[key] = tensor.contiguous().view(_float8_dtype_from_name(dtype_name))
    return state_dict


def _expected_quantized_linear_shapes(
    linear: torch.nn.Linear,
    mode: str,
    *,
    has_bias: bool,
) -> dict[str, tuple[int, ...]]:
    out_features = int(linear.out_features)
    in_features = int(linear.in_features)
    mode = mode.lower()
    if mode in {"int8", "fp8_e4m3fn", "fp8_e5m2", "fp8"}:
        qweight_shape = (out_features, in_features)
    elif mode in {"int4", "fp4"}:
        qweight_shape = (out_features, (in_features + 1) // 2)
    else:
        raise ValueError(f"不支持的量化格式: {mode}")

    shapes = {"qweight": qweight_shape}
    if mode in {"int8", "int4", "fp4"}:
        shapes["scale"] = (out_features,)
    if has_bias:
        shapes["bias"] = (out_features,)
    return shapes


def _validate_quantized_cache_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    quantized_module_names: list[str],
    *,
    mode: str,
) -> None:
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for module_name in quantized_module_names:
        module = model.get_submodule(module_name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"量化缓存期望 Linear 模块 {module_name!r}，实际为 {type(module).__name__}。")

        quantized_shapes = _expected_quantized_linear_shapes(
            module,
            mode,
            has_bias=f"{module_name}.bias" in state_dict,
        )
        for attr_name, shape in quantized_shapes.items():
            expected_shapes[f"{module_name}.{attr_name}"] = shape

    mismatches = []
    for key, expected_shape in expected_shapes.items():
        if key not in state_dict:
            mismatches.append(f"{key}: 缓存缺失，当前需要 {expected_shape}")
            continue
        actual_shape = tuple(state_dict[key].shape)
        if actual_shape != expected_shape:
            mismatches.append(f"{key}: 缓存 {actual_shape} != 当前 {expected_shape}")
        if len(mismatches) >= 5:
            break
    if mismatches:
        raise RuntimeError("量化缓存 tensor shape 与当前模型结构不匹配: " + "；".join(mismatches))


def _split_tensor_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return "", name
    return name.rsplit(".", 1)


def _replace_module(model: torch.nn.Module, module_name: str, module: torch.nn.Module) -> None:
    parent_name, child_name = _split_tensor_name(module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _set_module_tensor(model: torch.nn.Module, tensor_name: str, tensor: torch.Tensor) -> None:
    module_name, attr_name = _split_tensor_name(tensor_name)
    module = model.get_submodule(module_name) if module_name else model
    if attr_name in module._parameters:
        old_param = module._parameters[attr_name]
        requires_grad = bool(old_param.requires_grad) if old_param is not None else False
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if attr_name in module._buffers:
        module._buffers[attr_name] = tensor
        return
    raise KeyError(f"无法放置 tensor {tensor_name!r}: 目标属性不存在。")


def _quantization_chunk_rows(weight: torch.Tensor, mode: str) -> int:
    if weight.ndim != 2 or weight.shape[1] == 0:
        return 1
    max_temp_mb = int(os.environ.get("LANCE_QUANTIZE_MAX_TEMP_MB", "256"))
    multiplier = 4 if mode in {"int4", "fp4"} else 2
    bytes_per_row = max(1, weight.shape[1] * multiplier * 4)
    return max(1, min(weight.shape[0], (max_temp_mb * 1024 * 1024) // bytes_per_row))


def _fp4_e2m1fn_codes(values: torch.Tensor) -> torch.Tensor:
    abs_values = values.abs()
    magnitude = torch.zeros_like(abs_values, dtype=torch.uint8)
    magnitude = torch.where(abs_values >= 0.25, torch.ones_like(magnitude), magnitude)
    magnitude = torch.where(abs_values >= 0.75, torch.full_like(magnitude, 2), magnitude)
    magnitude = torch.where(abs_values >= 1.25, torch.full_like(magnitude, 3), magnitude)
    magnitude = torch.where(abs_values >= 1.75, torch.full_like(magnitude, 4), magnitude)
    magnitude = torch.where(abs_values >= 2.50, torch.full_like(magnitude, 5), magnitude)
    magnitude = torch.where(abs_values >= 3.50, torch.full_like(magnitude, 6), magnitude)
    magnitude = torch.where(abs_values >= 5.00, torch.full_like(magnitude, 7), magnitude)
    sign = (values < 0).to(torch.uint8) << 3
    return magnitude | sign


def _dequantize_fp4_e2m1fn(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    packed = packed.to(device=device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(out_features, -1)[:, :in_features]
    magnitude = codes & 0x07
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device, dtype=dtype)
    values = lut[magnitude.long()]
    signs = torch.where((codes & 0x08) != 0, -1.0, 1.0).to(dtype=dtype, device=device)
    return values * signs * scale.to(device=device, dtype=dtype)[:, None]


def _quantize_weight_tensor(
    weight: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, Optional[torch.Tensor], str, Optional[torch.dtype]]:
    if weight.ndim != 2:
        raise ValueError(f"只能量化 2D Linear 权重，实际 shape={tuple(weight.shape)}")

    mode = mode.lower()
    weight = weight.detach().cpu()
    out_features, in_features = weight.shape
    chunk_rows = _quantization_chunk_rows(weight, mode)

    if mode == "int8":
        qweight = torch.empty((out_features, in_features), dtype=torch.int8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            qweight[start:end] = torch.round(chunk / chunk_scale[:, None]).clamp(-127, 127).to(torch.int8)
            scale[start:end] = chunk_scale
        return qweight, scale, "int8", None

    if mode == "int4":
        qweight = torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 7.0
            q = torch.round(chunk / chunk_scale[:, None]).clamp(-8, 7).to(torch.int8)
            q = (q + 8).to(torch.uint8)
            if in_features % 2:
                q = torch.nn.functional.pad(q, (0, 1))
            qweight[start:end] = q[:, 0::2] | (q[:, 1::2] << 4)
            scale[start:end] = chunk_scale
        return qweight, scale, "int4", None

    if mode == "fp4":
        qweight = torch.empty((out_features, (in_features + 1) // 2), dtype=torch.uint8)
        scale = torch.empty((out_features,), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1).clamp(min=1e-8) / 6.0
            q = _fp4_e2m1fn_codes((chunk / chunk_scale[:, None]).clamp(-6.0, 6.0))
            if in_features % 2:
                q = torch.nn.functional.pad(q, (0, 1))
            qweight[start:end] = q[:, 0::2] | (q[:, 1::2] << 4)
            scale[start:end] = chunk_scale
        return qweight, scale, "fp4", None

    if mode in {"fp8_e4m3fn", "fp8"}:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("当前 PyTorch 不支持 torch.float8_e4m3fn。")
        return weight.to(torch.float8_e4m3fn), None, "fp8", torch.float8_e4m3fn

    if mode == "fp8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("当前 PyTorch 不支持 torch.float8_e5m2。")
        return weight.to(torch.float8_e5m2), None, "fp8", torch.float8_e5m2

    raise ValueError(f"不支持的量化格式: {mode}")


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
        mode: str,
        fp8_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = mode
        self.fp8_dtype = fp8_dtype
        self.register_buffer("qweight", qweight.contiguous())
        if scale is not None:
            self.register_buffer("scale", scale.contiguous())
        else:
            self.scale = None
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, module: torch.nn.Linear, mode: str) -> "QuantizedLinear":
        weight = module.weight.detach().float().cpu()
        bias = module.bias.detach().cpu() if module.bias is not None else None
        qweight, scale, internal_mode, fp8_dtype = _quantize_weight_tensor(weight, mode)
        return cls(qweight, scale, bias, module.in_features, module.out_features, internal_mode, fp8_dtype)

    @classmethod
    def from_quantized_tensors(
        cls,
        *,
        qweight: torch.Tensor,
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
        mode: str,
        fp8_dtype: Optional[torch.dtype] = None,
    ) -> "QuantizedLinear":
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.in_features = int(in_features)
        module.out_features = int(out_features)
        module.mode = mode
        module.fp8_dtype = fp8_dtype
        module.register_buffer("qweight", qweight)
        if scale is not None:
            module.register_buffer("scale", scale)
        else:
            module.scale = None
        if bias is not None:
            module.register_buffer("bias", bias)
        else:
            module.bias = None
        return module

    def _apply(self, fn):  # keep float8 storage from being promoted by module.to(dtype=...)
        qweight = self._buffers.pop("qweight")
        super()._apply(fn)
        moved = fn(qweight)
        if self.mode in {"int8", "int4", "fp4"} and moved.dtype != qweight.dtype:
            moved = moved.to(qweight.dtype)
        elif self.fp8_dtype is not None and moved.dtype != self.fp8_dtype:
            moved = moved.to(self.fp8_dtype)
        self._buffers["qweight"] = moved
        return self

    def _weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.mode == "int8":
            return self.qweight.to(device=device, dtype=dtype) * self.scale.to(device=device, dtype=dtype)[:, None]
        if self.mode == "int4":
            packed = self.qweight.to(device=device)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = torch.stack((low, high), dim=-1).reshape(self.out_features, -1)[:, : self.in_features]
            unpacked = unpacked.to(torch.int16) - 8
            return unpacked.to(dtype=dtype) * self.scale.to(device=device, dtype=dtype)[:, None]
        if self.mode == "fp4":
            return _dequantize_fp4_e2m1fn(
                self.qweight,
                self.scale,
                out_features=self.out_features,
                in_features=self.in_features,
                dtype=dtype,
                device=device,
            )
        return self.qweight.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._weight(input.dtype, input.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, weight, bias)


def _linear_like_weight_ptr(module: torch.nn.Module) -> Optional[int]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return int(weight.data.data_ptr())
    if isinstance(module, QuantizedLinear):
        return int(module.qweight.data_ptr())
    return None


def _embeddings_share_plain_weight(
    input_embeddings: torch.nn.Module,
    output_embeddings: torch.nn.Module,
) -> bool:
    if isinstance(input_embeddings, QuantizedLinear) or isinstance(output_embeddings, QuantizedLinear):
        return False
    input_ptr = _linear_like_weight_ptr(input_embeddings)
    output_ptr = _linear_like_weight_ptr(output_embeddings)
    return input_ptr is not None and output_ptr is not None and input_ptr == output_ptr


def _empty_quantized_linear_from_linear(
    linear: torch.nn.Linear,
    mode: str,
    *,
    has_bias: bool,
) -> QuantizedLinear:
    out_features = linear.out_features
    in_features = linear.in_features
    mode = mode.lower()
    fp8_dtype = None
    qweight_shape = _expected_quantized_linear_shapes(linear, mode, has_bias=has_bias)["qweight"]
    if mode == "int8":
        qweight = torch.empty(qweight_shape, dtype=torch.int8, device="meta")
        scale = torch.empty((out_features,), dtype=torch.float32, device="meta")
        internal_mode = "int8"
    elif mode == "int4":
        qweight = torch.empty(qweight_shape, dtype=torch.uint8, device="meta")
        scale = torch.empty((out_features,), dtype=torch.float32, device="meta")
        internal_mode = "int4"
    elif mode == "fp4":
        qweight = torch.empty(qweight_shape, dtype=torch.uint8, device="meta")
        scale = torch.empty((out_features,), dtype=torch.float32, device="meta")
        internal_mode = "fp4"
    elif mode in {"fp8_e4m3fn", "fp8"}:
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        qweight = torch.empty(qweight_shape, dtype=fp8_dtype, device="meta")
        scale = None
        internal_mode = "fp8"
    elif mode == "fp8_e5m2":
        fp8_dtype = getattr(torch, "float8_e5m2", torch.uint8)
        qweight = torch.empty(qweight_shape, dtype=fp8_dtype, device="meta")
        scale = None
        internal_mode = "fp8"
    else:
        raise ValueError(f"不支持的量化格式: {mode}")

    bias = torch.empty((out_features,), dtype=linear.weight.dtype, device="meta") if has_bias else None
    return QuantizedLinear.from_quantized_tensors(
        qweight=qweight,
        scale=scale,
        bias=bias,
        in_features=in_features,
        out_features=out_features,
        mode=internal_mode,
        fp8_dtype=fp8_dtype,
    )


def _replace_quantized_modules_from_names(
    model: torch.nn.Module,
    module_names: list[str],
    *,
    mode: str,
    state_dict: dict[str, torch.Tensor],
) -> None:
    for module_name in module_names:
        module = model.get_submodule(module_name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"量化缓存期望 Linear 模块 {module_name!r}，实际为 {type(module).__name__}。")
        _replace_module(
            model,
            module_name,
            _empty_quantized_linear_from_linear(
                module,
                mode,
                has_bias=f"{module_name}.bias" in state_dict,
            ),
        )


def _load_state_dict_assign(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], *, strict: bool):
    try:
        return model.load_state_dict(state_dict, strict=strict, assign=True)
    except TypeError:
        return model.load_state_dict(state_dict, strict=strict)


def _load_safetensors_into_module(
    module: torch.nn.Module,
    path: Path,
    *,
    strict: bool = True,
    progress_label: Optional[str] = None,
) -> None:
    expected_keys = set(module.state_dict().keys())
    loaded_keys: set[str] = set()
    with _PlainSafetensorsReader(path) as reader:
        keys = reader.keys()
        progress = _LanceProgress(len(keys), progress_label) if progress_label else None
        completed = False
        try:
            for key in keys:
                _check_interrupted()
                if key in expected_keys:
                    tensor = reader.get_tensor(key, clone=True)
                    _set_module_tensor(module, key, tensor)
                    loaded_keys.add(key)
                    del tensor
                if progress is not None:
                    progress.update(1)
            completed = True
        finally:
            if completed and progress is not None:
                progress.finish()

    if strict:
        missing = expected_keys - loaded_keys
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise RuntimeError(f"分块加载 safetensors 缺少 {len(missing)} 个 tensor，示例: {sample}")


def _replace_linear_modules(
    module: torch.nn.Module,
    quantization: str,
    *,
    module_names: Optional[list[str]] = None,
    prefix: str = "",
) -> int:
    mode = _normalize_quantization_mode(quantization)
    if mode is None:
        return 0
    count = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child, mode))
            if module_names is not None:
                module_names.append(full_name)
            count += 1
        else:
            count += _replace_linear_modules(child, mode, module_names=module_names, prefix=full_name)
    return count


def _load_lance_checkpoint_streaming_quantized(
    model: torch.nn.Module,
    model_path: Path,
    quantization: str,
    *,
    progress_label: Optional[str] = None,
) -> tuple[int, list[str]]:
    mode = _normalize_quantization_mode(quantization)
    if mode is None:
        raise ValueError("分块量化加载需要启用量化模式。")

    checkpoint_path = _checkpoint_file(model_path)
    expected_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    quantized_module_names: list[str] = []
    start_time = time.monotonic()
    last_report = start_time

    print(f"[ComfyUI-Lance] 分块读取并量化 checkpoint: {checkpoint_path}", flush=True)
    with _PlainSafetensorsReader(checkpoint_path) as shard:
        keys = shard.keys()
        total = len(keys)
        progress = _LanceProgress(total, progress_label) if progress_label else None
        completed = False
        try:
            for index, key in enumerate(keys, start=1):
                _check_interrupted()
                if key == "latent_pos_embed.pos_embed" or key not in expected_keys:
                    if progress is not None:
                        progress.update(1)
                    continue

                module_name, attr_name = _split_tensor_name(key)
                module = model.get_submodule(module_name) if module_name else model
                tensor = shard.get_tensor(key, clone=not (attr_name == "weight" and isinstance(module, torch.nn.Linear)))

                if attr_name == "weight" and isinstance(module, torch.nn.Linear):
                    expected_shape = (int(module.out_features), int(module.in_features))
                    if tuple(tensor.shape) != expected_shape:
                        raise RuntimeError(
                            f"checkpoint 中 {key} 的 shape={tuple(tensor.shape)} 与当前模型结构 {expected_shape} 不一致；"
                            "请检查 Lance latent_patch_size/模型版本是否匹配。"
                        )
                    qweight, scale, internal_mode, fp8_dtype = _quantize_weight_tensor(tensor, mode)
                    bias = None
                    if module.bias is not None:
                        bias = module.bias.detach()
                        if bias.is_floating_point():
                            bias = bias.to(dtype=torch.float32).cpu()
                        else:
                            bias = bias.cpu()
                    _replace_module(
                        model,
                        module_name,
                        QuantizedLinear.from_quantized_tensors(
                            qweight=qweight,
                            scale=scale,
                            bias=bias,
                            in_features=module.in_features,
                            out_features=module.out_features,
                            mode=internal_mode,
                            fp8_dtype=fp8_dtype,
                        ),
                    )
                    quantized_module_names.append(module_name)
                else:
                    tensor = tensor.detach().cpu()
                    _set_module_tensor(model, key, tensor)

                loaded_keys.add(key)
                if progress is not None:
                    progress.update(1)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    print(
                        f"[ComfyUI-Lance] 分块量化进度 {index}/{total}，已量化 {len(quantized_module_names)} 个 Linear。",
                        flush=True,
                    )
                    last_report = now
                del tensor
            completed = True
        finally:
            if completed and progress is not None:
                progress.finish()

    gc.collect()
    print(
        f"[ComfyUI-Lance] 分块量化 checkpoint 完成，用时 {time.monotonic() - start_time:.2f}s: "
        f"{len(quantized_module_names)} 个 Linear。",
        flush=True,
    )
    return len(quantized_module_names), quantized_module_names


@dataclass
class LanceRuntime:
    family: str
    model: Any
    vae_model: Any
    tokenizer: Any
    model_args: Any
    vae_config: Any
    new_token_ids: dict[str, int]
    image_token_id: int
    device: torch.device
    device_index: int
    dtype: torch.dtype
    quantization: str


class LanceModelHandle:
    def __init__(
        self,
        *,
        model_root: Path,
        paths: LancePaths,
        device: torch.device,
        device_index: int,
        dtype: torch.dtype,
        attention_backend: str,
        quantization: str,
        quantization_cache_dir: Optional[Path],
        rebuild_quantization_cache: bool,
        use_kv_cache: bool,
        download_missing: bool,
        download_source: str,
        revision: str,
    ) -> None:
        self.model_root = model_root
        self.paths = paths
        self.device = device
        self.device_index = int(device_index)
        self.dtype = dtype
        self.attention_backend = attention_backend
        self.quantization = quantization
        self.quantization_cache_dir = quantization_cache_dir
        self.rebuild_quantization_cache = bool(rebuild_quantization_cache)
        self.use_kv_cache = bool(use_kv_cache)
        self.download_missing = bool(download_missing)
        self.download_source = download_source
        self.revision = revision

    def _runtime_cache_key(self, family: str) -> tuple[Any, ...]:
        return (
            str(self.model_root.resolve()),
            family,
            str(self.device),
            str(self.dtype),
            self.attention_backend,
            self.quantization,
            str(self.quantization_cache_dir.resolve()) if self.quantization_cache_dir else None,
            self.rebuild_quantization_cache,
            self.use_kv_cache,
            self.revision,
        )

    def preload(self, scope: str) -> str:
        scope = (scope or "auto/lazy").strip().lower()
        if scope in {"image", "图像"}:
            self.get_runtime("image")
            return "已加载 Lance 图像模型。"
        if scope in {"video", "视频"}:
            self.get_runtime("video")
            return "已加载 Lance 视频模型。"
        if scope in {"image+video", "both", "全部"}:
            self.get_runtime("image")
            self.get_runtime("video")
            return "已加载 Lance 图像模型和视频模型。"
        return "已创建 Lance 懒加载句柄；首次运行任务时加载对应模型。"

    def get_runtime(self, family: str) -> LanceRuntime:
        family = family.lower()
        if family not in {"image", "video"}:
            raise ValueError(f"未知 Lance 模型类型: {family}")
        key = self._runtime_cache_key(family)
        cached = _RUNTIME_CACHE.get(key)
        if cached is not None:
            return cached

        with _RUNTIME_LOCK:
            cached = _RUNTIME_CACHE.get(key)
            if cached is not None:
                return cached
            runtime = self._load_runtime(family)
            _RUNTIME_CACHE[key] = runtime
            return runtime

    def _load_runtime(self, family: str) -> LanceRuntime:
        model_label = "图像" if family == "image" else "视频"
        load_progress = _LanceProgress(12, f"加载 Lance {model_label}模型")
        runtime_start = time.monotonic()
        _check_interrupted()
        _ensure_models(self.paths, self.download_missing, self.download_source, self.revision, family)
        load_progress.update(1, "检查 Lance 模型文件")
        _install_lance_path_config(self.paths)
        _patch_lance_device(self.device)
        _configure_attention_backend(self.attention_backend)
        load_progress.update(1, "配置 Lance 推理环境")

        from transformers import set_seed
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

        from common.utils.misc import AutoEncoderParams
        from config.config_factory import InferenceArguments, ModelArguments
        from data.data_utils import add_special_tokens
        from inference_lance import clean_memory
        from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.qwen2.modeling_qwen2 import Qwen2Config
        from modeling.vae.wan.model import WanVideoVAE
        from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel
        load_progress.update(1, "导入 Lance 模块")

        model_path = self.paths.image_model if family == "image" else self.paths.video_model
        quantization_mode = _normalize_quantization_mode(self.quantization)
        cache_path: Optional[Path] = None
        expected_cache_metadata: Optional[dict[str, Any]] = None
        if quantization_mode is not None and self.quantization_cache_dir is not None:
            cache_path, expected_cache_metadata = _quantization_cache_path(
                self.quantization_cache_dir,
                model_path,
                self.paths.vit,
                quantization_mode,
            )

        model_args = ModelArguments(
            model_path=str(model_path),
            llm_path=str(model_path),
            vit_path=str(self.paths.vit),
            vit_type="qwen_2_5_vl_original",
            latent_patch_size=[1, 1, 1],
        )

        inference_args = InferenceArguments(
            visual_gen=True,
            visual_und=True,
            text_template=True,
            apply_qwen_2_5_vl_pos_emb=True,
            use_KVcache=self.use_kv_cache,
        )
        set_seed(inference_args.global_seed)

        llm_config: Qwen2Config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.qk_norm_und = model_args.llm_qk_norm_und
        llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = False
        llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb
        _set_attention_impl(llm_config, _llm_attention_impl(self.attention_backend))
        load_progress.update(1, "读取 LLM 配置")

        language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
        load_progress.update(1, "初始化 LLM 结构")

        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(str(self.paths.vit))
        _set_attention_impl(vit_config, _vit_attention_impl(self.attention_backend))
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
        vit_weights_loaded = False
        load_progress.update(1, "初始化 ViT 结构")

        def load_vit_weights_if_needed() -> None:
            nonlocal vit_weights_loaded
            if vit_weights_loaded:
                return
            _load_safetensors_into_module(
                vit_model,
                self.paths.vit / "vit.safetensors",
                strict=True,
                progress_label="加载 Qwen2.5-VL ViT 权重",
            )
            vit_weights_loaded = True
            _check_interrupted()

        vae_config = AutoEncoderParams(
            downsample_spatial=16,
            downsample_temporal=4,
            z_channels=48,
        )

        config = LanceConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=model_args.latent_patch_size,
            max_num_frames=model_args.max_num_frames,
            max_latent_size=model_args.max_latent_size,
            vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            connector_act=model_args.connector_act,
            interpolate_pos=model_args.interpolate_pos,
            timestep_shift=inference_args.timestep_shift,
        )
        model: Lance = Lance(
            language_model=language_model,
            vit_model=vit_model,
            vit_type=model_args.vit_type,
            config=config,
            inference_args=inference_args,
        )
        _check_interrupted()
        load_progress.update(1, "初始化 Lance 结构")

        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        load_progress.update(1, "加载 tokenizer")

        def apply_tokenizer_shape_and_ties() -> int:
            nonlocal language_model
            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config

                language_model = hack_qwen2_5_vl_config(language_model)

            image_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                output_embeddings = model.language_model.get_output_embeddings()
                if isinstance(output_embeddings, QuantizedLinear):
                    if num_new_tokens > 0:
                        raise RuntimeError("量化 lm_head 不支持在新增 tokenizer token 后执行 tie_word_embeddings 解绑。")
                    print("[ComfyUI-Lance] lm_head 已量化，跳过 tie_word_embeddings 解绑。", flush=True)
                else:
                    model.language_model.untie_lm_head()
                    model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    not _embeddings_share_plain_weight(
                        model.language_model.get_input_embeddings(),
                        model.language_model.get_output_embeddings(),
                    )
                ), "tie_word_embeddings conflict"
            return image_id

        loaded_quantized_cache = False
        quantized_count = 0
        image_token_id = -1
        if cache_path is not None and expected_cache_metadata is not None and cache_path.exists() and not self.rebuild_quantization_cache:
            cache_applied_structural_changes = False
            payload = None
            state_dict = None
            try:
                cache_start = time.monotonic()
                load_progress.update(1, "加载量化缓存")
                print(f"[ComfyUI-Lance] 正在加载量化缓存: {cache_path}", flush=True)
                payload = _torch_load_weights(cache_path, map_location="cpu")
                if payload.get("metadata") != expected_cache_metadata:
                    raise RuntimeError("量化缓存元数据与当前模型文件不匹配")
                state_dict = _unpack_quantized_cache_state_dict(payload)
                quantized_module_names = list(payload["quantized_module_names"])
                _validate_quantized_cache_for_model(
                    model,
                    state_dict,
                    quantized_module_names,
                    mode=quantization_mode or self.quantization,
                )
                image_token_id = apply_tokenizer_shape_and_ties()
                cache_applied_structural_changes = True
                _replace_quantized_modules_from_names(
                    model,
                    quantized_module_names,
                    mode=quantization_mode or self.quantization,
                    state_dict=state_dict,
                )
                _load_state_dict_assign(model, state_dict, strict=True)
                clean_memory(state_dict, payload)
                quantized_count = len(quantized_module_names)
                loaded_quantized_cache = True
                print(
                    f"[ComfyUI-Lance] 量化缓存加载完成，用时 {time.monotonic() - cache_start:.2f}s: "
                    f"{quantized_count} 个 Linear。",
                    flush=True,
                )
            except Exception as exc:
                if cache_applied_structural_changes:
                    raise RuntimeError(
                        "Lance 量化缓存通过元数据检查，但加载到模型结构时失败；请打开“重建量化缓存”刷新缓存。"
                    ) from exc
                clean_memory(state_dict, payload)
                print(f"[ComfyUI-Lance] 无法使用量化缓存（{exc}），改为从原始权重重建。", flush=True)

        if not loaded_quantized_cache:
            load_vit_weights_if_needed()
            load_progress.update(1, "加载 ViT 权重")

            if quantization_mode is not None:
                if inference_args.copy_init_moe:
                    language_model.init_moe()

                quantized_count, quantized_module_names = _load_lance_checkpoint_streaming_quantized(
                    model,
                    model_path,
                    self.quantization,
                    progress_label=f"分块读取并量化 Lance {model_label} checkpoint",
                )
                remaining_count = _replace_linear_modules(
                    model,
                    self.quantization,
                    module_names=quantized_module_names,
                )
                if remaining_count:
                    quantized_count += remaining_count
                    print(
                        f"[ComfyUI-Lance] 已量化 checkpoint 外剩余 Linear: {remaining_count} 个。",
                        flush=True,
                    )
            else:
                if inference_args.copy_init_moe:
                    language_model.init_moe()

                _load_safetensors_into_module(
                    model,
                    _checkpoint_file(model_path),
                    strict=False,
                    progress_label=f"加载 Lance {model_label} checkpoint",
                )
                quantized_module_names = []
                quantized_count = 0
            load_progress.update(1, "加载 Lance checkpoint")

            image_token_id = apply_tokenizer_shape_and_ties()
            if cache_path is not None and expected_cache_metadata is not None and quantized_count:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_start = time.monotonic()
                    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    cache_state_dict, packed_float8_dtypes = _pack_quantized_cache_state_dict(model.state_dict())
                    torch.save(
                        {
                            "metadata": expected_cache_metadata,
                            "quantized_module_names": quantized_module_names,
                            "state_dict": cache_state_dict,
                            "packed_float8_dtypes": packed_float8_dtypes,
                        },
                        str(tmp_path),
                    )
                    os.replace(tmp_path, cache_path)
                    print(
                        f"[ComfyUI-Lance] 已保存量化缓存，用时 {time.monotonic() - cache_start:.2f}s: {cache_path}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[ComfyUI-Lance] 警告：保存量化缓存失败: {exc}", flush=True)

        if quantized_count:
            print(f"[ComfyUI-Lance] {self.quantization} quantized Linear layers: {quantized_count}")

        load_progress.update(1, f"移动 {model_label}模型到 {self.device}")
        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        vae_model = WanVideoVAE(dtype=self.dtype)
        if hasattr(vae_model, "eval"):
            vae_model.eval()
        load_progress.update(1, "初始化 Wan VAE")

        _check_interrupted()
        load_progress.finish(f"Lance {model_label}模型加载完成")
        print(
            f"[ComfyUI-Lance] Lance {model_label}模型总加载用时 {_format_seconds(time.monotonic() - runtime_start)}。",
            flush=True,
        )
        return LanceRuntime(
            family=family,
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            model_args=model_args,
            vae_config=vae_config,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=self.device,
            device_index=self.device_index,
            dtype=self.dtype,
            quantization=self.quantization,
        )

    def run_task(
        self,
        *,
        task: str,
        prompt: str = "",
        question: str = "",
        image_path: str = "",
        video_path: str = "",
        width: int = 768,
        height: int = 768,
        num_frames: int = 50,
        seed: int = 42,
        steps: int = 30,
        cfg_scale: float = 4.0,
        denoise_timestep_shift: float = 3.0,
        cfg_start: float = 0.4,
        cfg_end: float = 1.0,
        resolution: str = "auto",
        max_duration: float = 6.0,
        max_new_tokens: int = 256,
        vae_decode_mode: str = "auto",
        vae_tile_size: int = 384,
        vae_tile_overlap: int = 64,
    ) -> dict[str, Any]:
        _ensure_lance_import_path()
        from torch.utils.data import DataLoader
        from transformers import set_seed

        from config.config_factory import DataArguments, InferenceArguments
        from data.dataset_base import simple_custom_collate
        from data.inference_dataset import InferenceDataset
        from inference_lance import (
            GENERATION_TASKS,
            UNDERSTANDING_TASKS,
            TASK_DEFAULTS,
            apply_inference_defaults,
            build_dataset_config,
            build_direct_input_json,
            clean_memory,
            normalize_task,
            normalize_understanding_answer,
            run_inference_batch,
            save_prompt_results,
            save_understanding_results,
        )

        task = normalize_task(task)
        family = str(TASK_DEFAULTS[task]["model_family"])
        runtime = self.get_runtime(family)
        _patch_lance_device(runtime.device)

        output_dir = _comfy_output_dir() / "Lance" / task / uuid.uuid4().hex[:12]
        output_dir.mkdir(parents=True, exist_ok=True)

        model_args = copy.copy(runtime.model_args)

        inference_args = InferenceArguments(
            task=task,
            prompt=prompt or "",
            question=question or "",
            image=image_path or "",
            video=video_path or "",
            output=str(output_dir),
            save_path_gen=str(output_dir),
            video_width=int(width),
            video_height=int(height),
            num_frames=int(num_frames),
            seed=int(seed),
            global_seed=int(seed),
            num_timesteps=int(steps),
            cfg_scale=float(cfg_scale),
            denoise_timestep_shift=float(denoise_timestep_shift),
            cfg_interval=[float(cfg_start), float(cfg_end)],
            max_duration=float(max_duration),
            use_KVcache=self.use_kv_cache,
            max_samples=1,
            visual_gen=True,
            visual_und=True,
            text_template=True,
            apply_qwen_2_5_vl_pos_emb=True,
        )
        if resolution and resolution != "auto":
            inference_args.resolution = resolution
        inference_args.vae_decode_mode = str(vae_decode_mode or "auto")
        inference_args.vae_tile_size = int(vae_tile_size)
        inference_args.vae_tile_overlap = int(vae_tile_overlap)

        generation_progress = None
        generation_stage_progress = None
        if task in GENERATION_TASKS:
            task_label = "图片" if family == "image" else "视频"
            generation_progress = _LanceProgress(max(1, int(steps)), f"Lance {task_label}去噪生成")
            generation_stage_progress = _LanceProgress(5, f"Lance {task_label}生成阶段")

        def advance_generation_progress(amount: int = 1) -> None:
            if generation_progress is not None:
                generation_progress.update(amount)

        def advance_generation_stage(label: str) -> None:
            if generation_stage_progress is not None:
                generation_stage_progress.update(1, label)

        data_args = DataArguments()
        apply_inference_defaults(model_args, data_args, inference_args)
        inference_args.runtime_dtype = runtime.dtype
        data_args.input_json = build_direct_input_json(inference_args)
        set_seed(inference_args.global_seed)
        inference_args.prompt_data_dict = {}

        max_len_previous = None
        with _INFERENCE_LOCK:
            import inference_lance as inference_module

            max_len_previous = inference_module.MAX_GENERATION_LENGTH
            inference_module.MAX_GENERATION_LENGTH = int(max_new_tokens)
            inference_completed = False
            try:
                _check_interrupted()
                dataset_config = build_dataset_config(
                    data_args.input_json,
                    model_args,
                    inference_args,
                    runtime.vae_config if task in GENERATION_TASKS else None,
                )
                inference_dataset = InferenceDataset(
                    jsonl_path=data_args.input_json,
                    tokenizer=runtime.tokenizer,
                    data_args=data_args,
                    model_args=model_args,
                    inference_args=inference_args,
                    new_token_ids=runtime.new_token_ids,
                    dataset_config=dataset_config,
                    local_rank=0,
                    world_size=1,
                )
                advance_generation_stage("准备输入数据")
                loader = DataLoader(
                    inference_dataset,
                    batch_size=1,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=simple_custom_collate,
                    drop_last=True,
                    prefetch_factor=None,
                    persistent_workers=False,
                    multiprocessing_context=None,
                )

                for batch_cpu in loader:
                    _check_interrupted()
                    run_inference_batch(
                        fsdp_model=runtime.model,
                        vae_model=runtime.vae_model if task in GENERATION_TASKS else None,
                        tokenizer=runtime.tokenizer,
                        batch_cpu=batch_cpu,
                        model_args=model_args,
                        inference_args=inference_args,
                        new_token_ids=runtime.new_token_ids,
                        image_token_id=runtime.image_token_id,
                        device=runtime.device_index,
                        save_source_video=False,
                        save_path_gen=str(output_dir),
                        save_path_gt="",
                        progress_callback=advance_generation_progress if task in GENERATION_TASKS else None,
                        stage_callback=advance_generation_stage if task in GENERATION_TASKS else None,
                    )
                    del batch_cpu
                    clean_memory()
                    _check_interrupted()

                save_prompt_results(inference_args.prompt_data_dict, str(output_dir), None)
                if task in UNDERSTANDING_TASKS:
                    save_understanding_results(inference_args.prompt_data_dict, data_args.input_json, str(output_dir))
                inference_completed = True
            finally:
                inference_module.MAX_GENERATION_LENGTH = max_len_previous
                if inference_completed and generation_progress is not None:
                    generation_progress.finish()
                if inference_completed and generation_stage_progress is not None:
                    generation_stage_progress.finish()

        if task in UNDERSTANDING_TASKS:
            answer = ""
            for value in inference_args.prompt_data_dict.values():
                answer = normalize_understanding_answer(value)
                break
            return {"task": task, "answer": answer, "output_dir": str(output_dir)}

        result_file = ""
        if inference_args.prompt_data_dict:
            first_name = next(iter(inference_args.prompt_data_dict.keys()))
            candidate = output_dir / first_name
            if candidate.is_file():
                result_file = str(candidate)
        if not result_file:
            candidates = sorted(
                list(output_dir.glob("*.png")) + list(output_dir.glob("*.mp4")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                result_file = str(candidates[0])
        if not result_file:
            raise RuntimeError(f"Lance 未生成可用结果，输出目录: {output_dir}")

        return {"task": task, "path": result_file, "output_dir": str(output_dir)}

    def release(self, clear_cuda_cache: bool = True) -> str:
        keys = [key for key in _RUNTIME_CACHE if key[0] == str(self.model_root.resolve()) and key[2] == str(self.device)]
        for key in keys:
            runtime = _RUNTIME_CACHE.pop(key, None)
            if runtime is not None:
                runtime.model = None
                runtime.vae_model = None
                runtime.tokenizer = None
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
        if clear_cuda_cache and model_management is not None and hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
        return f"已释放 {len(keys)} 个 Lance 运行时。"


def _save_comfy_image(image: torch.Tensor, name: str, batch_index: int = 0) -> str:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image 输入必须是 ComfyUI IMAGE tensor。")
    if image.ndim != 4:
        raise ValueError(f"IMAGE tensor 形状应为 [B,H,W,C]，实际为 {tuple(image.shape)}")
    index = max(0, min(int(batch_index), image.shape[0] - 1))
    frame = image[index].detach().float().cpu().clamp(0, 1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    array = (frame.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    path = _comfy_temp_dir() / "Lance" / f"{name}_{uuid.uuid4().hex[:10]}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return str(path)


def _comfy_image_from_file(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _video_to_path(video: Any, name: str) -> str:
    if video is None:
        raise ValueError("video 输入不能为空。")
    path = _comfy_temp_dir() / "Lance" / f"{name}_{uuid.uuid4().hex[:10]}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(video, "save_to"):
        if Types is not None:
            video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
        else:
            video.save_to(str(path))
        return str(path)
    if hasattr(video, "get_stream_source"):
        try:
            source = video.get_stream_source()
            if isinstance(source, str) and Path(source).is_file():
                return source
        except Exception:
            pass
    raise TypeError("video 输入必须是 ComfyUI VIDEO 对象。")


def _video_from_file(path: str) -> Any:
    if InputImpl is None:
        return path
    return InputImpl.VideoFromFile(path)


def _video_frames_from_file(path: str) -> torch.Tensor:
    video = _video_from_file(path)
    if hasattr(video, "get_components"):
        return video.get_components().images
    return torch.zeros(0, 0, 0, 3)


def _images_to_video(images: torch.Tensor, fps: float = 12.0) -> Any:
    if InputImpl is None or Types is None:
        return images
    components = Types.VideoComponents(images=images, audio=None, frame_rate=Fraction(round(float(fps) * 1000), 1000))
    return InputImpl.VideoFromComponents(components)


def _resolution_for_image(resolution: str) -> str:
    return "auto" if resolution == "auto" else resolution


def _resolution_for_video(resolution: str) -> str:
    return "auto" if resolution == "auto" else resolution


COMMON_GENERATION_INPUTS = {
    "seed": ("INT", _ui("Seed", "采样随机种子。", default=42, min=0, max=2**31 - 1, step=1)),
    "steps": ("INT", _ui("Denoise Steps", "去噪步数；步数越高通常越慢。", default=30, min=1, max=200, step=1)),
    "cfg_scale": ("FLOAT", _ui("CFG Scale", "文本 CFG 强度。", default=4.0, min=0.1, max=30.0, step=0.1)),
    "denoise_timestep_shift": (
        "FLOAT",
        _ui("Timestep Shift", "去噪 timestep shift，沿用 Lance demo 默认值。", default=3.0, min=0.1, max=20.0, step=0.1),
    ),
    "cfg_start": ("FLOAT", _ui("CFG Start", "CFG 生效区间起点。", default=0.4, min=0.0, max=1.0, step=0.01)),
    "cfg_end": ("FLOAT", _ui("CFG End", "CFG 生效区间终点。", default=1.0, min=0.0, max=1.0, step=0.01)),
    "vae_decode_mode": (
        ["auto", "normal", "tiled"],
        _ui("VAE Decode Mode", "auto 在 ROCm/HIP 下自动启用空间分块；normal 使用原始解码；tiled 强制空间分块。", default="auto"),
    ),
    "vae_tile_size": (
        "INT",
        _ui("VAE Tile Size", "VAE 空间分块的输出像素尺寸；仅在 tiled/auto 分块时生效。", default=384, min=128, max=2048, step=16),
    ),
    "vae_tile_overlap": (
        "INT",
        _ui("VAE Tile Overlap", "VAE 分块拼接的输出像素重叠宽度；重叠越大接缝越少但更慢。", default=64, min=0, max=512, step=16),
    ),
}


class LanceLoadModel:
    CATEGORY = CATEGORY
    DESCRIPTION = "从 ComfyUI/models/Lance 加载 bytedance-research/Lance，支持设备选择、FlashAttention/SageAttention 选项和 int8/int4/fp4/fp8 量化包装。"
    RETURN_TYPES = ("LANCE_MODEL", "STRING")
    RETURN_NAMES = ("lance_model", "状态")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": (
                    "STRING",
                    _ui(
                        "模型根目录",
                        "auto 使用 ComfyUI/models/Lance；也可以填写自定义绝对路径。",
                        default="auto",
                    ),
                ),
                "model_scope": (
                    ["auto/lazy", "image", "video", "image+video"],
                    _ui("加载范围", "auto/lazy 会在首次任务运行时按需加载对应 image/video 模型。", default="auto/lazy"),
                ),
                "device": (_device_choices(), _ui("设备", "选择 Lance 推理使用的 CUDA 设备。", default="auto")),
                "compute_dtype": (
                    ["bf16", "fp16", "fp32"],
                    _ui("Compute dtype", "模型计算精度；Lance demo 默认使用 bf16。", default="bf16"),
                ),
                "attention_backend": (
                    ["auto", "flash_attention_2", "sage_attention", "sdpa"],
                    _ui("Attention Backend", "auto/flash_attention_2 使用 FlashAttention（若可用）；sage_attention 需要 sageattention 包。", default="auto"),
                ),
                "quantization": (
                    ["none", "int8", "int4", "fp4", "fp8_e4m3fn", "fp8_e5m2"],
                    _ui("量化加载", "实验性 Linear 量化；生成质量异常时请先使用 none 原始精度。", default="none"),
                ),
                "use_quantization_cache": (
                    "BOOLEAN",
                    _ui(
                        "量化缓存",
                        "启用后将可复用量化权重保存到 ComfyUI/models/Lance-quantized-cache。",
                        default=True,
                    ),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    _ui("重建量化缓存", "忽略已有量化缓存并重新从原始权重量化生成。", default=False),
                ),
                "use_kv_cache": (
                    "BOOLEAN",
                    _ui("启用 KV Cache", "生成视觉内容时启用 Lance 官方 KV cache 路径；可降低重复注意力计算。", default=True),
                ),
            },
            "optional": {
                "download_missing": (
                    "BOOLEAN",
                    _ui("缺失时下载", "模型缺失时从 Hugging Face 仓库下载到模型根目录。", default=False),
                ),
                "download_source": (
                    ["huggingface.co", "hf-mirror.com"],
                    _ui("下载源", "自动下载使用的源。", default="huggingface.co"),
                ),
                "revision": (
                    "STRING",
                    _ui("Revision", "Hugging Face revision；留空使用默认分支。", default=""),
                ),
            },
        }

    def load(
        self,
        model_root: str,
        model_scope: str,
        device: str,
        compute_dtype: str,
        attention_backend: str,
        quantization: str,
        use_quantization_cache: bool = True,
        rebuild_quantization_cache: bool = False,
        use_kv_cache: bool = True,
        download_missing: bool = False,
        download_source: str = "huggingface.co",
        revision: str = "",
    ):
        model_root_path = _resolve_model_root(model_root)
        paths = _lance_paths(model_root_path)
        device_obj, device_index = _resolve_device(device)
        quantization_mode = _normalize_quantization_mode(quantization)
        quantization_cache_dir = _lance_quantization_cache_dir() if use_quantization_cache and quantization_mode else None
        handle = LanceModelHandle(
            model_root=model_root_path,
            paths=paths,
            device=device_obj,
            device_index=device_index,
            dtype=_dtype_from_name(compute_dtype),
            attention_backend=attention_backend,
            quantization=quantization,
            quantization_cache_dir=quantization_cache_dir,
            rebuild_quantization_cache=rebuild_quantization_cache,
            use_kv_cache=use_kv_cache,
            download_missing=download_missing,
            download_source=download_source,
            revision=revision or "",
        )
        status = handle.preload(model_scope)
        cache_status = f"\n量化缓存目录: {quantization_cache_dir}" if quantization_cache_dir is not None else ""
        kv_cache_status = "开启" if use_kv_cache else "关闭"
        return (handle, f"{status}\n模型根目录: {model_root_path}\n设备: {device_obj}\nKV Cache: {kv_cache_status}{cache_status}")


class LanceImageUnderstanding:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 进行图片理解，输入/输出使用 ComfyUI IMAGE 和 STRING。"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回答",)
    FUNCTION = "understand"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "image": ("IMAGE", _ui("图片", "ComfyUI IMAGE 输入。")),
                "question": (
                    "STRING",
                    _ui("问题", "图片理解问题；可使用中文或英文。", default="请详细描述这张图片。", multiline=True),
                ),
                "max_new_tokens": (
                    "INT",
                    _ui("Max New Tokens", "理解任务最大生成 token 数。", default=256, min=1, max=2048, step=1),
                ),
                "batch_index": (
                    "INT",
                    _ui("Batch Index", "从批量 IMAGE 中选择第几张图。", default=0, min=0, max=4096, step=1),
                ),
            }
        }

    def understand(self, lance_model: LanceModelHandle, image: torch.Tensor, question: str, max_new_tokens: int, batch_index: int):
        image_path = _save_comfy_image(image, "image_understanding", batch_index)
        result = lance_model.run_task(
            task="x2t_image",
            question=question,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )
        answer = result["answer"]
        return {"ui": {"text": [answer]}, "result": (answer,)}


class LanceImageGeneration:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 文生图，输出 ComfyUI IMAGE。"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "路径")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "prompt": ("STRING", _ui("Prompt", "文生图提示词。", default="", multiline=True)),
                "width": ("INT", _ui("宽度", "目标图片宽度。", default=768, min=128, max=2048, step=8)),
                "height": ("INT", _ui("高度", "目标图片高度。", default=768, min=128, max=2048, step=8)),
                **COMMON_GENERATION_INPUTS,
                "resolution": (
                    ["auto", "image_768res", "image_512res", "image_256res"],
                    _ui("Resolution Preset", "Lance 数据预处理 resolution 预设；auto 使用 demo 默认值。", default="auto"),
                ),
            }
        }

    def generate(
        self,
        lance_model: LanceModelHandle,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg_scale: float,
        denoise_timestep_shift: float,
        cfg_start: float,
        cfg_end: float,
        vae_decode_mode: str,
        vae_tile_size: int,
        vae_tile_overlap: int,
        resolution: str,
    ):
        result = lance_model.run_task(
            task="t2i",
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            denoise_timestep_shift=denoise_timestep_shift,
            cfg_start=cfg_start,
            cfg_end=cfg_end,
            vae_decode_mode=vae_decode_mode,
            vae_tile_size=vae_tile_size,
            vae_tile_overlap=vae_tile_overlap,
            resolution=_resolution_for_image(resolution),
        )
        return (_comfy_image_from_file(result["path"]), result["path"])


class LanceImageEditing:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 图片编辑，输入/输出使用 ComfyUI IMAGE。"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图片", "路径")
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "image": ("IMAGE", _ui("输入图片", "要编辑的 ComfyUI IMAGE。")),
                "prompt": ("STRING", _ui("编辑指令", "图片编辑指令。", default="", multiline=True)),
                "width": ("INT", _ui("宽度", "目标图片宽度。", default=768, min=128, max=2048, step=8)),
                "height": ("INT", _ui("高度", "目标图片高度。", default=768, min=128, max=2048, step=8)),
                **COMMON_GENERATION_INPUTS,
                "resolution": (
                    ["auto", "image_768res", "image_512res", "image_256res"],
                    _ui("Resolution Preset", "Lance 数据预处理 resolution 预设；auto 使用 demo 默认值。", default="auto"),
                ),
                "batch_index": (
                    "INT",
                    _ui("Batch Index", "从批量 IMAGE 中选择第几张图。", default=0, min=0, max=4096, step=1),
                ),
            }
        }

    def edit(
        self,
        lance_model: LanceModelHandle,
        image: torch.Tensor,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg_scale: float,
        denoise_timestep_shift: float,
        cfg_start: float,
        cfg_end: float,
        vae_decode_mode: str,
        vae_tile_size: int,
        vae_tile_overlap: int,
        resolution: str,
        batch_index: int,
    ):
        image_path = _save_comfy_image(image, "image_edit", batch_index)
        result = lance_model.run_task(
            task="image_edit",
            prompt=prompt,
            image_path=image_path,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            denoise_timestep_shift=denoise_timestep_shift,
            cfg_start=cfg_start,
            cfg_end=cfg_end,
            vae_decode_mode=vae_decode_mode,
            vae_tile_size=vae_tile_size,
            vae_tile_overlap=vae_tile_overlap,
            resolution=_resolution_for_image(resolution),
        )
        return (_comfy_image_from_file(result["path"]), result["path"])


class LanceVideoUnderstanding:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 进行视频理解，输入/输出使用 ComfyUI VIDEO 和 STRING。"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回答",)
    FUNCTION = "understand"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "video": ("VIDEO", _ui("视频", "ComfyUI VIDEO 输入。")),
                "question": (
                    "STRING",
                    _ui("问题", "视频理解问题；可使用中文或英文。", default="请详细描述这个视频。", multiline=True),
                ),
                "max_new_tokens": (
                    "INT",
                    _ui("Max New Tokens", "理解任务最大生成 token 数。", default=256, min=1, max=2048, step=1),
                ),
                "max_duration": (
                    "FLOAT",
                    _ui("Max Duration", "视频采样时允许的最大时长，单位秒。", default=6.0, min=0.1, max=120.0, step=0.1),
                ),
            }
        }

    def understand(self, lance_model: LanceModelHandle, video: Any, question: str, max_new_tokens: int, max_duration: float):
        video_path = _video_to_path(video, "video_understanding")
        result = lance_model.run_task(
            task="x2t_video",
            question=question,
            video_path=video_path,
            max_duration=max_duration,
            max_new_tokens=max_new_tokens,
        )
        answer = result["answer"]
        return {"ui": {"text": [answer]}, "result": (answer,)}


VIDEO_GENERATION_INPUTS = {
    "width": ("INT", _ui("宽度", "目标视频宽度。", default=848, min=128, max=2048, step=8)),
    "height": ("INT", _ui("高度", "目标视频高度。", default=480, min=128, max=2048, step=8)),
    "num_frames": ("INT", _ui("帧数", "目标视频帧数。", default=50, min=2, max=257, step=1)),
    **COMMON_GENERATION_INPUTS,
    "resolution": (
        ["auto", "video_480p", "video_360p", "video_192p"],
        _ui("Resolution Preset", "Lance 数据预处理 resolution 预设；auto 使用 demo 默认值。", default="auto"),
    ),
    "max_duration": (
        "FLOAT",
        _ui("Max Duration", "视频采样时允许的最大时长，单位秒。", default=6.0, min=0.1, max=120.0, step=0.1),
    ),
}


class LanceTextToVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 文生视频，输出 ComfyUI VIDEO 和帧 IMAGE。"
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("视频", "帧", "路径")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "prompt": ("STRING", _ui("Prompt", "文生视频提示词。", default="", multiline=True)),
                **VIDEO_GENERATION_INPUTS,
            }
        }

    def generate(self, lance_model: LanceModelHandle, prompt: str, **kwargs):
        result = lance_model.run_task(
            task="t2v",
            prompt=prompt,
            width=kwargs["width"],
            height=kwargs["height"],
            num_frames=kwargs["num_frames"],
            seed=kwargs["seed"],
            steps=kwargs["steps"],
            cfg_scale=kwargs["cfg_scale"],
            denoise_timestep_shift=kwargs["denoise_timestep_shift"],
            cfg_start=kwargs["cfg_start"],
            cfg_end=kwargs["cfg_end"],
            vae_decode_mode=kwargs["vae_decode_mode"],
            vae_tile_size=kwargs["vae_tile_size"],
            vae_tile_overlap=kwargs["vae_tile_overlap"],
            resolution=_resolution_for_video(kwargs["resolution"]),
            max_duration=kwargs["max_duration"],
        )
        video = _video_from_file(result["path"])
        frames = _video_frames_from_file(result["path"])
        return (video, frames, result["path"])


class LanceImageToVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 图生视频，输入 IMAGE，输出 ComfyUI VIDEO。"
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("视频", "帧", "路径")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "image": ("IMAGE", _ui("参考图片", "图生视频的输入图片。")),
                "prompt": ("STRING", _ui("Prompt", "图生视频提示词。", default="", multiline=True)),
                **VIDEO_GENERATION_INPUTS,
                "batch_index": (
                    "INT",
                    _ui("Batch Index", "从批量 IMAGE 中选择第几张图。", default=0, min=0, max=4096, step=1),
                ),
            }
        }

    def generate(self, lance_model: LanceModelHandle, image: torch.Tensor, prompt: str, batch_index: int, **kwargs):
        image_path = _save_comfy_image(image, "image_to_video", batch_index)
        result = lance_model.run_task(
            task="i2v",
            prompt=prompt,
            image_path=image_path,
            width=kwargs["width"],
            height=kwargs["height"],
            num_frames=kwargs["num_frames"],
            seed=kwargs["seed"],
            steps=kwargs["steps"],
            cfg_scale=kwargs["cfg_scale"],
            denoise_timestep_shift=kwargs["denoise_timestep_shift"],
            cfg_start=kwargs["cfg_start"],
            cfg_end=kwargs["cfg_end"],
            vae_decode_mode=kwargs["vae_decode_mode"],
            vae_tile_size=kwargs["vae_tile_size"],
            vae_tile_overlap=kwargs["vae_tile_overlap"],
            resolution=_resolution_for_video(kwargs["resolution"]),
            max_duration=kwargs["max_duration"],
        )
        video = _video_from_file(result["path"])
        frames = _video_frames_from_file(result["path"])
        return (video, frames, result["path"])


class LanceVideoEditing:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 视频编辑，输入/输出使用 ComfyUI VIDEO。"
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("视频", "帧", "路径")
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "Lance Load Model 节点输出。")),
                "video": ("VIDEO", _ui("输入视频", "要编辑的 ComfyUI VIDEO。")),
                "prompt": ("STRING", _ui("编辑指令", "视频编辑指令。", default="", multiline=True)),
                **VIDEO_GENERATION_INPUTS,
            }
        }

    def edit(self, lance_model: LanceModelHandle, video: Any, prompt: str, **kwargs):
        video_path = _video_to_path(video, "video_edit")
        result = lance_model.run_task(
            task="video_edit",
            prompt=prompt,
            video_path=video_path,
            width=kwargs["width"],
            height=kwargs["height"],
            num_frames=kwargs["num_frames"],
            seed=kwargs["seed"],
            steps=kwargs["steps"],
            cfg_scale=kwargs["cfg_scale"],
            denoise_timestep_shift=kwargs["denoise_timestep_shift"],
            cfg_start=kwargs["cfg_start"],
            cfg_end=kwargs["cfg_end"],
            vae_decode_mode=kwargs["vae_decode_mode"],
            vae_tile_size=kwargs["vae_tile_size"],
            vae_tile_overlap=kwargs["vae_tile_overlap"],
            resolution=_resolution_for_video(kwargs["resolution"]),
            max_duration=kwargs["max_duration"],
        )
        video_out = _video_from_file(result["path"])
        frames = _video_frames_from_file(result["path"])
        return (video_out, frames, result["path"])


class LanceFramesToVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = "将 ComfyUI IMAGE 帧打包为 ComfyUI VIDEO，便于和 Lance 视频节点连接。"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("视频",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", _ui("帧", "ComfyUI IMAGE 帧序列。")),
                "fps": ("FLOAT", _ui("FPS", "输出 VIDEO 的帧率。", default=12.0, min=0.1, max=240.0, step=0.1)),
            }
        }

    def convert(self, images: torch.Tensor, fps: float):
        return (_images_to_video(images, fps),)


class LanceUnloadModel:
    CATEGORY = CATEGORY
    DESCRIPTION = "释放 Lance 模型缓存，并可清理 CUDA 显存。"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("状态",)
    FUNCTION = "release"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("Lance 模型", "要释放的 Lance 模型句柄。")),
                "clear_cuda_cache": ("BOOLEAN", _ui("清理 CUDA Cache", "释放后清理 CUDA cache。", default=True)),
            }
        }

    def release(self, lance_model: LanceModelHandle, clear_cuda_cache: bool):
        return (lance_model.release(clear_cuda_cache=bool(clear_cuda_cache)),)


NODE_CLASS_MAPPINGS = {
    "LanceLoadModel": LanceLoadModel,
    "LanceImageUnderstanding": LanceImageUnderstanding,
    "LanceImageGeneration": LanceImageGeneration,
    "LanceImageEditing": LanceImageEditing,
    "LanceVideoUnderstanding": LanceVideoUnderstanding,
    "LanceImageToVideo": LanceImageToVideo,
    "LanceTextToVideo": LanceTextToVideo,
    "LanceVideoEditing": LanceVideoEditing,
    "LanceFramesToVideo": LanceFramesToVideo,
    "LanceUnloadModel": LanceUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LanceLoadModel": "Lance 加载模型",
    "LanceImageUnderstanding": "Lance 图片理解",
    "LanceImageGeneration": "Lance 图片生成",
    "LanceImageEditing": "Lance 图片编辑",
    "LanceVideoUnderstanding": "Lance 视频理解",
    "LanceImageToVideo": "Lance 图片生成视频",
    "LanceTextToVideo": "Lance 文字生成视频",
    "LanceVideoEditing": "Lance 视频编辑",
    "LanceFramesToVideo": "Lance 帧转视频",
    "LanceUnloadModel": "Lance 释放模型",
}
