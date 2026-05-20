from __future__ import annotations

import copy
import gc
import os
import sys
import tempfile
import threading
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
        mode = mode.lower()
        if mode == "int8":
            scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            qweight = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
            return cls(qweight, scale, bias, module.in_features, module.out_features, mode)
        if mode == "int4":
            scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 7.0
            q = torch.round(weight / scale[:, None]).clamp(-8, 7).to(torch.int8)
            q = (q + 8).to(torch.uint8)
            if q.shape[1] % 2:
                q = torch.nn.functional.pad(q, (0, 1))
            low = q[:, 0::2]
            high = q[:, 1::2] << 4
            qweight = low | high
            return cls(qweight, scale, bias, module.in_features, module.out_features, mode)
        if mode in {"fp8_e4m3fn", "fp8"}:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("当前 PyTorch 不支持 torch.float8_e4m3fn。")
            dtype = torch.float8_e4m3fn
            return cls(weight.to(dtype), None, bias, module.in_features, module.out_features, "fp8", dtype)
        if mode == "fp8_e5m2":
            if not hasattr(torch, "float8_e5m2"):
                raise RuntimeError("当前 PyTorch 不支持 torch.float8_e5m2。")
            dtype = torch.float8_e5m2
            return cls(weight.to(dtype), None, bias, module.in_features, module.out_features, "fp8", dtype)
        raise ValueError(f"不支持的量化格式: {mode}")

    def _apply(self, fn):  # keep float8 storage from being promoted by module.to(dtype=...)
        qweight = self._buffers.pop("qweight")
        super()._apply(fn)
        moved = fn(qweight)
        if self.mode in {"int8", "int4"} and moved.dtype != qweight.dtype:
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
        return self.qweight.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._weight(input.dtype, input.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, weight, bias)


def _replace_linear_modules(module: torch.nn.Module, quantization: str) -> int:
    mode = (quantization or "none").strip().lower()
    if mode in {"none", "off", "false"}:
        return 0
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QuantizedLinear.from_linear(child, mode))
            count += 1
        else:
            count += _replace_linear_modules(child, mode)
    return count


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
        _check_interrupted()
        _ensure_models(self.paths, self.download_missing, self.download_source, self.revision, family)
        _install_lance_path_config(self.paths)
        _patch_lance_device(self.device)
        _configure_attention_backend(self.attention_backend)

        from safetensors.torch import load_file
        from transformers import set_seed
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

        from common.utils.misc import AutoEncoderParams
        from config.config_factory import InferenceArguments, ModelArguments
        from data.data_utils import add_special_tokens
        from inference_lance import clean_memory, init_from_model_path_if_needed
        from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.qwen2.modeling_qwen2 import Qwen2Config
        from modeling.vae.wan.model import WanVideoVAE
        from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

        model_path = self.paths.image_model if family == "image" else self.paths.video_model
        model_args = ModelArguments(
            model_path=str(model_path),
            llm_path=str(model_path),
            vit_path=str(self.paths.vit),
        )

        inference_args = InferenceArguments(
            visual_gen=True,
            visual_und=True,
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

        language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)

        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(str(self.paths.vit))
        _set_attention_impl(vit_config, _vit_attention_impl(self.attention_backend))
        vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
        vit_weights = load_file(str(self.paths.vit / "vit.safetensors"))
        vit_model.load_state_dict(vit_weights, strict=True)
        clean_memory(vit_weights)
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

        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

        if inference_args.copy_init_moe:
            language_model.init_moe()

        init_from_model_path_if_needed(model, model_args)

        if num_new_tokens > 0:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        if model_args.vit_type.lower() == "qwen2_5_vl":
            from common.model.hacks import hack_qwen2_5_vl_config

            language_model = hack_qwen2_5_vl_config(language_model)

        image_token_id = language_model.config.video_token_id
        new_token_ids.update({"image_token_id": image_token_id})
        model.update_tokenizer(tokenizer=tokenizer)

        if model_args.tie_word_embeddings:
            model.language_model.untie_lm_head()
            model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
            model_args.tie_word_embeddings = False
            llm_config.tie_word_embeddings = False
        else:
            assert (
                model.language_model.get_input_embeddings().weight.data.data_ptr()
                != model.language_model.get_output_embeddings().weight.data.data_ptr()
            ), "tie_word_embeddings conflict"

        quantized_count = _replace_linear_modules(model, self.quantization)
        if quantized_count:
            print(f"[ComfyUI-Lance] {self.quantization} quantized Linear layers: {quantized_count}")

        model = model.to(device=self.device, dtype=self.dtype)
        model.eval()
        vae_model = WanVideoVAE(dtype=self.dtype)
        if hasattr(vae_model, "eval"):
            vae_model.eval()

        _check_interrupted()
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
        )
        if resolution and resolution != "auto":
            inference_args.resolution = resolution

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
                    )
                    del batch_cpu
                    clean_memory()
                    _check_interrupted()

                save_prompt_results(inference_args.prompt_data_dict, str(output_dir), None)
                if task in UNDERSTANDING_TASKS:
                    save_understanding_results(inference_args.prompt_data_dict, data_args.input_json, str(output_dir))
            finally:
                inference_module.MAX_GENERATION_LENGTH = max_len_previous

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
}


class LanceLoadModel:
    CATEGORY = CATEGORY
    DESCRIPTION = "从 ComfyUI/models/Lance 加载 bytedance-research/Lance，支持设备选择、FlashAttention/SageAttention 选项和 int8/int4/fp8 量化包装。"
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
                    ["none", "int8", "int4", "fp8_e4m3fn", "fp8_e5m2"],
                    _ui("量化加载", "对 Linear 权重做轻量量化包装；none 为原始精度。", default="none"),
                ),
                "use_kv_cache": (
                    "BOOLEAN",
                    _ui("KV Cache", "生成视觉内容时启用 Lance 的 KV cache 路径；实验选项。", default=False),
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
        use_kv_cache: bool,
        download_missing: bool = False,
        download_source: str = "huggingface.co",
        revision: str = "",
    ):
        model_root_path = _resolve_model_root(model_root)
        paths = _lance_paths(model_root_path)
        device_obj, device_index = _resolve_device(device)
        handle = LanceModelHandle(
            model_root=model_root_path,
            paths=paths,
            device=device_obj,
            device_index=device_index,
            dtype=_dtype_from_name(compute_dtype),
            attention_backend=attention_backend,
            quantization=quantization,
            use_kv_cache=use_kv_cache,
            download_missing=download_missing,
            download_source=download_source,
            revision=revision or "",
        )
        status = handle.preload(model_scope)
        return (handle, f"{status}\n模型根目录: {model_root_path}\n设备: {device_obj}")


class LanceImageUnderstanding:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 进行图片理解，输入/输出使用 ComfyUI IMAGE 和 STRING。"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回答",)
    FUNCTION = "understand"

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
        return (result["answer"],)


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
            resolution=_resolution_for_image(resolution),
        )
        return (_comfy_image_from_file(result["path"]), result["path"])


class LanceVideoUnderstanding:
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 Lance 进行视频理解，输入/输出使用 ComfyUI VIDEO 和 STRING。"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("回答",)
    FUNCTION = "understand"

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
        return (result["answer"],)


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
