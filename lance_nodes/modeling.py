"""Model paths, device/dtype selection, Lance environment patches, and attention setup."""

from .common import *

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


def _llm_dtype_from_name(name: str) -> torch.dtype:
    selected = (name or "bf16").strip().lower()
    if selected in {"fp16", "float16"}:
        raise ValueError("LLM Compute dtype 已禁用 fp16；请改用 bf16 或 fp32。")
    return _dtype_from_name(selected)


def _diffusion_dtype_from_name(name: str, llm_dtype: torch.dtype) -> torch.dtype:
    selected = (name or "same as llm").strip().lower()
    if selected in {"same as llm", "same as compute", "same", "auto", "跟随 llm", "跟随 compute", "跟随模型"}:
        return llm_dtype
    return _dtype_from_name(selected)


def _vae_dtype_from_name(name: str, llm_dtype: torch.dtype, diffusion_dtype: torch.dtype) -> torch.dtype:
    selected = (name or "same as diffusion").strip().lower()
    if selected in {"same as diffusion", "same as compute", "same", "auto", "跟随 diffusion", "跟随 compute", "跟随模型"}:
        return diffusion_dtype
    if selected in {"same as llm", "跟随 llm"}:
        return llm_dtype
    return _dtype_from_name(selected)


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




__all__ = [name for name in globals() if not name.startswith("__")]
