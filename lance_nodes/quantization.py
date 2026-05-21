"""Streaming safetensors loading and optional Linear quantization support."""

from .common import *
from .modeling import *

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




__all__ = [name for name in globals() if not name.startswith("__")]
