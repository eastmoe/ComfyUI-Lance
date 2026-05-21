"""Lazy Lance runtime loading, caching, and task execution."""

from .common import *
from .modeling import *
from .quantization import *

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
    llm_dtype: torch.dtype
    dtype: torch.dtype
    vae_dtype: torch.dtype
    quantization: str


class LanceModelHandle:
    def __init__(
        self,
        *,
        model_root: Path,
        paths: LancePaths,
        device: torch.device,
        device_index: int,
        llm_dtype: torch.dtype,
        dtype: torch.dtype,
        vae_dtype: torch.dtype,
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
        self.llm_dtype = llm_dtype
        self.dtype = dtype
        self.vae_dtype = vae_dtype
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
            str(self.llm_dtype),
            str(self.dtype),
            str(self.vae_dtype),
            self.attention_backend,
            self.quantization,
            str(self.quantization_cache_dir.resolve()) if self.quantization_cache_dir else None,
            self.rebuild_quantization_cache,
            self.use_kv_cache,
            self.revision,
        )

    def evict_stale_runtimes(self) -> int:
        model_root_key = str(self.model_root.resolve())
        current_keys = {self._runtime_cache_key("image"), self._runtime_cache_key("video")}
        evicted = _evict_runtime_cache(
            lambda key: len(key) > 0 and key[0] == model_root_key and key not in current_keys,
            clear_cuda_cache=True,
        )
        if evicted:
            print(f"[ComfyUI-Lance] 加载参数变化，已释放 {evicted} 个旧 Lance 运行时。", flush=True)
        return evicted

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
            model_root_key = str(self.model_root.resolve())
            stale_count = _evict_runtime_cache(
                lambda cached_key: (
                    len(cached_key) > 1
                    and cached_key[0] == model_root_key
                    and cached_key[1] == family
                    and cached_key != key
                ),
                clear_cuda_cache=True,
            )
            if stale_count:
                print(f"[ComfyUI-Lance] 重新加载 Lance {family} 运行时前已释放 {stale_count} 个旧运行时。", flush=True)
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
        if hasattr(model, "language_model"):
            model.language_model.to(device=self.device, dtype=self.llm_dtype)
        model.eval()
        vae_model = WanVideoVAE(dtype=self.vae_dtype)
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
            llm_dtype=self.llm_dtype,
            dtype=self.dtype,
            vae_dtype=self.vae_dtype,
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
        inference_args.llm_runtime_dtype = runtime.llm_dtype
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
        model_root_key = str(self.model_root.resolve())
        device_key = str(self.device)
        evicted = _evict_runtime_cache(
            lambda key: len(key) > 2 and key[0] == model_root_key and key[2] == device_key,
            clear_cuda_cache=clear_cuda_cache,
        )
        return f"已释放 {evicted} 个 Lance 运行时。"




__all__ = [name for name in globals() if not name.startswith("__")]
