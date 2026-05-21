"""ComfyUI node classes and exported node mappings."""

from .common import *
from .modeling import *
from .runtime import LanceModelHandle
from .media import *

COMMON_GENERATION_INPUTS = {
    "seed": ("INT", _ui("common_generation.seed", "Seed", "Sampling random seed.", default=42, min=0, max=2**31 - 1, step=1)),
    "steps": ("INT", _ui("common_generation.steps", "Denoise Steps", "Denoising steps; higher values are usually slower.", default=30, min=1, max=200, step=1)),
    "cfg_scale": ("FLOAT", _ui("common_generation.cfg_scale", "CFG Scale", "Text classifier-free guidance scale.", default=4.0, min=0.1, max=30.0, step=0.1)),
    "denoise_timestep_shift": (
        "FLOAT",
        _ui("common_generation.denoise_timestep_shift", "Timestep Shift", "Denoising timestep shift, matching the Lance demo default.", default=3.0, min=0.1, max=20.0, step=0.1),
    ),
    "cfg_start": ("FLOAT", _ui("common_generation.cfg_start", "CFG Start", "Start of the CFG active interval.", default=0.4, min=0.0, max=1.0, step=0.01)),
    "cfg_end": ("FLOAT", _ui("common_generation.cfg_end", "CFG End", "End of the CFG active interval.", default=1.0, min=0.0, max=1.0, step=0.01)),
    "vae_decode_mode": (
        ["auto", "normal", "tiled"],
        _ui("common_generation.vae_decode_mode", "VAE Decode Mode", "auto enables spatial tiling on ROCm/HIP; normal uses standard decoding; tiled forces spatial tiling.", default="auto"),
    ),
    "vae_tile_size": (
        "INT",
        _ui("common_generation.vae_tile_size", "VAE Tile Size", "Output pixel size of each VAE spatial tile; only active in tiled/auto tiling.", default=384, min=128, max=2048, step=16),
    ),
    "vae_tile_overlap": (
        "INT",
        _ui("common_generation.vae_tile_overlap", "VAE Tile Overlap", "Output pixel overlap for stitching VAE tiles; larger overlap reduces seams but is slower.", default=64, min=0, max=512, step=16),
    ),
}


class LanceLoadModel:
    CATEGORY = CATEGORY
    DESCRIPTION = _tr_text(
        "descriptions.LanceLoadModel",
        "Load bytedance-research/Lance from ComfyUI/models/Lance with device selection, FlashAttention/SageAttention options, and int8/int4/fp4/fp8 quantization wrappers.",
    )
    RETURN_TYPES = ("LANCE_MODEL", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceLoadModel", ("lance_model", "Status"))
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": (
                    "STRING",
                    _ui(
                        "load_model.model_root",
                        "Model Root",
                        "Use auto for ComfyUI/models/Lance, or enter a custom absolute path.",
                        default="auto",
                    ),
                ),
                "model_scope": (
                    ["auto/lazy", "image", "video", "image+video"],
                    _ui("load_model.model_scope", "Load Scope", "auto/lazy loads the image/video model on demand when the first task runs.", default="auto/lazy"),
                ),
                "device": (_device_choices(), _ui("load_model.device", "Device", "Select the CUDA device used for Lance inference.", default="auto")),
                "compute_dtype": (
                    ["bf16", "fp32"],
                    _ui(
                        "load_model.compute_dtype",
                        "LLM Compute dtype",
                        "Compute dtype for the Lance language model, covering text embeddings, attention/MLP, and hidden states. FP16 is disabled; bf16 or fp32 is recommended.",
                        default="bf16",
                    ),
                ),
                "diffusion_compute_dtype": (
                    ["same as llm", "bf16", "fp16", "fp32"],
                    _ui(
                        "load_model.diffusion_compute_dtype",
                        "Diffusion Compute dtype",
                        "Compute dtype for the Lance diffusion/latent path, covering x_t, time_embedder, vae2llm, and llm2vae. Defaults to the LLM dtype.",
                        default="same as llm",
                    ),
                ),
                "vae_compute_dtype": (
                    ["same as diffusion", "same as llm", "bf16", "fp16", "fp32"],
                    _ui(
                        "load_model.vae_compute_dtype",
                        "VAE Compute dtype",
                        "Compute dtype for Wan VAE encoding/decoding. Defaults to the Diffusion Compute dtype.",
                        default="same as diffusion",
                    ),
                ),
                "attention_backend": (
                    ["auto", "flash_attention_2", "sage_attention", "sdpa"],
                    _ui("load_model.attention_backend", "Attention Backend", "auto/flash_attention_2 uses FlashAttention when available; sage_attention requires the sageattention package.", default="auto"),
                ),
                "quantization": (
                    ["none", "int8", "int4", "fp4", "fp8_e4m3fn", "fp8_e5m2"],
                    _ui("load_model.quantization", "Quantization", "Experimental Linear quantization. Use none for original precision first if generation quality is abnormal.", default="none"),
                ),
                "use_quantization_cache": (
                    "BOOLEAN",
                    _ui(
                        "load_model.use_quantization_cache",
                        "Quantization Cache",
                        "When enabled, reusable quantized weights are saved to ComfyUI/models/Lance-quantized-cache.",
                        default=True,
                    ),
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    _ui("load_model.rebuild_quantization_cache", "Rebuild Quantization Cache", "Ignore the existing quantization cache and regenerate it from the original weights.", default=False),
                ),
                "use_kv_cache": (
                    "BOOLEAN",
                    _ui("load_model.use_kv_cache", "Enable KV Cache", "Enable Lance's official KV cache path for visual generation to reduce repeated attention computation.", default=True),
                ),
            },
            "optional": {
                "download_missing": (
                    "BOOLEAN",
                    _ui("load_model.download_missing", "Download Missing Files", "Download missing model files from the Hugging Face repository to the model root.", default=False),
                ),
                "download_source": (
                    ["huggingface.co", "hf-mirror.com"],
                    _ui("load_model.download_source", "Download Source", "Source used for automatic downloads.", default="huggingface.co"),
                ),
                "revision": (
                    "STRING",
                    _ui("load_model.revision", "Revision", "Hugging Face revision. Leave empty to use the default branch.", default=""),
                ),
            },
        }

    def load(
        self,
        model_root: str,
        model_scope: str,
        device: str,
        compute_dtype: str,
        diffusion_compute_dtype: str = "same as llm",
        vae_compute_dtype: str = "same as diffusion",
        attention_backend: str = "auto",
        quantization: str = "none",
        use_quantization_cache: bool = True,
        rebuild_quantization_cache: bool = False,
        use_kv_cache: bool = True,
        download_missing: bool = False,
        download_source: str = "huggingface.co",
        revision: str = "",
    ):
        if (
            (vae_compute_dtype or "").strip().lower() in {"auto", "flash_attention_2", "sage_attention", "sdpa"}
            and (attention_backend or "").strip().lower() in {"none", "int8", "int4", "fp4", "fp8_e4m3fn", "fp8_e5m2"}
        ):
            quantization = attention_backend
            attention_backend = vae_compute_dtype
            vae_compute_dtype = diffusion_compute_dtype
            diffusion_compute_dtype = "same as llm"

        if (
            (diffusion_compute_dtype or "").strip().lower() in {"auto", "flash_attention_2", "sage_attention", "sdpa"}
            and (vae_compute_dtype or "").strip().lower() in {"none", "int8", "int4", "fp4", "fp8_e4m3fn", "fp8_e5m2"}
        ):
            quantization = vae_compute_dtype
            attention_backend = diffusion_compute_dtype
            diffusion_compute_dtype = "same as llm"
            vae_compute_dtype = "same as diffusion"

        model_root_path = _resolve_model_root(model_root)
        paths = _lance_paths(model_root_path)
        device_obj, device_index = _resolve_device(device)
        llm_dtype = _llm_dtype_from_name(compute_dtype)
        dtype = _diffusion_dtype_from_name(diffusion_compute_dtype, llm_dtype)
        vae_dtype = _vae_dtype_from_name(vae_compute_dtype, llm_dtype, dtype)
        quantization_mode = _normalize_quantization_mode(quantization)
        quantization_cache_dir = _lance_quantization_cache_dir() if use_quantization_cache and quantization_mode else None
        handle = LanceModelHandle(
            model_root=model_root_path,
            paths=paths,
            device=device_obj,
            device_index=device_index,
            llm_dtype=llm_dtype,
            dtype=dtype,
            vae_dtype=vae_dtype,
            attention_backend=attention_backend,
            quantization=quantization,
            quantization_cache_dir=quantization_cache_dir,
            rebuild_quantization_cache=rebuild_quantization_cache,
            use_kv_cache=use_kv_cache,
            download_missing=download_missing,
            download_source=download_source,
            revision=revision or "",
        )
        current_runtime_keys = {handle._runtime_cache_key("image"), handle._runtime_cache_key("video")}
        previous_runtime_keys = set(getattr(self, "_last_runtime_keys", set()))
        evicted = _evict_runtime_cache(
            lambda key: key in previous_runtime_keys and key not in current_runtime_keys,
            clear_cuda_cache=True,
        )
        evicted += handle.evict_stale_runtimes()
        self._last_runtime_keys = current_runtime_keys
        status = handle.preload(model_scope)
        if evicted:
            status = f"{status}\n已释放 {evicted} 个旧 Lance 运行时。"
        cache_status = f"\n量化缓存目录: {quantization_cache_dir}" if quantization_cache_dir is not None else ""
        kv_cache_status = "开启" if use_kv_cache else "关闭"
        return (
            handle,
            f"{status}\n模型根目录: {model_root_path}\n设备: {device_obj}\n"
            f"LLM Compute dtype: {str(llm_dtype).replace('torch.', '')}\n"
            f"Diffusion Compute dtype: {str(dtype).replace('torch.', '')}\n"
            f"VAE Compute dtype: {str(vae_dtype).replace('torch.', '')}\n"
            f"KV Cache: {kv_cache_status}{cache_status}",
        )


class LanceImageUnderstanding:
    CATEGORY = CATEGORY
    DESCRIPTION = _tr_text("descriptions.LanceImageUnderstanding", "Use Lance for image understanding with ComfyUI IMAGE input and STRING output.")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = _tr_names("return_names.LanceImageUnderstanding", ("Answer",))
    FUNCTION = "understand"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "image": ("IMAGE", _ui("image_understanding.image", "Image", "ComfyUI IMAGE input.")),
                "question": (
                    "STRING",
                    _ui(
                        "image_understanding.question",
                        "Question",
                        "Image understanding question. Chinese and English are both supported.",
                        default=_tr_text("defaults.image_question", "Describe this image in detail."),
                        multiline=True,
                    ),
                ),
                "max_new_tokens": (
                    "INT",
                    _ui("shared.max_new_tokens", "Max New Tokens", "Maximum number of tokens generated for understanding tasks.", default=256, min=1, max=2048, step=1),
                ),
                "batch_index": (
                    "INT",
                    _ui("shared.batch_index", "Batch Index", "Select which image to use from a batched IMAGE input.", default=0, min=0, max=4096, step=1),
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
    DESCRIPTION = _tr_text("descriptions.LanceImageGeneration", "Use Lance text-to-image generation and output a ComfyUI IMAGE.")
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceImageGeneration", ("Image", "Path"))
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "prompt": ("STRING", _ui("image_generation.prompt", "Prompt", "Text-to-image prompt.", default="", multiline=True)),
                "width": ("INT", _ui("image_generation.width", "Width", "Target image width.", default=768, min=128, max=2048, step=8)),
                "height": ("INT", _ui("image_generation.height", "Height", "Target image height.", default=768, min=128, max=2048, step=8)),
                **COMMON_GENERATION_INPUTS,
                "resolution": (
                    ["auto", "image_768res", "image_512res", "image_256res"],
                    _ui("image_generation.resolution", "Resolution Preset", "Lance data preprocessing resolution preset. auto uses the demo default.", default="auto"),
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
    DESCRIPTION = _tr_text("descriptions.LanceImageEditing", "Use Lance image editing with ComfyUI IMAGE input and output.")
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceImageEditing", ("Image", "Path"))
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "image": ("IMAGE", _ui("image_editing.image", "Input Image", "ComfyUI IMAGE to edit.")),
                "prompt": ("STRING", _ui("image_editing.prompt", "Edit Instruction", "Image editing instruction.", default="", multiline=True)),
                "width": ("INT", _ui("image_generation.width", "Width", "Target image width.", default=768, min=128, max=2048, step=8)),
                "height": ("INT", _ui("image_generation.height", "Height", "Target image height.", default=768, min=128, max=2048, step=8)),
                **COMMON_GENERATION_INPUTS,
                "resolution": (
                    ["auto", "image_768res", "image_512res", "image_256res"],
                    _ui("image_generation.resolution", "Resolution Preset", "Lance data preprocessing resolution preset. auto uses the demo default.", default="auto"),
                ),
                "batch_index": (
                    "INT",
                    _ui("shared.batch_index", "Batch Index", "Select which image to use from a batched IMAGE input.", default=0, min=0, max=4096, step=1),
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
    DESCRIPTION = _tr_text("descriptions.LanceVideoUnderstanding", "Use Lance for video understanding with ComfyUI VIDEO input and STRING output.")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = _tr_names("return_names.LanceVideoUnderstanding", ("Answer",))
    FUNCTION = "understand"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "video": ("VIDEO", _ui("video_understanding.video", "Video", "ComfyUI VIDEO input.")),
                "question": (
                    "STRING",
                    _ui(
                        "video_understanding.question",
                        "Question",
                        "Video understanding question. Chinese and English are both supported.",
                        default=_tr_text("defaults.video_question", "Describe this video in detail."),
                        multiline=True,
                    ),
                ),
                "max_new_tokens": (
                    "INT",
                    _ui("shared.max_new_tokens", "Max New Tokens", "Maximum number of tokens generated for understanding tasks.", default=256, min=1, max=2048, step=1),
                ),
                "max_duration": (
                    "FLOAT",
                    _ui("shared.max_duration", "Max Duration", "Maximum duration allowed for video sampling, in seconds.", default=6.0, min=0.1, max=120.0, step=0.1),
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
    "width": ("INT", _ui("video_generation.width", "Width", "Target video width.", default=848, min=128, max=2048, step=8)),
    "height": ("INT", _ui("video_generation.height", "Height", "Target video height.", default=480, min=128, max=2048, step=8)),
    "num_frames": ("INT", _ui("video_generation.num_frames", "Frames", "Target number of video frames.", default=50, min=2, max=257, step=1)),
    **COMMON_GENERATION_INPUTS,
    "resolution": (
        ["auto", "video_480p", "video_360p", "video_192p"],
        _ui("video_generation.resolution", "Resolution Preset", "Lance data preprocessing resolution preset. auto uses the demo default.", default="auto"),
    ),
    "max_duration": (
        "FLOAT",
        _ui("shared.max_duration", "Max Duration", "Maximum duration allowed for video sampling, in seconds.", default=6.0, min=0.1, max=120.0, step=0.1),
    ),
}


class LanceTextToVideo:
    CATEGORY = CATEGORY
    DESCRIPTION = _tr_text("descriptions.LanceTextToVideo", "Use Lance text-to-video generation and output ComfyUI VIDEO plus IMAGE frames.")
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceTextToVideo", ("Video", "Frames", "Path"))
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "prompt": ("STRING", _ui("text_to_video.prompt", "Prompt", "Text-to-video prompt.", default="", multiline=True)),
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
    DESCRIPTION = _tr_text("descriptions.LanceImageToVideo", "Use Lance image-to-video generation with IMAGE input and ComfyUI VIDEO output.")
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceImageToVideo", ("Video", "Frames", "Path"))
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "image": ("IMAGE", _ui("image_to_video.image", "Reference Image", "Input image for image-to-video generation.")),
                "prompt": ("STRING", _ui("image_to_video.prompt", "Prompt", "Image-to-video prompt.", default="", multiline=True)),
                **VIDEO_GENERATION_INPUTS,
                "batch_index": (
                    "INT",
                    _ui("shared.batch_index", "Batch Index", "Select which image to use from a batched IMAGE input.", default=0, min=0, max=4096, step=1),
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
    DESCRIPTION = _tr_text("descriptions.LanceVideoEditing", "Use Lance video editing with ComfyUI VIDEO input and output.")
    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = _tr_names("return_names.LanceVideoEditing", ("Video", "Frames", "Path"))
    FUNCTION = "edit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("shared.lance_model", "Lance Model", "Output from the Lance Load Model node.")),
                "video": ("VIDEO", _ui("video_editing.video", "Input Video", "ComfyUI VIDEO to edit.")),
                "prompt": ("STRING", _ui("video_editing.prompt", "Edit Instruction", "Video editing instruction.", default="", multiline=True)),
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
    DESCRIPTION = _tr_text("descriptions.LanceFramesToVideo", "Pack ComfyUI IMAGE frames into a ComfyUI VIDEO for connecting with Lance video nodes.")
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = _tr_names("return_names.LanceFramesToVideo", ("Video",))
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", _ui("frames_to_video.images", "Frames", "ComfyUI IMAGE frame sequence.")),
                "fps": ("FLOAT", _ui("frames_to_video.fps", "FPS", "Frame rate of the output VIDEO.", default=12.0, min=0.1, max=240.0, step=0.1)),
            }
        }

    def convert(self, images: torch.Tensor, fps: float):
        return (_images_to_video(images, fps),)


class LanceUnloadModel:
    CATEGORY = CATEGORY
    DESCRIPTION = _tr_text("descriptions.LanceUnloadModel", "Release the Lance model cache and optionally clear CUDA VRAM.")
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = _tr_names("return_names.LanceUnloadModel", ("Status",))
    FUNCTION = "release"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance_model": ("LANCE_MODEL", _ui("unload_model.lance_model", "Lance Model", "Lance model handle to release.")),
                "clear_cuda_cache": ("BOOLEAN", _ui("unload_model.clear_cuda_cache", "Clear CUDA Cache", "Clear CUDA cache after releasing the model.", default=True)),
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

NODE_DISPLAY_NAME_MAPPINGS = _tr_mapping(
    "node_display_names",
    {
        "LanceLoadModel": "Lance Load Model",
        "LanceImageUnderstanding": "Lance Image Understanding",
        "LanceImageGeneration": "Lance Image Generation",
        "LanceImageEditing": "Lance Image Editing",
        "LanceVideoUnderstanding": "Lance Video Understanding",
        "LanceImageToVideo": "Lance Image to Video",
        "LanceTextToVideo": "Lance Text to Video",
        "LanceVideoEditing": "Lance Video Editing",
        "LanceFramesToVideo": "Lance Frames to Video",
        "LanceUnloadModel": "Lance Unload Model",
    },
)


__all__ = [name for name in globals() if not name.startswith("__")]
