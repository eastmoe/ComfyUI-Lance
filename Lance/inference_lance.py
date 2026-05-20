# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

import json
import os
import os.path as osp
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, cast

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import HfArgumentParser, set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.inference_utils import decode_video_tensor, make_padded_latent
from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments, get_model_path
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.inference_dataset import InferenceDataset
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


MAX_GENERATION_LENGTH = 256
PROMPT_JSON_FILENAME = "prompt.json"
RESULT_JSON_FILENAME = "result.json"

TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_I2V = "i2v"
TASK_X2T_IMAGE = "x2t_image"
TASK_X2T_VIDEO = "x2t_video"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
PIL_RESAMPLING = getattr(Image, "Resampling", Image)

GENERATION_TASKS = {
    TASK_T2V,
    TASK_T2I,
    TASK_I2V,
    TASK_IMAGE_EDIT,
    TASK_VIDEO_EDIT,
}
UNDERSTANDING_TASKS = {
    TASK_X2T_IMAGE,
    TASK_X2T_VIDEO,
}

TASK_ALIASES = {
    "text_to_video": TASK_T2V,
    "txt2video": TASK_T2V,
    "text2video": TASK_T2V,
    "text_to_image": TASK_T2I,
    "txt2image": TASK_T2I,
    "text2image": TASK_T2I,
    "image_to_video": TASK_I2V,
    "img2video": TASK_I2V,
    "image2video": TASK_I2V,
    "image_understanding": TASK_X2T_IMAGE,
    "x2t": TASK_X2T_IMAGE,
    "i2t": TASK_X2T_IMAGE,
    "video_understanding": TASK_X2T_VIDEO,
    "v2t": TASK_X2T_VIDEO,
    "edit_image": TASK_IMAGE_EDIT,
    "edit_video": TASK_VIDEO_EDIT,
}

TASK_DEFAULTS = {
    TASK_T2I: {
        "model_family": "image",
        "save_path": "results/t2i",
        "resolution": "image_768res",
        "height": 768,
        "width": 768,
    },
    TASK_T2V: {
        "model_family": "video",
        "save_path": "results/t2v",
        "resolution": "video_480p",
        "height": 480,
        "width": 848,
    },
    TASK_I2V: {
        "model_family": "video",
        "dataset_task": "video_idip",
        "save_path": "results/i2v",
        "resolution": "video_480p",
        "height": 480,
        "width": 848,
    },
    TASK_IMAGE_EDIT: {
        "model_family": "image",
        "save_path": "results/image_edit",
        "resolution": "image_768res",
        "height": 768,
        "width": 768,
    },
    TASK_VIDEO_EDIT: {
        "model_family": "video",
        "save_path": "results/video_edit",
        "resolution": "video_480p",
        "height": 480,
        "width": 848,
    },
    TASK_X2T_IMAGE: {
        "model_family": "image",
        "save_path": "results/x2t_image",
        "resolution": "image_768res",
        "height": 768,
        "width": 768,
    },
    TASK_X2T_VIDEO: {
        "model_family": "video",
        "save_path": "results/x2t_video",
        "resolution": "video_480p",
        "height": 480,
        "width": 848,
    },
}


def normalize_task(task: str) -> str:
    normalized = (task or TASK_T2V).strip().lower().replace("-", "_")
    normalized = TASK_ALIASES.get(normalized, normalized)
    if normalized not in TASK_DEFAULTS:
        supported = ", ".join(sorted(TASK_DEFAULTS))
        raise ValueError(f"Unsupported task '{task}'. Supported tasks: {supported}")
    return normalized


def get_dataset_task(task: str) -> str:
    return str(TASK_DEFAULTS[task].get("dataset_task", task))


def clean_memory(*objects):
    """Clear temporary containers and release unused GPU cache."""
    for obj in objects:
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, (list, set)):
            obj.clear()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_from_model_path_if_needed(model: Lance, model_args: ModelArguments):
    path_dir = model_args.model_path
    ema_path = osp.join(path_dir, "ema.safetensors")
    model_path = osp.join(path_dir, "model.safetensors")

    checkpoint_path = None
    if osp.exists(model_path):
        checkpoint_path = model_path
    elif osp.exists(ema_path):
        checkpoint_path = ema_path

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Checkpoint load failed: no 'ema.safetensors' or 'model.safetensors' found in {path_dir}"
        )

    model_state_dict = load_file(checkpoint_path, device="cpu")

    if "latent_pos_embed.pos_embed" in model_state_dict:
        model_state_dict.pop("latent_pos_embed.pos_embed")

    msg = model.load_state_dict(model_state_dict, strict=False)
    clean_memory(model_state_dict)
    return msg


def apply_inference_defaults(
    model_args: ModelArguments,
    data_args: DataArguments,
    inference_args: InferenceArguments,
) -> None:
    inference_args.task = normalize_task(inference_args.task)
    task_config = TASK_DEFAULTS[inference_args.task]
    default_args = InferenceArguments()

    if inference_args.output:
        inference_args.save_path_gen = inference_args.output
    elif inference_args.save_path_gen == default_args.save_path_gen:
        inference_args.save_path_gen = str(task_config["save_path"])

    if inference_args.cfg_scale > 0:
        model_args.cfg_text_scale = inference_args.cfg_scale

    model_family = str(task_config["model_family"])
    if not model_args.model_path:
        model_args.model_path = get_model_path(f"lance.{model_family}")
    if not model_args.llm_path:
        model_args.llm_path = model_args.model_path
    if not model_args.vit_path:
        model_args.vit_path = get_model_path("vit.qwen2_5_vl")

    if inference_args.resolution == default_args.resolution:
        inference_args.resolution = str(task_config["resolution"])
    if inference_args.video_height == default_args.video_height:
        inference_args.video_height = int(task_config["height"])
    if inference_args.video_width == default_args.video_width:
        inference_args.video_width = int(task_config["width"])

    if inference_args.task == TASK_T2I:
        inference_args.num_frames = 1
    if inference_args.task == TASK_I2V:
        inference_args.max_duration = max(float(inference_args.max_duration), inference_args.num_frames / 12.0)
    if inference_args.task in UNDERSTANDING_TASKS:
        inference_args.visual_gen = False
    if inference_args.task in {TASK_T2I, TASK_T2V}:
        inference_args.visual_und = False

    if data_args.input_json:
        data_args.input_json = str(Path(data_args.input_json))


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def _require_text(value: str, name: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"--{name} is required for this task")
    return value


def _create_i2v_target_video(image_path: str, output_dir: Path, inference_args: InferenceArguments) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "_i2v_target.mp4"

    with Image.open(image_path) as image:
        frame = image.convert("RGB").resize(
            (inference_args.video_width, inference_args.video_height),
            PIL_RESAMPLING.LANCZOS,
        )
        frame_np = np.asarray(frame)

    frame_count = max(2, int(inference_args.num_frames) * 2)
    imageio.mimsave(str(output_path), [frame_np] * frame_count, fps=24, format="mp4")
    return str(output_path)


def build_direct_input_json(inference_args: InferenceArguments) -> str:
    task = inference_args.task
    save_dir = Path(inference_args.save_path_gen)
    request_path = save_dir / "request.json"

    if task == TASK_T2I:
        prompt = _require_text(inference_args.prompt, "prompt")
        return _write_json(request_path, {"000000.png": prompt})

    if task == TASK_T2V:
        prompt = _require_text(inference_args.prompt, "prompt")
        return _write_json(request_path, {"000000.mp4": prompt})

    if task == TASK_IMAGE_EDIT:
        prompt = _require_text(inference_args.prompt, "prompt")
        image = _require_text(inference_args.image, "image")
        payload = {
            "000000": {
                "interleave_array": [prompt, image, image],
                "element_dtype_array": ["text", "image", "image"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
        return _write_json(request_path, payload)

    if task == TASK_VIDEO_EDIT:
        prompt = _require_text(inference_args.prompt, "prompt")
        video = _require_text(inference_args.video, "video")
        payload = {
            "000000": {
                "interleave_array": [prompt, video, video],
                "element_dtype_array": ["text", "video", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
        return _write_json(request_path, payload)

    if task == TASK_I2V:
        prompt = _require_text(inference_args.prompt, "prompt")
        image = _require_text(inference_args.image, "image")
        target_video = _create_i2v_target_video(image, save_dir / "_tmp", inference_args)
        payload = {
            "000000": {
                "interleave_array": [prompt, image, target_video],
                "element_dtype_array": ["text", "image", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
        return _write_json(request_path, payload)

    if task == TASK_X2T_IMAGE:
        image = _require_text(inference_args.image, "image")
        question = (inference_args.question or inference_args.prompt or "Describe this image.").strip()
        payload = {
            "000000": {
                "interleave_array": [
                    image,
                    ["Look at the image carefully and answer the question.", question, ""],
                ],
                "element_dtype_array": ["image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
        return _write_json(request_path, payload)

    if task == TASK_X2T_VIDEO:
        video = _require_text(inference_args.video, "video")
        question = (inference_args.question or inference_args.prompt or "Describe this video.").strip()
        payload = {
            "000000": {
                "interleave_array": [
                    video,
                    ["Watch the video carefully and answer the question.", question, ""],
                ],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
        return _write_json(request_path, payload)

    raise ValueError(f"Unsupported task: {task}")


def save_prompt_results(prompt_data_dict, save_path_gen, logger):
    prompt_json_path = os.path.join(save_path_gen, PROMPT_JSON_FILENAME)
    with open(prompt_json_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data_dict, f, ensure_ascii=False, indent=2)


def normalize_understanding_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.replace("<|im_end|>", "").strip()


def save_understanding_results(
    prompt_data_dict: dict,
    input_json: str,
    save_path_gen: str,
) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        dataset_samples = json.load(f)

    result_entries = []
    for sample_key, sample in dataset_samples.items():
        interleave_array = sample.get("interleave_array", [])
        element_dtype_array = sample.get("element_dtype_array", [])
        if len(interleave_array) < 2 or not element_dtype_array:
            continue

        visual_path = interleave_array[0]
        text_payload = interleave_array[1]
        question = text_payload[1] if isinstance(text_payload, list) and len(text_payload) > 1 else ""
        modality = element_dtype_array[0]

        lookup_keys = [os.path.basename(visual_path), sample_key]
        generated_answer = ""
        for lookup_key in lookup_keys:
            if lookup_key in prompt_data_dict:
                generated_answer = prompt_data_dict[lookup_key]
                break

        result_entries.append(
            {
                modality: visual_path,
                "question": question,
                "answer": normalize_understanding_answer(generated_answer),
            }
        )

    result_json_path = os.path.join(save_path_gen, RESULT_JSON_FILENAME)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_entries, f, ensure_ascii=False, indent=2)


def run_inference_batch(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    tokenizer: Qwen2Tokenizer,
    batch_cpu: dict,
    model_args: ModelArguments,
    inference_args: InferenceArguments,
    new_token_ids,
    image_token_id: int,
    device: int,
    save_source_video: bool = False,
    save_path_gen: str = "",
    save_path_gt: str = "",
):
    batch = batch_cpu.cuda(device).to_dict()
    runtime_dtype = getattr(inference_args, "runtime_dtype", torch.bfloat16)
    fsdp_model = fsdp_model.to(device=device, dtype=runtime_dtype)

    autocast_enabled = runtime_dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=runtime_dtype):
        if inference_args.task in GENERATION_TASKS and batch.get("padded_videos"):
            batch["padded_latent"] = make_padded_latent(batch["padded_videos"], batch["vae_data_mode"], vae_model)

        if inference_args.task in GENERATION_TASKS:
            params = {
                "val_packed_text_ids": batch["packed_text_ids"],
                "val_packed_text_indexes": batch["packed_text_indexes"],
                "val_sample_lens": batch["sample_lens"],
                "val_packed_position_ids": batch["packed_position_ids"],
                "val_split_lens": batch["split_lens"],
                "val_attn_modes": batch["attn_modes"],
                "val_sample_N_target": batch["sample_N_target"],
                "val_packed_vae_token_indexes": batch["packed_vae_token_indexes"],
                "timestep_shift": inference_args.denoise_timestep_shift,
                "num_timesteps": inference_args.num_timesteps,
                "val_mse_loss_indexes": batch.get("mse_loss_indexes", None),
                "val_padded_latent": batch["padded_latent"],
                "video_sizes": batch["video_sizes"],
                "cfg_text_scale": model_args.cfg_text_scale,
                "cfg_interval": inference_args.cfg_interval,
                "cfg_renorm_min": inference_args.cfg_renorm_min,
                "cfg_renorm_type": inference_args.cfg_renorm_type,
                "device": device,
                "dtype": runtime_dtype,
                "new_token_ids": new_token_ids,
                "max_samples": inference_args.max_samples,
                "noise_seed": inference_args.seed,
                "apply_chat_template": inference_args.apply_chat_template,
                "apply_qwen_2_5_vl_pos_emb": inference_args.apply_qwen_2_5_vl_pos_emb,
                "image_token_id": image_token_id,
                "val_packed_vit_token_indexes": batch.get("packed_vit_token_indexes", None),
                "val_packed_vit_tokens": batch.get("packed_vit_tokens", None),
                "vit_video_grid_thw": batch.get("vit_video_grid_thw", None),
                "vae_video_grid_thw": batch["vae_video_grid_thw"],
                "video_grid_thw": batch.get("video_grid_thw", None),
                "caption": batch.get("caption", None),
                "sample_task": batch["sample_task"],
                "sample_modality": batch["sample_modality"],
                "cfg_type": inference_args.cfg_type,
                "cfg_uncond_token_id": inference_args.cfg_uncond_token_id,
                "index": batch["index"],
                "val_padded_videos": batch["padded_videos"] if save_source_video else None,
            }
            if inference_args.use_KVcache:
                denoise_latent, captions, padded_videos, index = fsdp_model.generate_visual_kvcache(**params)
            else:
                denoise_latent, captions, padded_videos, index = fsdp_model.generate_visual(**params)

            for i_val, latent in enumerate(denoise_latent):
                if inference_args.task in {TASK_IMAGE_EDIT, TASK_VIDEO_EDIT, TASK_I2V}:
                    target_latents = [latent[-1]]
                else:
                    target_latents = latent

                v_list = [vae_model.vae_decode([latent_])[0] for latent_ in target_latents]
                save_item_name = f"{index:06d}" if isinstance(index, int) else index
                v_thwc = decode_video_tensor(v_list, save_path=save_path_gen, save_half=False, save_item_name=save_item_name)
                prompt_data_path = f"{save_item_name}.mp4" if v_thwc.shape[0] > 1 else f"{save_item_name}.png"
                inference_args.prompt_data_dict[prompt_data_path] = captions[i_val]

                if save_source_video:
                    curr_padded_videos = padded_videos[i_val * 2 : (i_val + 1) * 2]
                    decode_video_tensor(curr_padded_videos[-1:], save_path=save_path_gt, save_item_name=save_item_name)

                del v_list, v_thwc, latent, target_latents
                clean_memory()

            del denoise_latent, captions, padded_videos, params
            clean_memory()

        elif inference_args.task in UNDERSTANDING_TASKS:
            generated_sequence_all, captions, index = fsdp_model.understand_visual(
                val_packed_text_ids=batch["packed_text_ids"],
                val_packed_text_indexes=batch["packed_text_indexes"],
                val_packed_position_ids=batch["packed_position_ids"],
                val_sample_N_target=batch["sample_N_target"],
                val_split_lens=batch["split_lens"],
                val_attn_modes=batch["attn_modes"],
                val_sample_lens=batch["sample_lens"],
                val_sample_type=batch["sample_type"],
                val_packed_vit_tokens=batch["packed_vit_tokens"],
                val_vit_video_grid_thw=batch["vit_video_grid_thw"],
                val_ce_loss_indexes=batch["ce_loss_indexes"],
                max_samples=inference_args.max_samples,
                max_length=MAX_GENERATION_LENGTH,
                device=device,
                dtype=runtime_dtype,
                new_token_ids=new_token_ids,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=len(tokenizer),
                caption=batch.get("caption_cn", None),
                tokenizer=tokenizer,
                apply_chat_template=inference_args.apply_chat_template,
                apply_qwen_2_5_vl_pos_emb=inference_args.apply_qwen_2_5_vl_pos_emb,
                do_sample=False,
                image_token_id=image_token_id,
                index=batch["index"],
            )

            for generated_sequence in generated_sequence_all:
                cap = tokenizer.decode(generated_sequence[:, 0])
                inference_args.prompt_data_dict[index] = f"{cap}"
                del generated_sequence

            del generated_sequence_all, captions
            clean_memory()

    del batch
    clean_memory()


def build_dataset_config(
    input_json: str,
    model_args: ModelArguments,
    inference_args: InferenceArguments,
    vae_config: Optional[AutoEncoderParams],
) -> DataConfig:
    dataset_config = DataConfig.from_yaml(input_json)

    if inference_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
        dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

    if inference_args.visual_gen:
        assert len(model_args.latent_patch_size) == 3, "len(latent_patch_size) must be 3"
        vae_downsample = tuple_mul(
            model_args.latent_patch_size,
            (
                vae_config.downsample_temporal,
                vae_config.downsample_spatial,
                vae_config.downsample_spatial,
            ),
        )
        dataset_config.latent_patch_size = model_args.latent_patch_size
        dataset_config.vae_downsample = vae_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.max_num_frames = model_args.max_num_frames

    dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    dataset_config.num_frames = inference_args.num_frames
    dataset_config.H = inference_args.video_height
    dataset_config.W = inference_args.video_width
    dataset_config.task = get_dataset_task(inference_args.task)
    dataset_config.resolution = inference_args.resolution
    dataset_config.text_template = inference_args.text_template
    dataset_config.max_duration = inference_args.max_duration
    dataset_config.system_prompt_type = inference_args.system_prompt_type
    return dataset_config


def main():
    assert torch.cuda.is_available(), "CUDA is required for Lance inference."
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    local_rank = global_rank % torch.cuda.device_count()
    device = local_rank
    torch.cuda.set_device(device)

    parser = HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, inference_args = cast(
        Tuple[ModelArguments, DataArguments, InferenceArguments],
        parser.parse_args_into_dataclasses(),
    )

    apply_inference_defaults(model_args, data_args, inference_args)
    if data_args.input_json is None:
        data_args.input_json = build_direct_input_json(inference_args)

    logger = get_logger()
    log_rank0 = print if global_rank == 0 else (lambda *_: None)

    def log_stage(stage_name: str, start_time: float, extra: str = ""):
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        log_rank0(f"[startup] {stage_name} done in {elapsed:.2f}s{suffix}")

    seed = inference_args.global_seed * world_size + global_rank
    set_seed(seed)

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading LLM config: {osp.join(model_args.model_path, 'llm_config.json')}")
    llm_config: Qwen2Config = Qwen2Config.from_json_file(osp.join(model_args.model_path, "llm_config.json"))
    log_stage("LLM config load", stage_start)

    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.qk_norm_und = model_args.llm_qk_norm_und
    llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = False
    llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Initializing LLM weights: {model_args.model_path}")
    language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
    log_stage("LLM weight init", stage_start)

    vit_model = None
    vit_config = None
    if inference_args.visual_und:
        if model_args.vit_type in ("qwen2_5_vl", "qwen_2_5_vl_original"):
            stage_start = time.perf_counter()
            log_rank0(f"[startup] Loading VIT config: {model_args.vit_path}")
            vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
            log_stage("VIT config load", stage_start)

            stage_start = time.perf_counter()
            log_rank0(f"[startup] Loading VIT weights: {osp.join(model_args.vit_path, 'vit.safetensors')}")
            vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
            vit_weights = load_file(osp.join(model_args.vit_path, "vit.safetensors"))
            vit_model.load_state_dict(vit_weights, strict=True)
            log_stage("VIT weight load", stage_start)
            clean_memory(vit_weights)
        else:
            raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")

    if inference_args.visual_gen:
        stage_start = time.perf_counter()
        log_rank0("[startup] Initializing VAE")
        vae_model = WanVideoVAE()
        vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
        log_stage("VAE init", stage_start)
    else:
        vae_model = None
        vae_config = None

    config = LanceConfig(
        visual_gen=inference_args.visual_gen,
        visual_und=inference_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if inference_args.visual_und else None,
        vae_config=vae_config if inference_args.visual_gen else None,
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
        vit_model=vit_model if inference_args.visual_und else None,
        vit_type=model_args.vit_type,
        config=config,
        inference_args=inference_args,
    )

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Moving Lance model to GPU {device}")
    model = model.to(device)
    log_stage("Lance model move to GPU", stage_start)

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading tokenizer: {model_args.model_path}")
    tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

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

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    if vae_model is not None and hasattr(vae_model, "eval"):
        vae_model.eval()

    stage_start = time.perf_counter()
    log_rank0(f"[startup] Loading inference input: {data_args.input_json}")
    dataset_config = build_dataset_config(data_args.input_json, model_args, inference_args, vae_config)
    inference_dataset = InferenceDataset(
        jsonl_path=data_args.input_json,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        inference_args=inference_args,
        new_token_ids=new_token_ids,
        dataset_config=dataset_config,
        local_rank=global_rank,
        world_size=world_size,
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
    log_stage("input dataset and DataLoader init", stage_start, extra=f"dataset_size={len(inference_dataset)}")

    if not hasattr(inference_args, "prompt_data_dict"):
        inference_args.prompt_data_dict = {}

    os.makedirs(inference_args.save_path_gen, exist_ok=True)

    loader_iter = iter(loader)
    for _ in trange(len(loader), desc="Running", unit="batch", leave=True, ncols=80, disable=(global_rank != 0)):
        try:
            batch_cpu = next(loader_iter)
        except StopIteration:
            break

        run_inference_batch(
            fsdp_model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            batch_cpu=batch_cpu,
            model_args=model_args,
            inference_args=inference_args,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=device,
            save_source_video=False,
            save_path_gen=inference_args.save_path_gen,
            save_path_gt="",
        )
        del batch_cpu
        clean_memory()

    if dist.is_initialized():
        dist.barrier()
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, inference_args.prompt_data_dict)

        if global_rank == 0:
            merged = {}
            for item in gathered:
                merged.update(item)
            inference_args.prompt_data_dict = merged
            save_prompt_results(inference_args.prompt_data_dict, inference_args.save_path_gen, logger)
            if inference_args.task in UNDERSTANDING_TASKS:
                save_understanding_results(inference_args.prompt_data_dict, data_args.input_json, inference_args.save_path_gen)

    elif global_rank == 0:
        save_prompt_results(inference_args.prompt_data_dict, inference_args.save_path_gen, logger)
        if inference_args.task in UNDERSTANDING_TASKS:
            save_understanding_results(inference_args.prompt_data_dict, data_args.input_json, inference_args.save_path_gen)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
