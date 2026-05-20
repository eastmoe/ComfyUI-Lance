# Lance Lightweight Inference

This directory keeps only the core Lance inference path: image understanding, image generation, image editing, video understanding, text-to-video generation, image-to-video generation, and video editing.

Removed components include sample assets, Gradio GUI demos, benchmark scripts, training/evaluation/fine-tuning entrypoints, and their task-specific dependencies.

## Install

Recommended environment: Python 3.10+, CUDA 12.4+, and a GPU with at least 40GB VRAM for inference.

```bash
bash setup_env.sh
```

Place checkpoints under `downloads/`:

```text
downloads/
  Lance_3B/
  Lance_3B_Video/
  Qwen2.5-VL-ViT/
  Wan2.2_VAE.pth
```

You can change defaults in `config/path_default.yaml`, or pass `--model_path` and `--vit_path` explicitly.

## Inference

All tasks run through `inference_lance.py` with explicit parameters. No interactive CLI or GUI is required.

### Text to Image

```bash
python inference_lance.py \
  --task t2i \
  --prompt "A cinematic portrait of a cat astronaut, highly detailed" \
  --output results/t2i
```

### Text to Video

```bash
python inference_lance.py \
  --task t2v \
  --prompt "A calm sunrise over a mountain lake, slow camera push-in" \
  --num_frames 50 \
  --video_height 480 \
  --video_width 848 \
  --output results/t2v
```

### Image to Video

```bash
python inference_lance.py \
  --task i2v \
  --image path/to/input.jpg \
  --prompt "Animate the subject with subtle natural motion and a gentle camera move" \
  --num_frames 50 \
  --video_height 480 \
  --video_width 848 \
  --output results/i2v
```

### Image Editing

```bash
python inference_lance.py \
  --task image_edit \
  --image path/to/input.jpg \
  --prompt "Remove the hat and keep the original painting style" \
  --output results/image_edit
```

### Video Editing

```bash
python inference_lance.py \
  --task video_edit \
  --video path/to/input.mp4 \
  --prompt "Change the car color to red and keep the motion consistent" \
  --output results/video_edit
```

### Image Understanding

```bash
python inference_lance.py \
  --task x2t_image \
  --image path/to/input.jpg \
  --question "What objects are visible in the image?" \
  --output results/x2t_image
```

### Video Understanding

```bash
python inference_lance.py \
  --task x2t_video \
  --video path/to/input.mp4 \
  --question "Describe the main action in this video." \
  --output results/x2t_video
```

Generated media and `prompt.json` are written to the output directory. Understanding tasks also write `result.json`.

## Useful Arguments

| Argument | Description |
| --- | --- |
| `--task` | `t2i`, `t2v`, `i2v`, `image_edit`, `video_edit`, `x2t_image`, or `x2t_video` |
| `--prompt` | Generation or editing instruction |
| `--image` | Input image for image understanding, image editing, or image-to-video |
| `--video` | Input video for video understanding or video editing |
| `--question` | Visual understanding question; defaults to a generic description request |
| `--output` / `--save_path_gen` | Output directory |
| `--model_path` | Lance checkpoint directory; defaults are read from `config/path_default.yaml` |
| `--vit_path` | Qwen2.5-VL ViT checkpoint directory |
| `--num_timesteps` | Denoising steps, default 30 |
| `--denoise_timestep_shift` | Denoising timestep shift, default 3.0 |
| `--cfg_scale` / `--cfg_text_scale` | Text CFG strength |
| `--seed` | Sampling seed |
| `--num_frames` | Number of video frames |
| `--video_height` / `--video_width` | Target video size |

## Batch Input

For custom batches, pass `--input_json path/to/request.json`. Direct task parameters are ignored when `--input_json` is supplied. The format matches the `request.json` automatically generated in each output directory.
