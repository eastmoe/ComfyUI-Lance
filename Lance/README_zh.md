# Lance 轻量推理版

本目录已精简为仅保留 Lance 核心推理能力：图像理解、图像生成、图像编辑、视频理解、文生视频、图生视频和视频编辑。

已移除内容包括：示例素材、Gradio GUI demo、benchmark 评测脚本、训练/验证/微调入口及其专用依赖。

## 安装

推荐环境：Python 3.10+、CUDA 12.4+、单卡推理显存建议 40GB 以上。

```bash
bash setup_env.sh
```

将模型权重放到 `downloads/`：

```text
downloads/
  Lance_3B/
  Lance_3B_Video/
  Qwen2.5-VL-ViT/
  Wan2.2_VAE.pth
```

路径可在 `config/path_default.yaml` 中修改，也可以通过 `--model_path`、`--vit_path` 传入。

## 统一推理入口

所有任务都通过 `inference_lance.py` 参数运行，不需要交互式 CLI 或 GUI。

### 文生图

```bash
python inference_lance.py \
  --task t2i \
  --prompt "A cinematic portrait of a cat astronaut, highly detailed" \
  --output results/t2i
```

### 文生视频

```bash
python inference_lance.py \
  --task t2v \
  --prompt "A calm sunrise over a mountain lake, slow camera push-in" \
  --num_frames 50 \
  --video_height 480 \
  --video_width 848 \
  --output results/t2v
```

### 图生视频

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

### 图像编辑

```bash
python inference_lance.py \
  --task image_edit \
  --image path/to/input.jpg \
  --prompt "Remove the hat and keep the original painting style" \
  --output results/image_edit
```

### 视频编辑

```bash
python inference_lance.py \
  --task video_edit \
  --video path/to/input.mp4 \
  --prompt "Change the car color to red and keep the motion consistent" \
  --output results/video_edit
```

### 图像理解

```bash
python inference_lance.py \
  --task x2t_image \
  --image path/to/input.jpg \
  --question "What objects are visible in the image?" \
  --output results/x2t_image
```

### 视频理解

```bash
python inference_lance.py \
  --task x2t_video \
  --video path/to/input.mp4 \
  --question "Describe the main action in this video." \
  --output results/x2t_video
```

输出目录中会保存生成的图片/视频、`prompt.json`；理解任务还会保存结构化 `result.json`。

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--task` | `t2i`、`t2v`、`i2v`、`image_edit`、`video_edit`、`x2t_image`、`x2t_video` |
| `--prompt` | 生成或编辑指令 |
| `--image` | 图像理解、图像编辑、图生视频输入图 |
| `--video` | 视频理解、视频编辑输入视频 |
| `--question` | 图像/视频理解问题；不传时使用描述类默认问题 |
| `--output` / `--save_path_gen` | 输出目录 |
| `--model_path` | Lance 权重目录；不传则按任务从 `config/path_default.yaml` 读取 |
| `--vit_path` | Qwen2.5-VL ViT 权重目录 |
| `--num_timesteps` | 去噪步数，默认 30 |
| `--denoise_timestep_shift` | 去噪 timestep shift，默认 3.0 |
| `--cfg_scale` / `--cfg_text_scale` | 文本 CFG 强度 |
| `--seed` | 采样随机种子 |
| `--num_frames` | 视频生成帧数 |
| `--video_height` / `--video_width` | 视频/图生视频目标尺寸 |

## 自定义批量输入

如果需要批量任务，可以传入 `--input_json path/to/request.json`。直接参数会被忽略，脚本会按 JSON 内容运行。JSON 格式与脚本自动生成的 `request.json` 一致。
