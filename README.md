# ComfyUI-Lance

`ComfyUI-Lance` 是对 `bytedance-research/Lance` 本地推理代码的 ComfyUI 自定义节点封装，提供图片理解、图片生成、图片编辑、视频理解、图片生成视频、文字生成视频和视频编辑能力。

相较于 `temp/Lance` 中的上游工程，这里把 Lance 改造成了更适合 ComfyUI 长时间运行的轻量推理版本，重点做了依赖瘦身、ComfyUI 原生类型适配、模型缓存、任务取消、显存清理和错误提示等健壮性改造。

## 安装

将仓库放到 ComfyUI 的 `custom_nodes` 目录：

```text
ComfyUI/
  custom_nodes/
    ComfyUI-Lance/
```

安装本节点额外依赖：

```bash
pip install -r requirements.txt
```

如果使用 ComfyUI Windows 便携包，可以在 ComfyUI 根目录执行：

```powershell
.\python_embeded\python.exe -m pip install -r .\custom_nodes\ComfyUI-Lance\requirements.txt
```

视频理解、视频编辑和视频生成相关节点需要系统能访问 `ffmpeg` 和 `ffprobe`。请先安装 FFmpeg，并确认它们在 `PATH` 中：

```bash
ffmpeg -version
ffprobe -version
```

安装完成后重启 ComfyUI，在节点菜单中选择 `Lance/多模态`。

## 模型目录

默认从 `ComfyUI/models/Lance` 加载模型：

```text
ComfyUI/models/Lance/
  Lance_3B/
  Lance_3B_Video/
  Qwen2.5-VL-ViT/
  Wan2.2_VAE.pth
ComfyUI/models/Lance-quantized-cache/
```

模型来源：<https://huggingface.co/bytedance-research/Lance>

`Lance 加载模型` 节点中 `模型根目录` 填 `auto` 即使用上述路径，也可以填写自定义绝对路径。

如果打开 `缺失时下载`，节点会在模型缺失时从 Hugging Face 下载到模型根目录；`下载源` 支持 `huggingface.co` 和 `hf-mirror.com`。自动下载需要当前 ComfyUI 环境中可用 `huggingface_hub`，网络不可用时请手动下载模型。

## 节点

| 节点 | 用途 |
| --- | --- |
| `Lance 加载模型` | 创建 Lance 模型句柄，支持懒加载、设备选择、attention backend、量化和自动下载。 |
| `Lance 图片理解` | 输入 ComfyUI `IMAGE` 和问题，输出文本回答。 |
| `Lance 图片生成` | 文生图，输出 ComfyUI `IMAGE` 和结果路径。 |
| `Lance 图片编辑` | 输入 `IMAGE` 和编辑指令，输出编辑后的 `IMAGE`。 |
| `Lance 视频理解` | 输入 ComfyUI `VIDEO` 和问题，输出文本回答。 |
| `Lance 图片生成视频` | 输入参考 `IMAGE` 和 prompt，输出 `VIDEO`、帧序列和结果路径。 |
| `Lance 文字生成视频` | 文生视频，输出 `VIDEO`、帧序列和结果路径。 |
| `Lance 视频编辑` | 输入 `VIDEO` 和编辑指令，输出编辑后的视频。 |
| `Lance 帧转视频` | 将 ComfyUI `IMAGE` 帧序列打包成 `VIDEO`。 |
| `Lance 释放模型` | 释放 Lance 运行时缓存，并可清理 CUDA cache。 |

节点使用 ComfyUI 原生 `IMAGE`、`VIDEO` 和 `STRING` 类型，便于和内置 Load Image、Load Video、Save Image、Save Video、文本节点和其他图像处理节点连接。

## 基本用法

1. 添加 `Lance 加载模型`。
2. `模型根目录` 保持 `auto`，`加载范围` 推荐先用 `auto/lazy`，首次运行具体任务时再加载对应图像或视频模型。
3. 将 `lance_model` 输出连接到图片理解、图片生成、图片编辑或视频节点。
4. 图片生成/编辑结果可接 `Save Image`；视频生成/编辑结果可接 ComfyUI 的 `Save Video`。
5. 任务结束后如需释放显存，连接并运行 `Lance 释放模型`。

仓库内提供了几个可直接拖入 ComfyUI 的示例工作流：

```text
workflows/
  Lance文生图.json
  Lance图片编辑.json
  Lance图片理解.json
```

## 加速与量化

`Lance 加载模型` 提供这些关键选项：

- `设备`：`auto`、`cuda`、`cuda:0` 等。
- `加载范围`：`auto/lazy`、`image`、`video`、`image+video`。
- `Compute dtype`：`bf16`、`fp16`、`fp32`。
- `Attention Backend`：`auto`、`flash_attention_2`、`sage_attention`、`sdpa`。
- `量化加载`：`none`、`int8`、`int4`、`fp4`、`fp8_e4m3fn`、`fp8_e5m2`。
- `量化缓存`：启用后，首次量化会分块读取 checkpoint、边加载边量化，并在 `ComfyUI/models/Lance-quantized-cache` 保存可复用缓存。
- `重建量化缓存`：忽略已有缓存并重新从原始权重量化生成。
- `启用 KV Cache`：生成视觉内容时启用 Lance 官方 KV cache 路径，减少重复注意力计算。

`flash_attention_2` 和 `sage_attention` 需要对应 Python 包已安装；未安装时请切回 `auto` 或 `sdpa`。量化是实验性的 Linear 包装，主要用于降低权重常驻显存；如果生成质量或兼容性异常，请先切回 `none`。

图片/视频生成节点还提供 `VAE Decode Mode`、`VAE Tile Size` 和 `VAE Tile Overlap`，可在显存紧张或特定后端下使用 VAE 分块解码。

## 相较原项目的改造

相较于 `temp/Lance` 中的原项目，本仓库主要做了这些适配：

- 依赖瘦身：原项目 `requirements.txt` 中大量固定版本依赖已移除，ComfyUI 已提供的 `torch`、`transformers`、`numpy`、`pillow`、`safetensors`、`einops` 等不再重复钉死版本。
- 依赖替代：用系统 `ffmpeg/ffprobe` 替代 `decord` 和 `imageio-ffmpeg`，减少 Windows/便携包环境下的视频解码安装问题。
- 去除 Jinja2：Qwen-VL prompt 渲染改为轻量字符串组装，避免为了单一模板引入额外依赖。
- 轻量推理化：移除了 Gradio demo、benchmark、训练、验证和微调相关入口，只保留 ComfyUI 节点需要的核心推理路径。
- ComfyUI 集成：使用 `folder_paths` 获取模型、输出和临时目录；输入输出对齐 ComfyUI 原生 `IMAGE`、`VIDEO`、`STRING`。
- 运行健壮性：在节点边界、文本生成循环和去噪采样循环接入 ComfyUI 中断检查，点击取消后会在下一个检查点退出。
- 显存管理：加入运行时缓存、懒加载、模型释放节点、临时对象清理和 CUDA cache 清理。
- 量化缓存：支持分块读取 checkpoint 进行 int8/int4/fp4/fp8 量化，缓存包含元数据校验，缓存失配时可重建。
- 错误提示：模型目录、CUDA 设备、checkpoint、视频输入类型和外部工具缺失时会给出更直接的报错信息。

## 注意事项

- Lance 当前推理路径需要 CUDA 设备；大分辨率图像和视频生成对显存要求较高。
- 视频节点要求 ComfyUI 环境支持 `VIDEO` 类型，并要求系统已安装 `ffmpeg/ffprobe`。
- 首次加载模型或首次创建量化缓存会比较慢；后续会复用 ComfyUI 运行时缓存或量化缓存。
- 如果遇到 attention backend、量化或 VAE 分块相关问题，优先使用 `attention_backend=sdpa`、`量化加载=none`、`VAE Decode Mode=normal` 排查。
