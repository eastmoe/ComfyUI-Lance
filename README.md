# ComfyUI-Lance

ComfyUI 自定义节点封装，基于 `bytedance-research/Lance` 本地推理代码，提供图片理解、图片生成、图片编辑、视频理解、图片生成视频、文字生成视频和视频编辑节点。

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

## 节点

- `Lance 加载模型`
- `Lance 图片理解`
- `Lance 图片生成`
- `Lance 图片编辑`
- `Lance 视频理解`
- `Lance 图片生成视频`
- `Lance 文字生成视频`
- `Lance 视频编辑`
- `Lance 帧转视频`
- `Lance 释放模型`

节点使用 ComfyUI 原生 `IMAGE`、`VIDEO` 和 `STRING` 类型，便于和内置 Load/Save Video、图片处理、文本节点连接。

## 加速与量化

加载节点提供：

- 设备选择：`auto`、`cuda`、`cuda:0` 等。
- Attention Backend：`auto`、`flash_attention_2`、`sage_attention`、`sdpa`。
- 量化加载：`none`、`int8`、`int4`、`fp8_e4m3fn`、`fp8_e5m2`。
- 量化缓存：启用后，首次量化会分块读取 checkpoint、边加载边量化，并在 `ComfyUI/models/Lance-quantized-cache` 保存可复用缓存；后续同一权重与量化模式会直接加载缓存。需要强制刷新时打开“重建量化缓存”。

`flash_attention_2` 和 `sage_attention` 需要对应 Python 包已安装；未安装时请切回 `auto` 或 `sdpa`。量化为轻量 Linear 包装，主要降低权重常驻显存，速度取决于设备和 PyTorch 支持情况。

## 取消任务

插件在 Lance 的采样循环、文本生成循环和 ComfyUI 节点边界都接入了 ComfyUI 的中断检查。点击 ComfyUI 的取消/中断后，当前 Lance 任务会在下一个检查点抛出中断并释放临时显存。
