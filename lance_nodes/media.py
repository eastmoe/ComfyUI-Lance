"""Conversions between ComfyUI IMAGE/VIDEO values and Lance file inputs/outputs."""

from .common import *
from .modeling import _comfy_temp_dir

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




__all__ = [name for name in globals() if not name.startswith("__")]
