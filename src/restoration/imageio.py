"""Image and mask I/O that preserves bit depth, channel count and exact sample values."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(RuntimeError):
    pass


@dataclass
class ImageData:
    raw: np.ndarray            # HxW (gray) or HxWxC (color), original dtype
    alpha: np.ndarray | None   # HxW original alpha if the file carried one
    is_gray: bool
    path: Path | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape[:2]

    @property
    def channels(self) -> int:
        return 1 if self.raw.ndim == 2 else self.raw.shape[2]

    @property
    def dtype(self) -> np.dtype:
        return self.raw.dtype

    @property
    def maxval(self) -> float:
        if self.raw.dtype == np.uint8:
            return 255.0
        if self.raw.dtype == np.uint16:
            return 65535.0
        return 1.0

    def to_float(self) -> np.ndarray:
        """Normalized float32 HxWxC (single channel kept as HxWxC with C=1)."""
        arr = self.raw.astype(np.float32)
        if self.raw.dtype != np.float32:
            arr /= self.maxval
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr


def load_image(path: str | Path) -> ImageData:
    p = Path(path)
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ImageLoadError(f"cannot read image: {p}")

    alpha = None
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3].copy()
        raw = raw[:, :, :3].copy()
    elif raw.ndim == 3 and raw.shape[2] == 2:
        alpha = raw[:, :, 1].copy()
        raw = raw[:, :, 0].copy()
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[:, :, 0]

    is_gray = raw.ndim == 2
    return ImageData(raw=raw, alpha=alpha, is_gray=is_gray, path=p)


def save_image(
    path: str | Path,
    raw: np.ndarray,
    *,
    is_gray: bool = False,
    alpha: np.ndarray | None = None,
    jpeg_quality: int = 95,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = raw
    if alpha is not None:
        base = arr if arr.ndim == 3 else arr[:, :, None]
        arr = np.dstack([base, alpha])

    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        if arr.dtype != np.uint8:
            raise ValueError("JPEG output requires 8-bit samples; use PNG for this image")
        ok = cv2.imwrite(str(p), arr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        ok = cv2.imwrite(str(p), arr)
    if not ok:
        raise ImageLoadError(f"cannot write image: {p}")
    return p


def load_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a mask; any non-black pixel is selected. 4-channel masks use their alpha."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageLoadError(f"cannot read mask: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 3] if arr.shape[2] == 4 else arr.max(axis=2)
    maxv = 1.0 if arr.dtype == np.float32 else float(np.iinfo(arr.dtype).max)
    mask = arr.astype(np.float32) > (maxv * 0.35)
    if shape is not None and mask.shape != tuple(shape):
        mask = cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        ) > 0
    return mask


def save_mask(path: str | Path, mask: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), (mask.astype(np.uint8) * 255))
    if not ok:
        raise ImageLoadError(f"cannot write mask: {p}")
    return p


def to_display_png(src: str | Path, dst: str | Path, max_side: int | None = None) -> Path:
    """8-bit sRGB PNG for browser display (never used as the delivered result)."""
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ImageLoadError(f"cannot read image: {src}")
    if raw.dtype == np.uint16:
        raw = (raw / 257.0).round().astype(np.uint8)
    elif raw.dtype != np.uint8:
        raw = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    if raw.ndim == 3 and raw.shape[2] == 1:
        raw = raw[:, :, 0]
    if max_side and max(raw.shape[:2]) > max_side:
        scale = max_side / max(raw.shape[:2])
        raw = cv2.resize(
            raw, (max(1, int(raw.shape[1] * scale)), max(1, int(raw.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), raw)
    return dst


def gray_float(img: np.ndarray) -> np.ndarray:
    """Luminance in [0,1] for a float HxW or HxWxC array."""
    arr = img[:, :, None] if img.ndim == 2 else img
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    return (
        0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    ).astype(np.float32)
