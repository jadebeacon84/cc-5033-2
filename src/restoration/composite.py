"""Composite the reconstruction back at the original bit depth and prove preservation.

The writer does not merely claim that untouched pixels are unchanged: it re-reads
its own output file and compares every sample outside the permitted zone with the
source.  Any deviation is reported as a failure, so the "keep" policy is
demonstrated rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .imageio import ImageData, save_image
from .maskzones import MaskZones


@dataclass
class PreservationCheck:
    outside_allowed_pixels: int
    changed_outside_allowed: int
    max_abs_delta_outside: float
    core_changed: int
    band_changed: int
    ok: bool
    note: str


def _round_to_dtype(arr: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
    if dtype == np.float32:
        return arr.astype(np.float32)
    return np.clip(np.rint(arr * maxval), 0, maxval).astype(dtype)


def composite(
    original: ImageData,
    reconstructed: np.ndarray,
    zones: MaskZones,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (output_raw, allowed_mask) at the original dtype.

    Reconstruction is applied in ``core`` fully and in ``band`` weighted by the
    feather, so the band is a transition, not a second repair region.  Pixels
    outside ``core | band`` get the exact original samples.
    """
    orig_f = original.to_float()
    rec_f = reconstructed[:, :, None] if reconstructed.ndim == 2 else reconstructed
    if rec_f.shape[2] != orig_f.shape[2]:
        raise ValueError(f"channel mismatch: reconstruction {rec_f.shape[2]}, original {orig_f.shape[2]}")

    feather = zones.feather[:, :, None]
    blended = orig_f * (1.0 - feather) + rec_f * feather
    out_f = orig_f.copy()
    allowed = zones.allowed
    out_f[allowed] = blended[allowed]
    out_raw = _round_to_dtype(out_f, original.raw.dtype, original.maxval)
    if original.raw.ndim == 2:
        out_raw = out_raw[:, :, 0]
    return out_raw, allowed


def verify_preservation(
    source_path: str | Path,
    output_path: str | Path,
    zones,
    *,
    tolerance: int = 0,
) -> PreservationCheck:
    """Re-read the written file and compare against the source outside the zone.

    Comparison is done on the *file as written*, with the original channel
    layout (alpha, if any, is excluded).  ``tolerance`` is in raw sample units.
    """
    src = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    out = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    if src is None or out is None:
        raise RuntimeError("verification could not re-read source or output")
    if src.ndim == 3 and src.shape[2] == 4:
        src = src[:, :, :3]
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[:, :, 0]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    if src.shape != out.shape:
        raise RuntimeError(f"shape changed: {src.shape} -> {out.shape}")

    delta = np.abs(src.astype(np.int64) - out.astype(np.int64))
    if delta.ndim == 3:
        delta = delta.max(axis=2)

    outside = ~zones.allowed
    changed_outside = int((delta[outside] > tolerance).sum())
    max_delta_outside = int(delta[outside].max()) if outside.any() else 0
    core_changed = int((delta[zones.core] > tolerance).sum())
    band_changed = int((delta[zones.band] > tolerance).sum())
    return PreservationCheck(
        outside_allowed_pixels=int(outside.sum()),
        changed_outside_allowed=changed_outside,
        max_abs_delta_outside=float(max_delta_outside),
        core_changed=core_changed,
        band_changed=band_changed,
        ok=changed_outside == 0,
        note=(
            "pass: every sample outside core+band is bit-identical"
            if changed_outside == 0
            else f"FAIL: {changed_outside} sample(s) changed outside the allowed zone"
        ),
    )
