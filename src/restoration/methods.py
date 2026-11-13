"""Reconstruction methods and their honest descriptions.

Every method rebuilds sample values from *surrounding* pixels only. None knows
what was actually behind the defect, so none of them may be described as
recovering lost content.

* ``exemplar``  - patch synthesis (see ``exemplar.py``). Continues grain and
  texture by copying nearby patches. Suits textured areas. On a wide gap it
  produces plausible texture, not the original detail.
* ``smooth``    - biharmonic interpolation. Fits a smooth surface to the
  boundary. Suits flat areas (sky, studio backdrop, plain emulsion). Turns
  pasty over texture.
* ``hybrid``    - per connected region, measure surrounding texture energy and
  choose ``exemplar`` (textured) or ``smooth`` (flat). Each region's choice is
  reported.

Wide gaps are additionally flagged by ``speculative_pixels`` so the interface
can mark exactly where the result is a guess of shape, not a recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.restoration import inpaint_biharmonic

from .exemplar import exemplar_fill

METHODS = ("hybrid", "exemplar", "smooth")

METHOD_TEXT = {
    "exemplar": (
        "Patch synthesis. Fills from the boundary inward, voting between the "
        "closest-matching nearby patches, so grain and texture continue across "
        "the repair. Suitable for textured regions. Detail inside a wide gap is "
        "invented plausibly; it is not the lost original."
    ),
    "smooth": (
        "Biharmonic interpolation. Solves a smooth surface matching the "
        "boundary, blending gradients across the gap. Suitable for smooth "
        "regions (sky, backdrop, skin at low frequency). Over texture it turns "
        "pasty, and across a wide gap it is an interpolation, not a recovery."
    ),
    "hybrid": (
        "Per-region choice: surrounding texture energy decides between patch "
        "synthesis (textured) and biharmonic interpolation (flat). The method "
        "chosen for every region is listed in the report so the choice can be "
        "audited."
    ),
}


@dataclass
class RegionDecision:
    region_id: int
    method: str
    texture_energy: float
    area: int
    bbox: tuple[int, int, int, int]


@dataclass
class ReconstructResult:
    image: np.ndarray                      # float32 HxWxC, same channel count
    regions: list[RegionDecision]
    notes: list[str] = field(default_factory=list)


def _texture_energy(gray: np.ndarray, mask: np.ndarray, inner: int = 4, outer: int = 10) -> float:
    """Mean squared gradient in an annulus *outside* the defect's own edge.

    Measuring on the immediate ring would pick up the defect's own boundary
    (a dark scratch against light emulsion is a huge gradient), which would
    misclassify every scratch as textured.  The annulus starts ``inner`` px
    away and ends ``outer`` px away from the mask.
    """
    outside = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    sel = (dist > inner) & (dist <= outer) & ~mask
    if not sel.any():
        sel = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sel = sel & ~mask
        if not sel.any():
            return 0.0
    gy, gx = np.gradient(gray.astype(np.float32))
    return float((gx * gx + gy * gy)[sel].mean())


def _as_float_channel(img: np.ndarray) -> tuple[np.ndarray, bool]:
    if img.ndim == 2:
        return img[:, :, None], True
    return img, False


def inpaint_smooth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Biharmonic interpolation per channel, on float32 data."""
    work, squeeze = _as_float_channel(img.astype(np.float32))
    # skimage's channel-aware form supports arbitrary channel counts (unlike
    # cv2.inpaint, which rejects >1 channel for float/uint16 data).
    out = inpaint_biharmonic(work, mask, channel_axis=2, split_into_regions=True)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return out[:, :, 0] if squeeze else out


def reconstruct(
    img: np.ndarray,
    core: np.ndarray,
    *,
    method: str = "hybrid",
    texture_threshold: float = 3.0e-4,
    seed: int = 0,
    patch: int = 7,
    search_radius: int = 40,
) -> ReconstructResult:
    """Reconstruct inside ``core`` only (float32 in [0,1], HxW or HxWxC).

    Pixels outside ``core`` are returned unchanged; the caller composites and
    feathers.  Per-region method choices and texture evidence are reported.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}; expected one of {METHODS}")
    if not core.any():
        return ReconstructResult(img.astype(np.float32).copy(), [], ["empty mask: nothing reconstructed"])

    work, squeeze = _as_float_channel(img.astype(np.float32))
    gray = work[:, :, 0] if work.shape[2] == 1 else (
        0.2126 * work[:, :, 0] + 0.7152 * work[:, :, 1] + 0.0722 * work[:, :, 2]
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)

    composed = work.copy()
    decisions: list[RegionDecision] = []
    for lab in range(1, n):
        region = labels == lab
        area = int(stats[lab, cv2.CC_STAT_AREA])
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        w0 = int(stats[lab, cv2.CC_STAT_WIDTH])
        h0 = int(stats[lab, cv2.CC_STAT_HEIGHT])
        energy = _texture_energy(gray, region)
        chosen = ("exemplar" if energy >= texture_threshold else "smooth") if method == "hybrid" else method
        if area <= 2 and chosen == "exemplar":
            chosen = "smooth"  # a couple of pixels cannot define a patch
        if chosen == "exemplar":
            filled = exemplar_fill(work, region, patch=patch, search_radius=search_radius, seed=seed)
        else:
            filled = inpaint_smooth(work, region)
        composed[region] = np.clip(filled, 0.0, 1.0)[region]
        decisions.append(RegionDecision(lab, chosen, energy, area, (x0, y0, x0 + w0, y0 + h0)))

    notes = []
    if decisions:
        counts = {}
        for d in decisions:
            counts[d.method] = counts.get(d.method, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        notes.append(f"reconstructed {len(decisions)} region(s): {summary}")
    return ReconstructResult(composed[:, :, 0] if squeeze else composed, decisions, notes)


def conservative_width_proxy(mask: np.ndarray) -> float:
    """Widest inscribed disc minus one pixel ring - an honest gap-width lower bound.

    The distance transform measures to the nearest *background pixel centre*,
    so ``2 * d.max()`` overestimates a strip's width by about one pixel.  The
    subtraction makes this a genuine lower bound.  Used only for reporting; a
    curved defect can be wider than its widest disc.
    """
    if not mask.any():
        return 0.0
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(max(0.0, 2.0 * d.max() - 1.0))


def speculative_pixels(mask: np.ndarray, width_threshold: float) -> np.ndarray:
    """Pixels inside gaps whose local width is at least ``width_threshold`` px."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return mask & (2.0 * d - 1.0 >= float(width_threshold))
