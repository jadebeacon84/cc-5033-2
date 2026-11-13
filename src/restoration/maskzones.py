"""Mask zoning: the preservation contract.

Three nested zones decide what may change:

* ``core``  - pixels the user selected.  Reconstruction happens here.
* ``band``  - ``core`` dilated by ``band_px``.  Only used for a feathered
              transition so the repair does not end on a hard seam.  Reported.
* ``keep``  - every other pixel.  Copied verbatim from the original; the writer
              verifies this with an exact comparison, so no wide, unrequested
              edits can occur.

``band_px=0`` disables the transition band entirely (a visible hard edge then,
but nothing outside the selected pixels changes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MaskZones:
    core: np.ndarray            # bool HxW user-selected
    band: np.ndarray            # bool HxW core-dilated minus core
    feather: np.ndarray         # float32 HxW, 1 in core, 1..0 across band
    band_px: int
    dropped_components: int = 0
    dropped_pixels: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> np.ndarray:
        """Pixels the pipeline is permitted to touch (core ∪ band)."""
        return self.core | self.band

    @property
    def core_pixels(self) -> int:
        return int(self.core.sum())

    @property
    def band_pixels(self) -> int:
        return int(self.band.sum())


def _disc(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r + 0.01).astype(np.uint8)


def drop_small_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int, int]:
    """Remove connected components below ``min_area`` pixels (explicit user opt-in)."""
    if min_area <= 0:
        return mask, 0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    dropped = 0
    dropped_px = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep |= labels == i
        else:
            dropped += 1
            dropped_px += area
    return keep, dropped, dropped_px


def build_zones(
    mask: np.ndarray,
    band_px: int = 2,
    min_area: int = 0,
    max_band_fraction: float = 0.05,
) -> MaskZones:
    """Construct core/band/feather zones with a bounded transition band.

    ``max_band_fraction`` caps band pixels as a fraction of the *image area*
    (floor 2000 px).  Thin scratch masks naturally need a band several times
    larger than their core, so the cap is relative to the image, not the mask.
    The effective band width is reported.
    """
    core = mask.astype(bool)
    notes: list[str] = []
    dropped_c = dropped_px = 0
    if min_area > 0:
        core, dropped_c, dropped_px = drop_small_components(core, min_area)
        notes.append(
            f"removed {dropped_c} components smaller than {min_area}px "
            f"({dropped_px}px)" if dropped_c else "no speckle below min_area"
        )

    shape = core.shape
    band = np.zeros(shape, dtype=bool)
    band_px_used = 0
    if core.any() and band_px > 0:
        cap = max(2000, int(max_band_fraction * core.size))
        # Grow the band one ring at a time until the ring budget is exhausted.
        grown = core.copy()
        for r in range(1, int(band_px) + 1):
            cand = cv2.dilate(grown.astype(np.uint8), _disc(1), iterations=1) > 0
            new_ring = cand & ~grown
            if band.sum() + int(new_ring.sum()) > cap:
                notes.append(
                    f"transition band capped at {int(band.sum())}px "
                    f"(would have reached {int(band.sum() + new_ring.sum())}px)"
                )
                break
            grown = cand | grown
            band = grown & ~core
            band_px_used = r
    elif core.any():
        notes.append("transition band disabled (band_px=0): repair may show a hard edge")

    feather = np.zeros(shape, dtype=np.float32)
    feather[core] = 1.0
    if band.any():
        # OpenCV's transform measures distance to the nearest *zero*.  Marking
        # core+band as nonzero therefore yields, for each band pixel, its
        # distance to the nearest keep pixel: ~1 at the outer edge, the band
        # width beside core.
        dist = cv2.distanceTransform((core | band).astype(np.uint8), cv2.DIST_L2, 5)
        # Keep the band's weights strictly inside (0, 1) so no band pixel is
        # treated as fully repaired; dmax+1 leaves the outer ring explicit.
        dmax = float(dist[band].max()) + 1.0 if band.any() else 1.0
        feather[band] = np.clip(dist[band] / max(dmax, 1.0), 0.0, 1.0)
    return MaskZones(
        core=core,
        band=band,
        feather=feather,
        band_px=band_px_used,
        dropped_components=dropped_c,
        dropped_pixels=dropped_px,
        notes=notes,
    )


def boundary_preview(rgb: np.ndarray, zones: MaskZones) -> np.ndarray:
    """Overlay for checking the transition: core blue, band amber, keep untouched.

    ``rgb`` is an OpenCV BGR array; the tint constants are given in BGR order.
    """
    out = rgb.copy()
    band_bgr = np.array([0.0, 0.62, 1.0])
    core_bgr = np.array([1.0, 0.42, 0.08])
    out[zones.band] = 0.55 * out[zones.band] + 0.45 * band_bgr
    out[zones.core] = 0.55 * out[zones.core] + 0.45 * core_bgr
    return out
