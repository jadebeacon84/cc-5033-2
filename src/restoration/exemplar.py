"""Exemplar-based patch synthesis for textured regions.

Boundary-inward ("onion peel") ordering: pixels closest to known material are
filled first, and each unknown pixel receives its value by voting between the
best matching source patches found in nearby known material.  This continues
grain and texture (unlike a diffusion fill) but it can only rearrange what is
already in the picture: it cannot recreate a face or lettering that no longer
exists.

Search is a randomized local patch search with a global fallback, seeded so
results are reproducible.
"""
from __future__ import annotations

import cv2
import numpy as np


def _offsets(patch: int) -> tuple[np.ndarray, np.ndarray, int]:
    half = patch // 2
    gy, gx = np.mgrid[-half : half + 1, -half : half + 1]
    return gy.ravel(), gx.ravel(), gy.size


def _erode(mask: np.ndarray, patch: int) -> np.ndarray:
    k = np.ones((patch, patch), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def exemplar_fill(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    patch: int = 7,
    search_radius: int = 40,
    candidates: int = 24,
    seed: int = 0,
    smooth_iters: int = 2,
) -> np.ndarray:
    """Fill ``mask`` in ``img`` (float32 [0,1], HxW, HxWx1 or HxWxC) by patch synthesis."""
    if not mask.any():
        return img.copy()
    if mask.all():
        # nothing to learn from; neutral mid-grey is the only honest answer
        return np.full_like(img, 0.5, dtype=np.float32)

    rng = np.random.default_rng(seed)
    work = img.astype(np.float32).copy()
    squeeze = False
    if work.ndim == 2:
        work = work[:, :, None]
        squeeze = True
    else:
        squeeze = False
    H, W, C = work.shape
    oy, ox, npatch = _offsets(patch)
    half = patch // 2

    known = ~mask
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ring_ids = np.ceil(dist).astype(np.int32)
    src_valid = _erode(known, patch)
    src_centers = np.argwhere(src_valid)  # (M,2) as (y,x)
    if len(src_centers) == 0:
        # gap too wide for a patch to fit entirely in known material: shrink
        patch = max(3, patch // 2)
        oy, ox, npatch = _offsets(patch)
        src_valid = _erode(known, patch)
        src_centers = np.argwhere(src_valid)
        if len(src_centers) == 0:
            out = _global_fill(work, mask).astype(np.float32)
            return out[:, :, 0] if squeeze else out

    max_ring = int(ring_ids.max())
    for k in range(1, max_ring + 1):
        sel = (ring_ids == k) & ~known
        if not sel.any():
            continue
        ys, xs = np.nonzero(sel)
        n = ys.size

        # target patch samples with validity flags
        ty = ys[:, None] + oy[None, :]
        tx = xs[:, None] + ox[None, :]
        ty_c = np.clip(ty, 0, H - 1)
        tx_c = np.clip(tx, 0, W - 1)
        tvals = work[ty_c, tx_c, :]                       # (n,P,C)
        known_t = known[ty_c, tx_c]                       # (n,P)
        n_known = np.maximum(known_t.sum(1), 1)

        # candidate source centres: random local + global fallback + neighbours
        cand = np.empty((n, candidates, 2), dtype=np.int32)
        cand[:, :, 0] = ys[:, None] + rng.integers(-search_radius, search_radius + 1, (n, candidates))
        cand[:, :, 1] = xs[:, None] + rng.integers(-search_radius, search_radius + 1, (n, candidates))
        flat = rng.integers(0, len(src_centers), (n, 4))
        cand[:, -4:, 0] = src_centers[flat, 0]
        cand[:, -4:, 1] = src_centers[flat, 1]
        cand[:, :, 0] = np.clip(cand[:, :, 0], half, H - 1 - half)
        cand[:, :, 1] = np.clip(cand[:, :, 1], half, W - 1 - half)

        best_d = np.full(n, np.inf, dtype=np.float32)
        best_src = np.zeros((n, 2), dtype=np.int32)
        for c in range(candidates):
            sy = cand[:, c, 0][:, None] + oy[None, :]
            sx = cand[:, c, 1][:, None] + ox[None, :]
            ok_c = src_valid[cand[:, c, 0], cand[:, c, 1]]          # (n,)
            svals = work[np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1), :]
            diff = ((tvals - svals) ** 2).sum(axis=2)                # (n,P)
            dsum = (diff * known_t).sum(axis=1) / (n_known * C)
            dsum = np.where(ok_c & np.isfinite(dsum), dsum, np.inf)
            better = dsum < best_d
            best_d[better] = dsum[better]
            best_src[better, 0] = cand[better, c, 0]
            best_src[better, 1] = cand[better, c, 1]

        found = np.isfinite(best_d)
        if not found.any():
            continue
        ys, xs, best_src, best_d = ys[found], xs[found], best_src[found], best_d[found]
        n = ys.size

        # vote: add each chosen source patch into an accumulator over its target
        weights = 1.0 / (best_d + 1e-3)
        ay = ys[:, None] + oy[None, :]
        ax = xs[:, None] + ox[None, :]
        sy = best_src[:, 0][:, None] + oy[None, :]
        sx = best_src[:, 1][:, None] + ox[None, :]
        vals = work[np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1), :]   # (n,P,C)
        ay_c = np.clip(ay, 0, H - 1)
        ax_c = np.clip(ax, 0, W - 1)
        valid_dest = mask[ay_c, ax_c]                                # only fill unknowns
        w = (weights[:, None] * valid_dest)[:, :, None]
        acc = np.zeros((H, W, C), np.float64)
        wacc = np.zeros((H, W), np.float64)
        np.add.at(acc, (ay_c.ravel(), ax_c.ravel()), (vals * w).reshape(-1, C))
        np.add.at(wacc, (ay_c.ravel(), ax_c.ravel()), w[:, :, 0].ravel())
        filled = wacc > 0
        work[filled] = (acc[filled] / wacc[filled][:, None]).astype(np.float32)
        known[filled] = True
        src_valid |= _erode(known, patch)

    # Blend only the seam *band* where synthesized patches meet known pixels;
    # blurring the whole filled area would erase the very grain the synthesis
    # was meant to continue (defect hiding, not restoration).
    if smooth_iters > 0 and mask.any():
        near_known = cv2.dilate((~mask).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        seam = mask & near_known
        if seam.any():
            blur = work.copy()
            for _ in range(smooth_iters):
                blur = cv2.GaussianBlur(blur, (0, 0), 1.0)
            if work.shape[2] == 1:
                work[:, :, 0][seam] = blur[seam]
            else:
                work[seam] = blur[seam]
    return work[:, :, 0] if squeeze else work


def _global_fill(work: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.restoration import inpaint_biharmonic

    out = inpaint_biharmonic(work, mask, channel_axis=2, split_into_regions=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
