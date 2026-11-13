"""Scratch/dust candidate proposal, with visible evidence and structure rejection.

A candidate mask is only a *proposal*.  Nothing is removed until the user
reviews it and writes a mask; the delivered mask is the file the user edits.
The CLI refuses to restore from detection alone unless ``--auto-mask`` is
given explicitly, and the UI requires an explicit "apply candidates" step.

Detection is deliberately conservative and evidence-based.  The hard problem is
telling a scratch from legitimate thin structure (building edges, carved seams,
railings, hair).  A scratch has these measurable signatures, and the detector
requires all of them:

1. **Unpredicted by its flanks.** A thin scratch sits on the same surface as
   its immediate surroundings, so the pixels a few px to either side predict
   it; its own value deviates strongly.  A building edge does not satisfy this
   because each side predicts a different value.
2. **Symmetric flanks.** The two sides agree with each other.  An edge between
   two different surfaces fails this (the ``side contrast`` veto).
3. **Not local texture.**  The stroke contrast is measured against the mean
   fine-scale energy of the surrounding surface (parallel bands beside the
   stroke), not against the sensor floor alone.  Wood grain and brickwork
   produce many thin ridges, but their surroundings are equally busy, so the
   ratio stays low.
4. **Even along its length, and it ends.**  Contrast persists with near-constant
   coverage, but does not continue past both endpoints (``continues beyond
   ends`` veto).  A carved seam runs out of the stroke and keeps going.
5. **Isolated.**  A carved molding or frame edge has parallel companions a
   few px away; a scratch is alone.  The ``parallel partners`` veto rejects
   any candidate with a parallel, overlapping ridge beside it.

Every proposal is reported with its measured numbers whether accepted or
rejected.  Fragments are not merged: a long scratch can yield several
overlapping Hough proposals, all of which the user can see and select.

Dust/speck auto-detection is **off by default**.  On the reference photographs
the fine-scale texture around a genuine dust speck is itself as busy as the
speck, so "isolated on a smooth surface" is not a separable signature; firing
on it would select thousands of texture grains.  When enabled, it is gated on
local isolation and capped hard, and still requires review.

Wide gaps are, by construction, invisible to a thin-line detector - the report
states this limitation rather than hiding it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class Candidate:
    candidate_id: int
    kind: str                      # "line" | "speck"
    accepted: bool
    reason: str
    area: int
    length: float
    width: float
    curvature: float               # traced ridge drift, px (0 straight)
    side_contrast: float           # |left flank - right flank| / stroke contrast
    line_contrast: float           # stroke contrast, sensor-noise units
    coverage: float                # fraction of length where the stroke stands out
    bbox: tuple[int, int, int, int]
    end_contrast: tuple[float, float] = (0.0, 0.0)
    texture_ratio: float = 0.0     # stroke contrast / surrounding fine-scale energy
    parallel_partners: int = 0
    segments: list[tuple[int, int, int, int]] = field(repr=False, default_factory=list)
    mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), bool))

    def summary(self) -> dict:
        return {
            "id": self.candidate_id, "kind": self.kind, "accepted": self.accepted,
            "reason": self.reason, "area": self.area,
            "length": round(self.length, 1), "width": round(self.width, 1),
            "side_contrast": round(self.side_contrast, 3),
            "line_contrast": round(self.line_contrast, 1),
            "coverage": round(self.coverage, 2),
            "drift": round(self.curvature, 2),
            "end_contrast": [round(v, 1) for v in self.end_contrast],
            "texture_ratio": round(self.texture_ratio, 2),
            "parallel_partners": self.parallel_partners,
            "bbox": list(self.bbox),
            "segments": [list(s) for s in self.segments],
        }


@dataclass
class DetectParams:
    sensitivity: float = 0.55       # 0 conservative .. 1 permissive
    min_length: int = 60            # shortest stroke, px
    min_contrast_sigma: float = 8.0
    min_coverage: float = 0.85
    max_side_contrast: float = 0.6
    min_texture_ratio: float = 4.0
    max_drift: float = 1.0          # px; non-linear wander of the traced ridge
    max_end_contrast: float = 3.0   # sigma; stroke continuing past both ends
    border_margin: int = 6
    veto_parallel: bool = True
    parallel_gap_min: float = 5.0
    parallel_gap_max: float = 34.0
    parallel_angle_deg: float = 6.0
    parallel_min_overlap: float = 0.35
    include_specks: bool = False
    speck_max_area: int = 60
    speck_min_sigma: float = 8.0
    max_exposure_fraction: float = 0.006
    n_orientations: int = 12
    hough_gap: int = 14              # px; links a faint scratch across noise gaps

    def threshold_sigma(self) -> float:
        # sensitivity 0 -> 9 sigma, 1 -> 4 sigma
        return 9.0 - 5.0 * float(np.clip(self.sensitivity, 0.0, 1.0))


# ------------------------------------------------------------------ helpers

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        arr = img
    elif img.shape[2] == 1:
        arr = img[:, :, 0]
    else:
        arr = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr /= 65535.0 if arr.max() > 255.5 else 255.0
    return arr


def sensor_noise(g: np.ndarray) -> float:
    """Robust sensor-noise floor from the flattest tiles of the high-pass image.

    Floored to 8-bit quantisation (1/255) so a mathematically noiseless field
    cannot turn ordinary grain into thousands-of-sigma "peaks".
    """
    hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    tile = 32
    h, w = g.shape
    vals = []
    for y in range(0, max(1, h - tile), tile):
        for x in range(0, max(1, w - tile), tile):
            t = np.abs(hp[y:y + tile, x:x + tile])
            if t.size:
                vals.append(float(np.median(t)) * 1.4826)
    if not vals:
        base = float(np.median(np.abs(hp)) * 1.4826)
    else:
        vals = np.sort(np.array(vals))
        base = float(np.median(vals[: max(4, len(vals) // 4)]))
    return max(base, 1.0 / 255.0) + 1e-9


def perpendicular_predictor(g: np.ndarray, n_ang: int, sep: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Max over orientations of |g - mean(two flank samples perpendicular)|."""
    h, w = g.shape
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    best = np.zeros_like(g)
    signed = np.zeros_like(g)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    for i in range(n_ang):
        th = np.pi * i / n_ang
        nx, ny = -np.sin(th), np.cos(th)
        ay = np.clip((yy + ny * sep).round().astype(np.int32), 0, h - 1)
        ax = np.clip((xx + nx * sep).round().astype(np.int32), 0, w - 1)
        by = np.clip((yy - ny * sep).round().astype(np.int32), 0, h - 1)
        bx = np.clip((xx - nx * sep).round().astype(np.int32), 0, w - 1)
        avg = 0.5 * (b[ay, ax] + b[by, bx])
        dev = b - avg
        take = np.abs(dev) > np.abs(signed)
        best = np.where(take, np.abs(dev), best)
        signed = np.where(take, dev, signed)
    return best, signed


def _texture_floor(g: np.ndarray, p1, p2, inner=6.0, outer=11.0, n=64) -> float:
    """Median fine-scale energy in parallel bands beside the stroke."""
    h, w = g.shape
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.2))
    t = np.linspace(0, 1, n)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = float(np.hypot(dx, dy)) + 1e-6
    nx, ny = -dy / ln, dx / ln
    xs = p1[0] + dx * t
    ys = p1[1] + dy * t
    vals = []
    for off in (-1, 1):
        for dist in (inner, (inner + outer) / 2, outer):
            px = np.rint(xs + off * dist * nx).astype(np.int32)
            py = np.rint(ys + off * dist * ny).astype(np.int32)
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if ok.any():
                vals.append(float(np.median(hp[py[ok], px[ok]])))
    return float(np.median(vals)) if vals else 0.0


def measure_stroke(g, ns, p1, p2, half=5.0, side=4.0, n=64, max_drift=1.2) -> dict | None:
    """Cross-profile per sample, following the ridge within +/-``half`` px.

    A scratch is often 1 px wide and slightly irregular; a single constant
    offset drifts off it and under-reports contrast.  We instead take a cross
    profile at every sample, pick the strongest ridge offset per sample, and
    report median contrast / coverage plus how *non-linear* the traced offsets
    are.  Hair curves away and scores high drift; a scratch traces nearly
    straight.  The whole thing is vectorised over the profile grid.
    """
    h, w = g.shape
    offs = np.arange(-half, half + 1e-9, 0.5)
    si = max(1, int(round(side / 0.5)))
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    t = np.linspace(0, 1, n)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = float(np.hypot(dx, dy)) + 1e-6
    nx, ny = -dy / ln, dx / ln
    bx = (p1[0] + dx * t)[None, :] + offs[:, None] * nx
    by = (p1[1] + dy * t)[None, :] + offs[:, None] * ny
    ok = (bx >= 0) & (bx <= w - 1) & (by >= 0) & (by <= h - 1)
    prof = cv2.remap(b, bx.astype(np.float32), by.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    K = len(offs)
    if K <= 2 * si:
        return None
    core = prof[si:K - si]
    left = prof[:K - 2 * si]
    right = prof[2 * si:]
    contrast = np.abs(core - 0.5 * (left + right))
    valid = ok[si:K - si] & ok[:K - 2 * si] & ok[2 * si:]
    contrast = np.where(valid, contrast, -1.0)
    k_best = np.argmax(contrast, axis=0)
    cols = np.arange(n)
    best = contrast[k_best, cols]
    good = best >= 0
    if good.sum() < 0.6 * n:
        return None
    off_best = offs[si + k_best]
    line = core[k_best, cols]
    lft = left[k_best, cols]
    rgt = right[k_best, cols]
    sel = good
    c = best[sel]
    off = off_best[sel]
    rr = np.linspace(0.0, 1.0, int(sel.sum()))
    if off.size >= 8:
        coef = np.polyfit(rr, off, 1)
        drift = float(np.std(off - np.polyval(coef, rr)))
    else:
        drift = 9.9
    asym = (abs(float(np.median(lft[sel])) - float(np.median(rgt[sel])))
            / (float(np.median(c)) + 1e-9))
    return {
        "line_contrast": float(np.median(c)) / ns,
        "side_contrast": asym,
        "coverage": float((c >= 3.0 * ns).mean()),
        "drift": drift,
        "width": float(np.median(np.abs(off))),
    }


def _end_contrast(g, ns, p1, p2, ext=28, side=4.0) -> tuple[float, float]:
    """Symmetric stroke contrast in the ``ext`` px beyond each endpoint."""
    h, w = g.shape
    b = cv2.GaussianBlur(g, (0, 0), 1.0)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ln = float(np.hypot(dx, dy)) + 1e-6
    ux, uy = dx / ln, dy / ln
    nx, ny = -uy, ux
    out = []
    for end, sgn in ((p1, -1.0), (p2, 1.0)):
        vals = []
        for dist in np.arange(4.0, ext + 1e-9, 4.0):
            cx = end[0] + sgn * ux * dist
            cy = end[1] + sgn * uy * dist
            px = np.rint(cx + np.array([-side, 0.0, side]) * nx).astype(np.int32)
            py = np.rint(cy + np.array([-side, 0.0, side]) * ny).astype(np.int32)
            if not ((0 <= px).all() and (px < w).all() and (0 <= py).all() and (py < h).all()):
                continue
            a, c, bb = b[py[0], px[0]], b[py[1], px[1]], b[py[2], px[2]]
            vals.append(float(min(abs(c - a), abs(c - bb))))
        out.append(float(np.median(vals)) / ns if vals else 0.0)
    return out[0], out[1]


def _parallel_partners(seg, segs: np.ndarray, p: DetectParams) -> int:
    """Other proposed ridges running parallel to ``seg`` at 5..34 px laterally.

    A carved frame edge has companions; an isolated scratch does not.
    """
    x1, y1, x2, y2 = seg
    ln = float(np.hypot(x2 - x1, y2 - y1)) + 1e-9
    ux, uy = (x2 - x1) / ln, (y2 - y1) / ln
    nx, ny = -uy, ux
    ang = float(np.arctan2(y2 - y1, x2 - x1)) % np.pi
    others = segs
    if others.size == 0:
        return 0
    oa = (np.arctan2(others[:, 3] - others[:, 1], others[:, 2] - others[:, 0]) % np.pi)
    da = np.abs(oa - ang)
    da = np.minimum(da, np.pi - da)
    mxx = (others[:, 0] + others[:, 2]) / 2.0
    myy = (others[:, 1] + others[:, 3]) / 2.0
    lat = np.abs((mxx - x1) * nx + (myy - y1) * ny)
    tl = np.hypot(others[:, 2] - others[:, 0], others[:, 3] - others[:, 1]) + 1e-9
    u2x = (others[:, 2] - others[:, 0]) / tl
    u2y = (others[:, 3] - others[:, 1]) / tl
    flip = (u2x * ux + u2y * uy) < 0
    u2x = np.where(flip, -u2x, u2x)
    u2y = np.where(flip, -u2y, u2y)
    pr0 = (others[:, 0] - x1) * ux + (others[:, 1] - y1) * uy
    pr1 = (pr0 + (others[:, 2] - others[:, 0]) * ux * np.where(flip, -1, 1)
           + (others[:, 3] - others[:, 1]) * uy * np.where(flip, -1, 1))
    lo = np.minimum(pr0, pr1); hi = np.maximum(pr0, pr1)
    ov = np.maximum(0.0, np.minimum(hi, ln) - np.maximum(lo, 0.0)) / np.minimum(ln, tl)
    sel = ((da <= np.deg2rad(p.parallel_angle_deg))
           & (lat >= p.parallel_gap_min) & (lat <= p.parallel_gap_max)
           & (ov >= p.parallel_min_overlap))
    # do not count a segment as its own partner
    sel &= ~((np.abs(others[:, 0] - x1) < 2) & (np.abs(others[:, 1] - y1) < 2)
             & (np.abs(others[:, 2] - x2) < 2) & (np.abs(others[:, 3] - y2) < 2))
    return int(sel.sum())


# ------------------------------------------------------------------ detector

def detect_candidates(
    img: np.ndarray,
    params: DetectParams | None = None,
) -> tuple[np.ndarray, list[Candidate], dict]:
    """Propose defect candidates.  Returns (accepted_mask, all_candidates, diagnostics)."""
    p = params or DetectParams()
    g = to_gray(img)
    h, w = g.shape
    ns = sensor_noise(g)
    thr = p.threshold_sigma()
    resp, _signed = perpendicular_predictor(g, p.n_orientations)

    # Segmentation map for Hough.  A fine flank separation (1.5 px) keeps
    # grooves a few px apart distinct; the 4 px form used for measurement
    # merges them.  Only the *segmentation* uses the fine map.
    resp_fine, _ = perpendicular_predictor(g, p.n_orientations, sep=1.5)
    raw = resp_fine >= thr * ns
    # Light closing bridges dotted noise; heavier closing would merge the
    # companions the parallel veto needs.  Skeletonising yields one centreline
    # per ridge, so Hough does not return overlapping chords across a blob.
    cand = skeletonize(cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE,
                                       np.ones((3, 3), np.uint8)) > 0)
    lines = cv2.HoughLinesP(
        (cand.astype(np.uint8) * 255), 1, np.pi / 720, 20,
        minLineLength=p.min_length, maxLineGap=p.hough_gap,
    )
    segs = [] if lines is None else [tuple(int(v) for v in r)
                                     for r in np.asarray(lines, np.int32).reshape(-1, 4)]
    segs.sort(key=lambda s: -float(np.hypot(s[2] - s[0], s[3] - s[1])))
    seg_arr = np.array(segs, np.float32) if segs else np.zeros((0, 4), np.float32)

    candidates: list[Candidate] = []
    accepted_mask = np.zeros((h, w), bool)
    cid = 0
    thick = 3
    for (x1, y1, x2, y2) in segs:
        length = float(np.hypot(x2 - x1, y2 - y1))
        m = measure_stroke(g, ns, (x1, y1), (x2, y2))
        if m is None:
            continue
        ends = _end_contrast(g, ns, (x1, y1), (x2, y2))
        texture = _texture_floor(g, (x1, y1), (x2, y2))
        ratio = m["line_contrast"] / (texture / ns + 1e-9)
        border = min(x1, y1, x2, y2, w - 1 - x1, h - 1 - y1, w - 1 - x2, h - 1 - y2)
        partners = _parallel_partners((x1, y1, x2, y2), seg_arr, p) if p.veto_parallel else 0

        reasons = []
        if length < p.min_length:
            reasons.append(f"too short ({length:.0f}px < {p.min_length})")
        if m["line_contrast"] < p.min_contrast_sigma:
            reasons.append(
                f"line not distinct ({m['line_contrast']:.1f} sigma < {p.min_contrast_sigma:.0f})"
            )
        if ratio < p.min_texture_ratio:
            reasons.append(
                f"indistinct from local texture (ratio {ratio:.1f} < {p.min_texture_ratio:.0f})"
            )
        if m["side_contrast"] > p.max_side_contrast:
            reasons.append(
                f"flanks disagree (side contrast {m['side_contrast']:.2f} > "
                f"{p.max_side_contrast}) - structure, not defect"
            )
        if m["coverage"] < p.min_coverage:
            reasons.append(f"contrast fades ({m['coverage']:.0%} coverage < {p.min_coverage:.0%})")
        if m["drift"] > p.max_drift:
            reasons.append(f"curved/tapered (drift {m['drift']:.1f}px > {p.max_drift}) - hair-like")
        if max(ends) >= p.max_end_contrast:
            reasons.append(
                f"continues beyond ends ({max(ends):.1f} sigma >= {p.max_end_contrast:.1f})"
                " - structural seam, not a scratch"
            )
        if partners > 0:
            reasons.append(
                f"parallel structure beside it ({partners} companion ridge(s))"
                " - carved frame / molding, not a scratch"
            )
        if border < p.border_margin:
            reasons.append(f"touches image border ({border}px) - scan edge, not a scratch")
        accepted = not reasons
        cid += 1
        candidates.append(Candidate(
            candidate_id=cid, kind="line", accepted=accepted,
            reason="; ".join(reasons) or "thin, symmetric, even, isolated, ends on plain surface",
            area=int(round(length * thick)), length=length, width=m["width"],
            curvature=m["drift"], side_contrast=m["side_contrast"],
            line_contrast=m["line_contrast"], coverage=m["coverage"],
            bbox=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            end_contrast=ends, texture_ratio=ratio, parallel_partners=partners,
            segments=[(x1, y1, x2, y2)],
        ))
        if accepted:
            cv2.line(accepted_mask.view(np.uint8).reshape(h, w),
                     (x1, y1), (x2, y2), 1, thick)

    accepted_specks = np.zeros((h, w), bool)
    speck_count = 0
    if p.include_specks:
        hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.2))
        peak = cv2.dilate(hp, np.ones((5, 5), np.uint8))
        ismax = hp >= peak - 1e-9
        mask = ismax & (hp >= p.speck_min_sigma * ns) & ~accepted_mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 1 or area > p.speck_max_area:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if max(bw, bh) > 12:
                continue
            x0, y0 = int(stats[i, 0]), int(stats[i, 1])
            # A 1px speck's own bbox is 1x1, so crop with a margin to include
            # the surrounding ground the isolation test must judge against.
            mx = 4
            cx0, cy0 = max(0, x0 - mx), max(0, y0 - mx)
            cx1, cy1 = min(w, x0 + bw + mx), min(h, y0 + bh + mx)
            sub = hp[cy0:cy1, cx0:cx1]
            comp_c = labels[cy0:cy1, cx0:cx1] == i
            if not comp_c.any():
                continue
            ring = cv2.dilate(comp_c.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ring_px = ring & ~comp_c
            if not ring_px.any():
                continue
            peak_sig = float(sub[comp_c].mean()) / ns
            bg_sig = float(np.percentile(sub[ring_px], 90)) / ns
            comp = comp_c
            bbox = (cx0, cy0, cx0 + comp_c.shape[1] - 1, cy0 + comp_c.shape[0] - 1)
            if bg_sig > 2.5 and peak_sig / bg_sig < 3.0:
                cid += 1
                candidates.append(Candidate(
                    candidate_id=cid, kind="speck", accepted=False,
                    reason=(f"grain-like speck ({peak_sig:.0f} sigma on {bg_sig:.0f} sigma "
                            "textured ground) - not separable from texture"),
                    area=area, length=0.0, width=float(max(bw, bh)),
                    curvature=0.0, side_contrast=0.0,
                    line_contrast=peak_sig, coverage=1.0,
                    bbox=bbox,
                ))
                continue
            accepted_specks[cy0:cy1, cx0:cx1] |= comp
            speck_count += 1
            cid += 1
            candidates.append(Candidate(
                candidate_id=cid, kind="speck", accepted=True,
                reason=f"isolated {area}px speck, {peak_sig:.0f} sigma on plain ground",
                area=area, length=0.0, width=float(max(bw, bh)),
                curvature=0.0, side_contrast=0.0,
                line_contrast=peak_sig, coverage=1.0,
                bbox=bbox,
                mask=comp,
            ))

    combined = accepted_mask | accepted_specks
    cap = int(p.max_exposure_fraction * g.size)
    exposure_capped = False
    if int(combined.sum()) > cap:
        exposure_capped = True
        accepted_mask = np.zeros((h, w), bool)
        accepted_specks = np.zeros((h, w), bool)
        ranked = sorted([c for c in candidates if c.accepted],
                        key=lambda c: -(c.line_contrast - 2.0 * c.side_contrast) * max(c.length, 1.0))
        for c in ranked:
            if int((accepted_mask | accepted_specks).sum()) >= cap:
                break
            for (x1, y1, x2, y2) in c.segments:
                cv2.line(accepted_mask.view(np.uint8).reshape(h, w),
                         (x1, y1), (x2, y2), 1, thick)
            if c.kind == "speck" and c.mask.size:
                x0, y0, x1, y1 = c.bbox
                accepted_specks[max(0, y0):y1 + 1, max(0, x0):x1 + 1] |= c.mask
        combined = accepted_mask | accepted_specks

    diagnostics = {
        "sensor_noise": round(ns, 6),
        "threshold_sigma": round(thr, 2),
        "segment_proposals": len(segs),
        "strokes_accepted": sum(c.accepted for c in candidates if c.kind == "line"),
        "strokes_rejected": sum(not c.accepted for c in candidates if c.kind == "line"),
        "specks_accepted": speck_count,
        "specks_rejected": sum(not c.accepted for c in candidates if c.kind == "speck"),
        "rejection_breakdown": {
            "local_texture": sum("local texture" in c.reason for c in candidates),
            "structure": sum("structure" in c.reason for c in candidates),
            "fading": sum("fades" in c.reason for c in candidates),
            "curvature": sum("hair-like" in c.reason for c in candidates),
            "faint": sum("not distinct" in c.reason for c in candidates),
            "end_continuation": sum("beyond ends" in c.reason for c in candidates),
            "parallel_structure": sum("parallel structure" in c.reason for c in candidates),
            "border": sum("border" in c.reason for c in candidates),
            "short": sum("too short" in c.reason for c in candidates),
        },
        "selected_pixels": int(combined.sum()),
        "exposure_cap_pixels": cap,
        "exposure_capped": exposure_capped,
        "specks_enabled": bool(p.include_specks),
        "limitation": ("Thin-line detection cannot recover wide gaps; those must be "
                       "painted by hand. Dust is not separable from fine texture on these "
                       "photographs, so speck detection is off unless explicitly enabled."),
    }
    return combined, candidates, diagnostics


def evidence_overlay(img: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    """Colour every proposal by verdict so the basis is inspectable.

    amber = accepted line, red = rejected, green = accepted speck.
    ``img`` is BGR (as read by OpenCV); the constants below are BGR.
    """
    if img.ndim == 2:
        base = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 1:
        base = np.repeat(img, 3, axis=2)
    else:
        base = img
    if base.dtype != np.uint8:
        base = (np.clip(base, 0, 1) * 255).round().astype(np.uint8)
    out = base.copy()
    amber = (0, 170, 255)      # BGR for RGB amber (255,170,0)
    red = (40, 40, 230)        # BGR for RGB red (230,40,40)
    green = (60, 200, 40)      # BGR for RGB green (40,200,60)
    for c in candidates:
        if c.kind == "speck":
            if not c.mask.size:
                continue
            x0, y0, x1, y1 = c.bbox
            view = out[y0:y1 + 1, x0:x1 + 1]
            if view.shape[:2] != c.mask.shape[:2]:
                continue
            view[c.mask] = green if c.accepted else red
        else:
            color = amber if c.accepted else red
            thickness = 2 if c.accepted else 1
            for (x1, y1, x2, y2) in c.segments:
                cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out
