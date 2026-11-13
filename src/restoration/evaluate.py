"""Evaluation: inject known defects into complete references and measure repair.

A clean reference image stands in for the "complete" picture.  We draw known
scratches (thin lines), dust specks and gaps of several widths on top, record
the exact mask that was corrupted, and then run the restoration.  Because the
pre-defect image is known, error is measured against truth as well as against
the surrounding continuity.

This deliberately does **not** substitute a hand-touched single image for the
algorithm: every scripted defect is applied programmatically and every
measurement is computed from arrays.

Metrics per case:
* ``mask_iou`` / ``mask_recall`` - did the detector's proposal cover the intended
  defect? (reported in the end-to-end path)
* ``leak_pixels`` - band pixels outside the true defect (documented transition).
* ``mae`` / ``psnr`` inside the true defect - reconstruction error vs truth.
* ``edge_continuity`` - difference in boundary gradient between result and truth.
* ``preservation`` - re-read output vs source outside the allowed zone.

Two tracks are reported so the algorithm and the detector are not conflated:
* ``mask=truth`` scores the reconstruction method given a perfect mask;
* ``mask=detector`` scores the whole pipeline with the detector's proposal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .composite import composite, verify_preservation
from .detect import DetectParams, detect_candidates
from .imageio import ImageData, load_image, save_image
from .maskzones import build_zones
from .methods import reconstruct


@dataclass
class DefectSpec:
    kind: str                # "scratch" | "dust" | "gap"
    count: int
    width: int
    value: float = 1.0       # defect intensity in [0,1] before quantization
    seed: int = 0
    length: tuple[int, int] = (60, 260)
    curved: bool = False


def _draw_line(img: np.ndarray, spec: DefectSpec, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for _ in range(spec.count):
        cy = int(rng.integers(h * 0.1, h * 0.9))
        cx = int(rng.integers(w * 0.1, w * 0.9))
        ang = float(rng.uniform(0, np.pi))
        ln = int(rng.integers(spec.length[0], spec.length[1]))
        x2 = int(cx + ln * np.cos(ang))
        y2 = int(cy + ln * np.sin(ang))
        if spec.curved:
            ctrl_x = (cx + x2) // 2 + int(rng.integers(-ln // 3, ln // 3))
            ctrl_y = (cy + y2) // 2 + int(rng.integers(-ln // 3, ln // 3))
            pts = np.array([[cx, cy], [ctrl_x, ctrl_y], [x2, y2]], np.int32)
            cv2.polylines(mask, [pts], False, 1, spec.width, cv2.LINE_AA)
        else:
            cv2.line(mask, (cx, cy), (x2, y2), 1, spec.width, cv2.LINE_AA)
    return mask > 0.4


def _draw_dust(img: np.ndarray, spec: DefectSpec, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for _ in range(spec.count):
        cy = int(rng.integers(4, h - 4))
        cx = int(rng.integers(4, w - 4))
        r = int(rng.integers(1, max(2, spec.width)))
        cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask > 0


def _draw_gap(img: np.ndarray, spec: DefectSpec, rng: np.random.Generator) -> np.ndarray:
    """Gaps of the requested width, with both ends touching image content."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for _ in range(spec.count):
        cy = int(rng.integers(h * 0.2, h * 0.8))
        ln = int(rng.integers(max(spec.length[0], 40), spec.length[1]))
        cx = int(rng.integers(0, max(1, w - ln)))
        cv2.rectangle(mask, (cx, cy - spec.width // 2), (cx + ln, cy + spec.width // 2), 1, -1)
    return mask > 0


DRAWERS = {"scratch": _draw_line, "dust": _draw_dust, "gap": _draw_gap}


def corrupt(img_float: np.ndarray, mask: np.ndarray, value: float, tint: np.ndarray | None = None) -> np.ndarray:
    out = img_float.copy()
    val = tint if tint is not None else np.array([value], np.float32)
    if out.ndim == 2:
        out[mask] = np.clip(value, 0, 1)
    else:
        out[mask] = np.clip(val, 0, 1)
    return out


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, peak: float = 1.0) -> float:
    sel = mask if mask is not None else np.ones(a.shape[:2], bool)
    diff = (a - b)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    mse = float((diff[sel] ** 2).mean()) if sel.any() else 0.0
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(peak ** 2 / mse))


def edge_continuity(repaired: np.ndarray, truth: np.ndarray, mask: np.ndarray, band: np.ndarray) -> float:
    """Mean gradient-magnitude difference on the defect boundary (lower = smoother)."""
    g1 = cv2.Laplacian(repaired if repaired.ndim == 2 else repaired.mean(axis=2), cv2.CV_32F, ksize=3)
    g2 = cv2.Laplacian(truth if truth.ndim == 2 else truth.mean(axis=2), cv2.CV_32F, ksize=3)
    boundary = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~mask
    if not boundary.any():
        return 0.0
    return float(np.abs(np.abs(g1[boundary]) - np.abs(g2[boundary])).mean())


def evaluate_case(
    clean_float: np.ndarray,
    defect_mask: np.ndarray,
    corrupted_float: np.ndarray,
    *,
    method: str = "hybrid",
    band_px: int = 2,
    seed: int = 0,
    pipeline_mask: np.ndarray | None = None,
    mask_source: str = "truth",
) -> dict:
    """Run restoration in memory and score it against the clean truth.

    ``pipeline_mask`` is what the pipeline is allowed to fill; defaulting to the
    true defect mask scores the reconstruction method in isolation.  Passing the
    detector's output scores the whole pipeline, and detection IoU/recall is
    reported against ``defect_mask``.
    """
    used_mask = defect_mask if pipeline_mask is None else pipeline_mask
    zones = build_zones(used_mask, band_px=band_px, min_area=0)
    rec = reconstruct(corrupted_float, zones.core, method=method, seed=seed)
    orig_f = corrupted_float[:, :, None] if corrupted_float.ndim == 2 else corrupted_float
    rec_f = rec.image[:, :, None] if rec.image.ndim == 2 else rec.image
    feather = zones.feather[:, :, None]
    out = orig_f * (1 - feather) + rec_f * feather

    truth_f = clean_float[:, :, None] if clean_float.ndim == 2 else clean_float
    err_mask = defect_mask
    mae = float(np.abs(out[err_mask] - truth_f[err_mask]).mean()) if err_mask.any() else 0.0
    p = psnr(out, truth_f, err_mask)
    touched_outside_defect = zones.allowed & ~defect_mask
    leak_pixels = int(touched_outside_defect.sum())
    ec = edge_continuity(out, truth_f, defect_mask, zones.band)

    inter = int((used_mask & defect_mask).sum())
    union = int((used_mask | defect_mask).sum())
    iou = inter / union if union else 1.0
    recall = inter / int(defect_mask.sum()) if defect_mask.any() else 1.0
    precision = inter / int(used_mask.sum()) if used_mask.any() else 1.0
    # Pixels the detector selected that were *not* defects: this is where
    # false positives (building edges, hair) would show up as needless edits.
    false_positive = int((used_mask & ~defect_mask).sum())

    # Samples outside the permitted zone must be bit-identical.  Compare the
    # float composite rounded to 8-bit with the 8-bit corrupted input.
    out8 = np.rint(np.clip(out, 0, 1) * 255).astype(np.int32)
    src8 = np.rint(np.clip(orig_f, 0, 1) * 255).astype(np.int32)
    if out8.ndim == 3:
        out8 = out8.max(axis=2)
    if src8.ndim == 3:
        src8 = src8.max(axis=2)
    outside_changed = int((np.abs(out8 - src8)[~zones.allowed] > 0).sum())

    return {
        "method": method,
        "mask_source": mask_source,
        "mask_iou": round(iou, 4),
        "mask_recall": round(recall, 4),
        "mask_precision": round(precision, 4),
        "false_positive_pixels": false_positive,
        "outside_changed_samples": outside_changed,
        "band_px": zones.band_px,
        "core_pixels": zones.core_pixels,
        "band_pixels": zones.band_pixels,
        "mae_in_defect": round(mae, 5),
        "psnr_in_defect_db": round(p, 2),
        "edge_continuity": round(ec, 4),
        "leak_pixels_in_band": leak_pixels,
        "regions": [
            {"id": r.region_id, "method": r.method, "texture_energy": round(r.texture_energy, 5),
             "area": r.area, "bbox": r.bbox}
            for r in rec.regions
        ],
        "notes": rec.notes,
    }


def run_reference_suite(
    reference: Path,
    out_dir: Path,
    *,
    methods: tuple[str, ...] = ("hybrid", "exemplar", "smooth"),
    band_px: int = 2,
) -> dict:
    """Inject scripted defects into a complete reference and score each method."""
    img = load_image(reference)
    clean = img.to_float()
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = clean.shape[:2]
    results: list[dict] = []

    cases = [
        ("thin_dark_scratches", DefectSpec("scratch", count=6, width=1, value=0.02, seed=11,
                                           length=(120, 320))),
        ("thin_bright_scratches", DefectSpec("scratch", count=5, width=1, value=0.98, seed=12,
                                             length=(100, 280))),
        ("hair_like_curved", DefectSpec("scratch", count=4, width=1, value=0.05, seed=13,
                                        length=(90, 200), curved=True)),
        ("dust_specks", DefectSpec("dust", count=90, width=3, value=0.95, seed=14)),
        ("gap_4px", DefectSpec("gap", count=2, width=4, value=0.5, seed=15, length=(80, 180))),
        ("gap_10px", DefectSpec("gap", count=1, width=10, value=0.5, seed=16, length=(100, 200))),
        ("gap_24px", DefectSpec("gap", count=1, width=24, value=0.5, seed=17, length=(120, 240))),
    ]

    for name, spec in cases:
        rng = np.random.default_rng(spec.seed)
        true_mask = DRAWERS[spec.kind](clean, spec, rng)
        # Scratch/dust colours: mid-grey gaps simulate missing emulsion with no
        # inherent colour; scratches use near-black/near-white.
        if spec.kind == "gap":
            tint = np.full(clean.shape[2], spec.value, np.float32) if clean.ndim == 3 else None
        elif clean.ndim == 3:
            tint = np.full(clean.shape[2], spec.value, np.float32)
        else:
            tint = None
        corrupted = corrupt(clean, true_mask, spec.value, tint)

        # Ground-truth PNGs for inspection: the corrupted input, its exact mask.
        save_image(out_dir / f"{name}__input.png",
                   (corrupted * 255).round().astype(np.uint8) if corrupted.dtype != np.uint8
                   else corrupted, is_gray=(corrupted.ndim == 2))
        save_image(out_dir / f"{name}__mask.png",
                   true_mask.astype(np.uint8) * 255, is_gray=True)

        for method in methods:
            metrics = evaluate_case(clean, true_mask, corrupted, method=method,
                                    band_px=band_px, mask_source="truth")
            metrics.update({"case": name, "kind": spec.kind, "width": spec.width,
                            "count": spec.count, "reference": str(reference)})
            results.append(metrics)

        # End-to-end: detector proposes the mask after seeing the corrupted
        # image, then the default hybrid repair runs on that proposal.
        detected, _cands, diag = detect_candidates(corrupted, DetectParams())
        save_image(out_dir / f"{name}__detected_mask.png",
                   detected.astype(np.uint8) * 255, is_gray=True)
        e2e = evaluate_case(clean, true_mask, corrupted, method="hybrid",
                            band_px=band_px, pipeline_mask=detected, mask_source="detector")
        e2e.update({"case": name, "kind": spec.kind, "width": spec.width,
                    "count": spec.count, "reference": str(reference),
                    "detector_diagnostics": diag})
        results.append(e2e)

    summary = {"reference": str(reference), "shape": [h, w],
               "cases": sorted({r["case"] for r in results}), "results": results}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
