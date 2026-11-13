"""End-to-end restoration run with a machine-readable report.

Order of operations:
1. load original and mask (mask may come from detection, edited by the user);
2. build core/band/keep zones (preservation policy);
3. reconstruct only inside the core, with per-region method choice;
4. composite at the original bit depth, feathering the band;
5. write the deliverable, then re-read it and prove preservation outside the zone;
6. flag pixels in gaps wider than the honest threshold as speculative.

The report records every parameter, every region decision, timings, the
preservation proof and per-defect speculation, so the delivered file is
auditable and not just a pretty picture.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .composite import composite, verify_preservation
from .detect import DetectParams, detect_candidates, evidence_overlay
from .imageio import ImageData, load_image, load_mask, save_image, save_mask, to_display_png
from .maskzones import boundary_preview, build_zones
from .methods import METHODS, METHOD_TEXT, conservative_width_proxy, reconstruct, speculative_pixels


@dataclass
class RunParams:
    method: str = "hybrid"
    band_px: int = 2
    min_area: int = 0
    texture_threshold: float = 3.0e-4
    patch: int = 7
    search_radius: int = 40
    seed: int = 0
    speculative_width: float = 4.0
    jpeg_quality: int = 95


def run(
    image_path: str | Path,
    mask_path: str | Path | None,
    out_dir: str | Path,
    params: RunParams | None = None,
    *,
    auto_mask_params: DetectParams | None = None,
    compare_original: bool = True,
) -> dict:
    p = params or RunParams()
    if p.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    if mask_path is not None:
        mask = load_mask(mask_path, shape=image.shape)
        mask_source = str(mask_path)
    else:
        mask, _cands, _diag = detect_candidates(image.raw, auto_mask_params)
        mask_source = "detector (unreviewed - review before delivery)"
        save_mask(out / "mask_proposed.png", mask)

    zones = build_zones(mask, band_px=p.band_px, min_area=p.min_area)
    rec = reconstruct(
        image.to_float(), zones.core, method=p.method,
        texture_threshold=p.texture_threshold, seed=p.seed,
        patch=p.patch, search_radius=p.search_radius,
    )
    composed, allowed = composite(image, rec.image, zones)

    ext = ".png"  # lossless: JPEG would perturb samples and break preservation
    result_path = out / f"restored{ext}"
    result_path = save_image(result_path, composed, is_gray=image.is_gray, alpha=image.alpha,
                             jpeg_quality=p.jpeg_quality)
    preserved = verify_preservation(image_path, result_path, zones)

    spec_px = speculative_pixels(zones.core, p.speculative_width)
    gap_width_floor = conservative_width_proxy(zones.core)

    # Visual outputs for review. PNGs at display depth; never the deliverable.
    to_display_png(result_path, out / "preview_restored.png")
    if compare_original:
        diff = _diff_image(image.raw, composed, allowed)
        cv2.imwrite(str(out / "preview_difference.png"), diff)
        to_display_png(image_path, out / "preview_original.png")
        save_mask(out / "mask_used.png", zones.core)
        viz = boundary_preview(image.to_float(), zones)
        save_image(out / "preview_boundary.png",
                   (viz * 255).round().astype(np.uint8), is_gray=image.is_gray)
        spec_rgb = image.to_float()
        spec_rgb = spec_rgb if spec_rgb.ndim == 3 else np.repeat(spec_rgb[:, :, None], 3, axis=2)
        if spec_px.any():
            spec_rgb = spec_rgb.copy()
            magenta_bgr = np.array([0.9, 0.1, 1.0])
            spec_rgb[spec_px] = 0.5 * spec_rgb[spec_px] + 0.5 * magenta_bgr
        save_image(out / "preview_speculative.png",
                   (np.clip(spec_rgb, 0, 1) * 255).round().astype(np.uint8), is_gray=False)

    elapsed = time.perf_counter() - t0
    report = {
        "tool": {"name": "restoration", "version": __version__,
                 "python": platform.python_version(), "opencv": cv2.__version__},
        "inputs": {
            "image": str(image_path),
            "mask": mask_source,
            "image_shape": list(image.shape),
            "channels": image.channels,
            "dtype": str(image.raw.dtype),
        },
        "parameters": asdict(p),
        "zones": {
            "core_pixels": zones.core_pixels,
            "band_pixels": zones.band_pixels,
            "band_px_used": zones.band_px,
            "kept_pixels": int((~zones.allowed).sum()),
            "notes": zones.notes,
            "dropped_components": zones.dropped_components,
            "dropped_pixels": zones.dropped_pixels,
        },
        "preservation": asdict(preserved),
        "regions": [asdict(r) for r in rec.regions],
        "method_notes": rec.notes + [METHOD_TEXT[p.method]],
        "speculation": {
            "threshold_width_px": p.speculative_width,
            "speculative_pixels": int(spec_px.sum()),
            "gap_width_floor_px": round(gap_width_floor, 1),
            "statement": (
                "Pixels marked speculative lie inside a gap wider than "
                f"{p.speculative_width:.0f}px. Their colour is plausible, not "
                "recovered: the original emulsion is gone. Treat these areas as "
                "interpolation, not history."
            ),
        },
        "timing_s": round(elapsed, 3),
        "outputs": {
            "restored": str(result_path),
            "mask_used": str(out / "mask_used.png") if compare_original else None,
            "preview_restored": str(out / "preview_restored.png"),
            "preview_difference": str(out / "preview_difference.png") if compare_original else None,
            "preview_boundary": str(out / "preview_boundary.png") if compare_original else None,
            "preview_speculative": str(out / "preview_speculative.png") if compare_original else None,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not preserved.ok:
        raise RuntimeError(f"preservation check failed: {preserved.note}")
    return report


def _diff_image(original_raw: np.ndarray, composed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Amplified absolute difference restricted to the allowed zone."""
    a = original_raw.astype(np.float32)
    b = composed.astype(np.float32)
    if a.ndim != b.ndim:
        a = a[:, :, None] if a.ndim == 2 else a
        b = b[:, :, None] if b.ndim == 2 else b
    d = np.abs(a - b)
    if d.ndim == 2:
        d = d[:, :, None]
    d = np.clip(d * 4.0, 0, 255)
    d[~allowed] = 0
    if d.shape[2] == 1:
        d = np.repeat(d, 3, axis=2)
    return d.astype(np.uint8)


def detect_only(image_path: str | Path, out_dir: str | Path, params: DetectParams | None = None) -> dict:
    """Produce the reviewable candidate set and its evidence, without restoring."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    mask, candidates, diag = detect_candidates(image.raw, params)
    save_mask(out / "candidates_accepted.png", mask)
    overlay = evidence_overlay(image.to_float(), candidates)
    save_image(out / "candidates_evidence.png", overlay, is_gray=False)
    payload = {"params": asdict(params or DetectParams()), "diagnostics": diag,
               "candidates": [c.summary() for c in candidates]}
    (out / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
