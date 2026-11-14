"""Repair processor: evaluate repair pixels against a frozen mask snapshot.

The processor owns pixel evaluation (zoning, reconstruction, compositing,
preservation proof) but deliberately has no handle to the revision store or the
UI selection: it accepts a :class:`~restoration.scratch.FrozenMask` and returns
a report.  Every preview and the deliverable derive from that one snapshot, so
the mask that was reviewed is provably the mask that was repaired.

The boundary (core/band/keep) and texture-protection rules are unchanged from
the legacy pipeline, and no new reconstruction algorithm is introduced here.
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
from .imageio import load_image, save_image, save_mask, to_display_png
from .maskzones import boundary_preview, build_zones
from .methods import METHODS, METHOD_TEXT, conservative_width_proxy, reconstruct, speculative_pixels
from .scratch import FrozenMask, mask_identity


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


def diff_image(original_raw: np.ndarray, composed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
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


@dataclass
class RepairResult:
    report: dict
    result_path: Path
    frozen: FrozenMask


def _difference_evidence(
    source_path: str | Path,
    result_path: str | Path,
    zones,
    amplified: np.ndarray,
) -> dict:
    """Before/after difference measured on the written files, across the edge.

    Numeric deltas come from re-reading the delivered PNG (the same files the
    preservation proof uses), so the evidence describes what was actually
    written rather than an in-memory intermediate.
    """
    src = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    out = cv2.imread(str(result_path), cv2.IMREAD_UNCHANGED)
    if src is None or out is None:
        raise RuntimeError("difference evidence could not re-read source or output")
    if src.ndim == 3 and src.shape[2] == 4:
        src = src[:, :, :3]
    if out.ndim == 3 and out.shape[2] == 4:
        out = out[:, :, :3]
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[:, :, 0]
    if out.ndim == 3 and out.shape[2] == 1:
        out = out[:, :, 0]
    delta = np.abs(src.astype(np.int64) - out.astype(np.int64))
    if delta.ndim == 3:
        per_sample = delta
        delta = delta.max(axis=2)
    else:
        per_sample = delta[:, :, None]

    def _stats(sel: np.ndarray) -> dict:
        if not sel.any():
            return {"changed_samples": 0, "mean_abs_delta": 0.0, "max_abs_delta": 0}
        return {
            "changed_samples": int((delta[sel] > 0).sum()),
            "mean_abs_delta": round(float(per_sample[sel].mean()), 4),
            "max_abs_delta": int(delta[sel].max()),
        }

    core_stats = _stats(zones.core)
    band_stats = _stats(zones.band)
    outside = ~zones.allowed
    outside_stats = _stats(outside)
    nonzero_amplified = int((amplified.sum(axis=2) > 0).sum()) if amplified.ndim == 3 else 0
    return {
        "cross_edge_changed_samples": outside_stats["changed_samples"],
        "cross_edge_max_abs_delta": outside_stats["max_abs_delta"],
        "core": core_stats,
        "band": band_stats,
        "amplified_preview_gain": 4.0,
        "amplified_diff_max": int(amplified.max()),
        "amplified_diff_nonzero_pixels": nonzero_amplified,
        "statement": (
            "Difference is measured between the delivered output and the source, "
            "re-read from disk. Samples outside core+band must show zero delta; "
            "the amplified preview only renders deltas inside the allowed zone."
        ),
    }


def repair_with_frozen_mask(
    image_path: str | Path,
    frozen: FrozenMask,
    out_dir: str | Path,
    params: RunParams,
    *,
    mask_source: str,
    compare_original: bool = True,
) -> RepairResult:
    """Run the repair against an immutable mask snapshot.

    The frozen mask is never written back to any session state; the caller's
    revision store is untouched by this function.
    """
    if params.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    mask = frozen.mask
    zones = build_zones(mask, band_px=params.band_px, min_area=params.min_area)
    rec = reconstruct(
        image.to_float(), zones.core, method=params.method,
        texture_threshold=params.texture_threshold, seed=params.seed,
        patch=params.patch, search_radius=params.search_radius,
    )
    composed, allowed = composite(image, rec.image, zones)

    ext = ".png"  # lossless: JPEG would perturb samples and break preservation
    result_path = out / f"restored{ext}"
    result_path = save_image(result_path, composed, is_gray=image.is_gray, alpha=image.alpha,
                             jpeg_quality=params.jpeg_quality)
    preserved = verify_preservation(image_path, result_path, zones)

    spec_px = speculative_pixels(zones.core, params.speculative_width)
    gap_width_floor = conservative_width_proxy(zones.core)

    # The exact reviewed snapshot, persisted under its content identity.
    frozen_mask_path = out / "mask_frozen.png"
    save_mask(frozen_mask_path, mask)

    diff_evidence = None
    if compare_original:
        diff = diff_image(image.raw, composed, allowed)
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
        to_display_png(result_path, out / "preview_restored.png")
        diff_evidence = _difference_evidence(image_path, result_path, zones, diff)

    elapsed = time.perf_counter() - t0
    core_id = mask_identity(zones.core)
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
        "mask": {
            "mask_id": frozen.mask_id,
            "core_mask_id": core_id,
            "identical_to_core": bool(np.array_equal(mask, zones.core)),
            "source_revision": frozen.source_revision,
            "based_on_proposal": frozen.based_on_proposal,
            "mask_pixels": frozen.pixel_count,
            "core_pixels": zones.core_pixels,
            "frozen_mask_export": str(frozen_mask_path),
            "mask_used": str(out / "mask_used.png") if compare_original else None,
        },
        "parameters": asdict(params),
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
        "difference": diff_evidence,
        "regions": [asdict(r) for r in rec.regions],
        "method_notes": rec.notes + [METHOD_TEXT[params.method]],
        "speculation": {
            "threshold_width_px": params.speculative_width,
            "speculative_pixels": int(spec_px.sum()),
            "gap_width_floor_px": round(gap_width_floor, 1),
            "statement": (
                "Pixels marked speculative lie inside a gap wider than "
                f"{params.speculative_width:.0f}px. Their colour is plausible, not "
                "recovered: the original emulsion is gone. Treat these areas as "
                "interpolation, not history."
            ),
        },
        "timing_s": round(elapsed, 3),
        "outputs": {
            "restored": str(result_path),
            "mask_used": str(out / "mask_used.png") if compare_original else None,
            "preview_restored": str(out / "preview_restored.png") if compare_original else None,
            "preview_difference": str(out / "preview_difference.png") if compare_original else None,
            "preview_boundary": str(out / "preview_boundary.png") if compare_original else None,
            "preview_speculative": str(out / "preview_speculative.png") if compare_original else None,
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not preserved.ok:
        raise RuntimeError(f"preservation check failed: {preserved.note}")
    return RepairResult(report=report, result_path=result_path, frozen=frozen)
