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

Compatibility note: :func:`run` is the legacy entry point and now acts as an
adapter over the versioned scratch workflow.  It freezes the supplied mask into
a :class:`~restoration.scratch.FrozenMask` and delegates pixel evaluation to
:func:`~restoration.services.repair_with_frozen_mask`, which cannot reach back
into any UI selection.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .detect import DetectParams, detect_candidates, evidence_overlay
from .imageio import load_image, load_mask, save_image, save_mask
from .maskzones import build_zones  # re-exported for legacy callers/tests
from .methods import METHODS
from .scratch import FrozenMask
from .services import RunParams, diff_image, repair_with_frozen_mask

__all__ = ["RunParams", "run", "detect_only", "build_zones", "diff_image"]


def run(
    image_path: str | Path,
    mask_path: str | Path | None,
    out_dir: str | Path,
    params: RunParams | None = None,
    *,
    auto_mask_params: DetectParams | None = None,
    compare_original: bool = True,
) -> dict:
    """Legacy entry: load/derive a mask, freeze it, then repair.

    Kept byte-compatible for existing callers; the mask is frozen before the
    processor runs, so detection or later edits cannot change what is repaired.
    """
    p = params or RunParams()
    if p.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    if mask_path is not None:
        mask = load_mask(mask_path, shape=image.shape)
        mask_source = str(mask_path)
        source_revision = 0
        based_on_proposal = None
    else:
        mask, _cands, _diag = detect_candidates(image.raw, auto_mask_params)
        mask_source = "detector (unreviewed - review before delivery)"
        save_mask(out / "mask_proposed.png", mask)
        source_revision = 0
        based_on_proposal = None

    frozen = FrozenMask.from_array(
        mask, source_revision=source_revision, based_on_proposal=based_on_proposal
    )
    result = repair_with_frozen_mask(
        image_path, frozen, out, p,
        mask_source=mask_source, compare_original=compare_original,
    )
    return result.report


def _diff_image(original_raw: np.ndarray, composed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Legacy alias for :func:`restoration.services.diff_image`."""
    return diff_image(original_raw, composed, allowed)


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
