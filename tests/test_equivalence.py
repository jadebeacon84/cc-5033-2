"""Equivalence evidence for the mask-flow decoupling.

The refactor must not change repair behaviour.  The same image/mask/params go
through the legacy :func:`restoration.pipeline.run` adapter and directly through
:func:`restoration.services.repair_with_frozen_mask`; equivalence is bounded by
mask identity (same reviewed snapshot) and output difference (byte-identical
PNGs and zero cross-edge deltas).
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from restoration.evaluate import DRAWERS, DefectSpec, corrupt
from restoration.imageio import load_image, load_mask, save_image
from restoration.pipeline import RunParams, run
from restoration.scratch import FrozenMask, mask_identity
from restoration.services import repair_with_frozen_mask
from restoration.webui_store import FileMaskRevisionStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _defect_case(tmp_path: Path):
    rng = np.random.default_rng(7)
    img = (rng.random((90, 120, 3), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
    spec = DefectSpec("scratch", count=2, width=1, value=0.05, seed=7, length=(40, 70))
    mask = DRAWERS["scratch"](img, spec, rng)
    corrupted = corrupt(img, mask, 0.05, np.full(3, 0.05, np.float32))
    inp = tmp_path / "input.png"
    mpath = tmp_path / "mask.png"
    save_image(inp, (corrupted * 255).round().astype(np.uint8), is_gray=False)
    save_image(mpath, mask.astype(np.uint8) * 255, is_gray=True)
    return inp, mpath, mask


PNG_OUTPUTS = (
    "restored.png", "mask_used.png", "preview_restored.png",
    "preview_original.png", "preview_difference.png",
    "preview_boundary.png", "preview_speculative.png",
)


def test_legacy_adapter_and_frozen_service_are_equivalent(tmp_path):
    inp, mpath, mask = _defect_case(tmp_path)
    params = RunParams(band_px=2, patch=5, search_radius=20, seed=3)

    legacy_dir = tmp_path / "legacy"
    legacy = run(inp, mpath, legacy_dir, params)

    frozen_dir = tmp_path / "frozen"
    frozen = FrozenMask.from_array(load_mask(mpath))
    result = repair_with_frozen_mask(inp, frozen, frozen_dir, params, mask_source="test")
    direct = result.report

    # 1) mask identity: both runs consumed the exact same reviewed snapshot.
    assert legacy["mask"]["mask_id"] == direct["mask"]["mask_id"] == mask_identity(mask)
    frozen_written = cv2.imread(str(frozen_dir / "mask_frozen.png"),
                                cv2.IMREAD_GRAYSCALE) > 0
    assert mask_identity(frozen_written) == frozen.mask_id

    # 2) output difference: every shared PNG is byte-identical.
    for name in PNG_OUTPUTS:
        assert _sha256(legacy_dir / name) == _sha256(frozen_dir / name), name

    # 3) cross-edge difference is zero and proven from the written files.
    for report in (legacy, direct):
        assert report["preservation"]["ok"]
        assert report["preservation"]["changed_outside_allowed"] == 0
        assert report["difference"]["cross_edge_changed_samples"] == 0
        assert report["difference"]["cross_edge_max_abs_delta"] == 0
        assert report["difference"]["core"]["changed_samples"] > 0
        assert report["difference"]["amplified_diff_nonzero_pixels"] > 0


def test_processor_does_not_mutate_the_frozen_snapshot(tmp_path):
    inp, mpath, mask = _defect_case(tmp_path)
    frozen = FrozenMask.from_array(load_mask(mpath))
    before = frozen.mask.copy()
    repair_with_frozen_mask(
        inp, frozen, tmp_path / "out", RunParams(patch=5, search_radius=16, seed=1),
        mask_source="test",
    )
    assert np.array_equal(frozen.mask, before)
    assert frozen.mask_id == mask_identity(before)


def test_file_revision_store_persists_history_and_undo_across_reopen(tmp_path):
    shape = (24, 32)
    store = FileMaskRevisionStore(tmp_path, shape)
    m1 = np.zeros(shape, bool); m1[5, 5:10] = True
    m2 = np.zeros(shape, bool); m2[8:10, 8:12] = True
    r1 = store.commit(m1, "stroke")
    store.commit(m2, "stroke")
    assert store.head_revision == 2

    reopened = FileMaskRevisionStore.open(tmp_path)
    assert reopened.head_revision == 2
    assert np.array_equal(reopened.head_mask, m2)
    reopened.undo()
    assert reopened.head_revision == 1
    assert np.array_equal(reopened.head_mask, m1)

    # mask.png always mirrors the head and matches its recorded identity.
    disk = cv2.imread(str(tmp_path / "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert np.array_equal(disk, m1)
    manifest = json.loads((tmp_path / "revisions.json").read_text())
    assert manifest["revisions"][1]["mask_id"] == r1.mask_id


def test_report_records_mask_identity_linked_to_preview_assets(tmp_path):
    inp, mpath, mask = _defect_case(tmp_path)
    report = run(inp, mpath, tmp_path / "out", RunParams(patch=5, search_radius=16))
    mask_used = cv2.imread(str(report["outputs"]["mask_used"]),
                           cv2.IMREAD_GRAYSCALE) > 0
    assert report["mask"]["mask_id"] == mask_identity(mask)
    assert report["mask"]["core_mask_id"] == mask_identity(mask_used)
    assert report["mask"]["identical_to_core"] is True
    assert Path(report["mask"]["frozen_mask_export"]).exists()
    # every declared output path exists; the evidence is not pointing at phantoms
    for key, value in report["outputs"].items():
        assert value is None or Path(value).exists(), key
