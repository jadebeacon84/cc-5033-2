"""Small I/O and evaluation-metric contracts."""
import numpy as np
import pytest

from restoration.evaluate import (
    DefectSpec,
    DRAWERS,
    corrupt,
    edge_continuity,
    evaluate_case,
    psnr,
)


def test_load_mask_thresholds_and_resizes(tmp_path):
    from restoration.imageio import load_mask, save_image

    m = np.zeros((20, 30), np.uint8)
    m[5:15, 10:20] = 255
    p = save_image(tmp_path / "m.png", m, is_gray=True)
    loaded = load_mask(p, shape=(20, 30))
    assert loaded.dtype == bool
    assert loaded[10, 15]
    assert np.array_equal(loaded, m > 0)


def test_corrupt_only_touches_mask():
    rng = np.random.default_rng(0)
    img = rng.random((30, 40, 3), dtype=np.float32)
    mask = np.zeros((30, 40), bool)
    mask[10:20, 10:20] = True
    out = corrupt(img, mask, 0.0)
    assert np.array_equal(out[~mask], img[~mask])
    assert np.all(out[mask] == 0)


def test_drawers_create_expected_shapes_and_widths():
    img = np.zeros((120, 200, 3), np.float32)
    rng = np.random.default_rng(1)
    s = DefectSpec("scratch", 2, 1, 0.0, 4, (60, 100))
    m = DRAWERS["scratch"](img, s, rng)
    assert m.dtype == bool and m.sum() > 0
    g = DefectSpec("gap", 1, 10, 0.5, 5, (60, 100))
    gm = DRAWERS["gap"](img, g, rng)
    # a 10px band: column-wise fill is at least 8 px thick in the interior
    cols = np.nonzero(gm.any(axis=0))[0]
    assert cols.size > 0
    thickness = np.median(gm[:, cols].sum(axis=0))
    assert 8 <= thickness <= 12


def test_psnr_is_high_for_identical_and_low_for_noise():
    a = np.zeros((20, 20), np.float32)
    assert psnr(a, a) >= 99
    b = np.full((20, 20), 0.5, np.float32)
    assert psnr(a, b) < 10


def test_evaluate_case_reports_scope_and_preservation():
    rng = np.random.default_rng(2)
    clean = rng.random((60, 80, 3), dtype=np.float32)
    mask = np.zeros((60, 80), bool)
    mask[20:40, 20:50] = True
    corrupted = clean.copy()
    corrupted[mask] = 0.0

    truth = evaluate_case(clean, mask, corrupted, method="hybrid", band_px=2)
    assert truth["outside_changed_samples"] == 0
    assert truth["mask_recall"] == 1.0 and truth["mask_iou"] == 1.0
    assert truth["mae_in_defect"] >= 0
    assert truth["edge_continuity"] >= 0

    # A detector proposal that misses half the defect must show reduced recall
    # and non-zero false positives only where it selects non-defect pixels.
    proposal = mask.copy()
    proposal[20:40, 20:35] = False
    proposal[45:55, 55:75] = True
    e2e = evaluate_case(clean, mask, corrupted, method="hybrid",
                        pipeline_mask=proposal, mask_source="detector")
    assert 0.0 < e2e["mask_recall"] < 1.0
    assert e2e["false_positive_pixels"] > 0
    assert e2e["mask_source"] == "detector"


def test_edge_continuity_zero_when_repaired_matches_truth():
    rng = np.random.default_rng(3)
    clean = rng.random((40, 40), dtype=np.float32)
    mask = np.zeros((40, 40), bool)
    mask[15:20, 15:20] = True
    band = np.zeros_like(mask)
    assert edge_continuity(clean, clean, mask, band) == 0.0
