"""Preservation is proven by re-reading the written file, not asserted."""
import numpy as np
import pytest

from restoration.composite import composite, verify_preservation
from restoration.imageio import ImageData, save_image
from restoration.maskzones import build_zones


def _img(arr, is_gray=False):
    return ImageData(raw=arr, alpha=None, is_gray=is_gray, path=None)


def test_composite_leaves_outside_bit_exact():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    img = _img(raw)
    mask = np.zeros((40, 50), bool)
    mask[15:25, 20:30] = True
    zones = build_zones(mask, band_px=2)
    rec = np.full_like(raw, 200, dtype=np.float32)
    composed, allowed = composite(img, rec, zones)
    assert np.array_equal(composed[~allowed], raw[~allowed])


def test_verify_preservation_detects_outside_edit(tmp_path):
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    src = save_image(tmp_path / "src.png", raw, is_gray=False)
    mask = np.zeros((30, 40), bool)
    mask[10:20, 10:20] = True
    zones = build_zones(mask, band_px=1)

    clean = raw.copy()
    clean[mask] = 250
    dst_ok = save_image(tmp_path / "ok.png", clean, is_gray=False)
    ok = verify_preservation(src, dst_ok, zones)
    assert ok.ok
    assert ok.changed_outside_allowed == 0

    bad = clean.copy()
    bad[0, 0] = 0 if bad[0, 0, 0] != 0 else 255
    dst_bad = save_image(tmp_path / "bad.png", bad, is_gray=False)
    res = verify_preservation(src, dst_bad, zones)
    assert not res.ok
    assert res.changed_outside_allowed >= 1
    assert res.max_abs_delta_outside >= 1


def test_verify_preservation_gray_roundtrip(tmp_path):
    raw = np.tile(np.arange(40, dtype=np.uint8), (30, 1))
    src = save_image(tmp_path / "s.png", raw, is_gray=True)
    mask = np.zeros((30, 40), bool)
    mask[5:10, 5:10] = True
    zones = build_zones(mask, band_px=1)
    out = raw.copy()
    out[mask] = 128
    dst = save_image(tmp_path / "d.png", out, is_gray=True)
    res = verify_preservation(src, dst, zones)
    assert res.ok
