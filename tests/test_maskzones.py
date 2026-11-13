"""Mask zoning and the preservation contract."""
import numpy as np
import pytest

from restoration.maskzones import build_zones


def test_core_matches_mask_and_band_is_bounded():
    mask = np.zeros((40, 60), bool)
    mask[10:20, 10:30] = True
    z = build_zones(mask, band_px=2)
    assert np.array_equal(z.core, mask)
    assert z.band_px == 2
    assert z.band.any()
    # band is entirely outside the core
    assert not (z.band & z.core).any()
    # nothing outside core|band is ever allowed
    allowed = z.core | z.band
    assert np.array_equal(z.allowed, allowed)


def test_thin_scratch_keeps_its_band():
    """Regression: the band cap was once relative to the core, zeroing it."""
    mask = np.zeros((200, 200), bool)
    mask[100, 20:180] = True          # a 1px scratch
    z = build_zones(mask, band_px=2)
    assert z.band_px == 2
    assert z.band.sum() > 0
    assert not (z.band & z.core).any()


def test_feather_is_one_in_core_and_zero_outside_band():
    mask = np.zeros((50, 50), bool)
    mask[20:30, 20:30] = True
    z = build_zones(mask, band_px=3)
    assert np.all(z.feather[z.core] == 1.0)
    assert np.all(z.feather[~z.allowed] == 0.0)
    band_vals = z.feather[z.band]
    assert band_vals.min() > 0.0 and band_vals.max() < 1.0


def test_empty_mask_allows_nothing():
    z = build_zones(np.zeros((20, 20), bool), band_px=2)
    assert z.core_pixels == 0
    assert z.allowed.sum() == 0
    assert z.feather.sum() == 0


def test_dropped_tiny_components_are_reported():
    mask = np.zeros((60, 60), bool)
    mask[5, 5] = True                 # 1 px
    mask[30:40, 30:40] = True
    z = build_zones(mask, band_px=1, min_area=10)
    assert z.dropped_components >= 1
    assert not z.core[5, 5]
    assert z.core[35, 35]
    assert z.notes


def test_boundary_preview_uses_bgr_constants():
    """Core must be blue and band amber in the stored (BGR) array."""
    from restoration.maskzones import boundary_preview

    img = np.full((40, 60, 3), 0.5, np.float32)
    mask = np.zeros((40, 60), bool)
    mask[15:25, 20:40] = True
    z = build_zones(mask, band_px=3)
    viz = boundary_preview(img, z)
    core_bgr = viz[z.core][0]
    band_bgr = viz[z.band][0]
    assert core_bgr[0] > core_bgr[1] > core_bgr[2]          # blue
    assert band_bgr[0] < band_bgr[1] < band_bgr[2]          # amber/orange
    # untouched pixels are still the original 0.5 grey
    keep = viz[~z.allowed][0]
    assert np.allclose(keep, 0.5)
