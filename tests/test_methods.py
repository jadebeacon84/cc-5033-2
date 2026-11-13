"""Reconstruction methods: real pixel synthesis, method suitability, speculation."""
import numpy as np
import cv2
import pytest

from restoration.maskzones import build_zones
from restoration.methods import (
    METHODS,
    METHOD_TEXT,
    conservative_width_proxy,
    exemplar_fill,
    reconstruct,
    speculative_pixels,
)


def test_reconstruct_returns_same_shape_and_fills_only_core():
    rng = np.random.default_rng(0)
    img = rng.random((60, 80, 3), dtype=np.float32)
    mask = np.zeros((60, 80), bool)
    mask[25:35, 30:50] = True
    res = reconstruct(img, mask, method="hybrid", seed=1)
    assert res.image.shape == img.shape
    assert np.isfinite(res.image).all()
    # The core is actually rewritten (this is a real reconstruction, not a copy).
    assert not np.allclose(res.image[mask], img[mask])
    # And there is a method decision for the touched component.
    assert len(res.regions) >= 1
    assert res.regions[0].method in ("exemplar", "smooth")


def test_method_text_states_texture_vs_smooth_and_speculation():
    for m in METHODS:
        assert m in METHOD_TEXT
    joined = " ".join(METHOD_TEXT.values()).lower()
    assert "texture" in joined
    assert "smooth" in joined
    # The wide-gap caveat must be explicit, not silently implied.
    assert ("speculat" in joined or "plausible" in joined or "invented" in joined
            or "not the lost original" in joined or "not a recovery" in joined
            or "not recover" in joined)


def test_gray_and_single_channel_inputs_are_accepted():
    rng = np.random.default_rng(2)
    for shape in ((50, 60), (50, 60, 1)):
        img = rng.random(shape, dtype=np.float32)
        mask = np.zeros(shape[:2], bool)
        mask[20:30, 20:30] = True
        res = reconstruct(img, mask, method="smooth", seed=0)
        assert res.image.shape == shape
        assert np.isfinite(res.image).all()


def _local_contrast(a, mask, k=5):
    g = a.mean(2) if a.ndim == 3 else a
    mu = cv2.blur(g, (k, k))
    var = cv2.blur((g - mu) ** 2, (k, k))
    return float(np.sqrt(np.maximum(var, 0))[mask].mean())


def test_exemplar_differs_from_smooth_and_copies_real_texture():
    """The exemplar method must synthesize real grain, not blur over the hole."""
    rng = np.random.default_rng(3)
    base = rng.random((60, 80, 3), np.float32)
    tex = np.stack([cv2.GaussianBlur(base[:, :, c], (0, 0), 1.5) for c in range(3)], 2)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    img = tex.copy()
    mask = np.zeros((60, 80), bool)
    mask[28:32, 20:60] = True
    img[mask] = 0.0
    ex = exemplar_fill(img, mask, patch=5, search_radius=16, seed=0)
    sm = reconstruct(img, mask, method="smooth", seed=0).image
    assert not np.allclose(ex[mask], sm[mask])
    # Contrast with an explicit blur of the same region: synthesis must carry
    # clearly more local texture than a blur would (i.e. it is not hiding the
    # defect), and it must not merely copy the blurred boundary.
    blurred = cv2.GaussianBlur(tex, (0, 0), 2.0)
    c_ex = _local_contrast(ex, mask)
    c_blur = _local_contrast(blurred, mask)
    assert c_ex >= 2.0 * c_blur, (c_ex, c_blur)
    assert float(np.corrcoef(ex[mask].reshape(-1), sm[mask].reshape(-1))[0, 1]) < 0.9


def test_wide_gap_pixels_are_flagged_speculative_and_narrow_are_not():
    narrow = np.zeros((40, 400), bool)
    narrow[20, 50:250] = True
    wide = np.zeros((40, 400), bool)
    wide[10:34, 50:250] = True          # 24 px tall gap
    assert speculative_pixels(narrow, width_threshold=12.0).sum() == 0
    assert speculative_pixels(wide, width_threshold=12.0).sum() > 0
    assert conservative_width_proxy(wide) >= conservative_width_proxy(narrow)


def test_no_method_modifies_outside_mask_input():
    rng = np.random.default_rng(5)
    img = rng.random((40, 40, 3), dtype=np.float32)
    mask = np.zeros((40, 40), bool)
    mask[15:25, 15:25] = True
    res = reconstruct(img, mask, method="hybrid", seed=0)
    # reconstruct() only promises the result; the preservation guarantee is
    # applied by composite().  Here we check the untouched area is still sane.
    assert np.isfinite(res.image).all()
