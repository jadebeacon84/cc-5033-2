"""End-to-end pipeline on a complete reference with scripted defects."""
import json
from pathlib import Path

import numpy as np
import pytest

from restoration.evaluate import DRAWERS, DefectSpec, corrupt
from restoration.imageio import load_image, save_image
from restoration.pipeline import RunParams, detect_only, run


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    src = Path(__file__).resolve().parents[1] / "fixtures" / "source" / "emir_bukhara.jpg"
    if not src.exists():
        pytest.skip("reference fixture not available")
    img = load_image(src)
    clean = img.to_float()
    # Downscale for a fast test while keeping real texture.
    small = clean[::3, ::3]
    p = tmp_path_factory.mktemp("ref") / "small.png"
    save_image(p, (np.clip(small, 0, 1) * 255).round().astype(np.uint8), is_gray=False)
    return p


def test_run_delivers_report_preservation_and_outputs(reference, tmp_path):
    img = load_image(reference)
    clean = img.to_float()
    spec = DefectSpec("scratch", count=3, width=1, value=0.02, seed=7, length=(40, 100))
    mask = DRAWERS["scratch"](clean, spec, np.random.default_rng(spec.seed))
    corrupted = corrupt(clean, mask, 0.02, np.full(clean.shape[2], 0.02, np.float32))
    inp = tmp_path / "input.png"
    save_image(inp, (corrupted * 255).round().astype(np.uint8), is_gray=False)
    mpath = tmp_path / "mask.png"
    save_image(mpath, mask.astype(np.uint8) * 255, is_gray=True)

    report = run(inp, mpath, tmp_path / "out", RunParams(band_px=2, patch=5,
                                                         search_radius=24, seed=1))
    assert report["preservation"]["ok"] is True
    assert report["preservation"]["changed_outside_allowed"] == 0
    assert Path(report["outputs"]["restored"]).exists()
    assert Path(report["outputs"]["preview_difference"]).exists()
    assert report["speculation"]["statement"]
    assert "parameters" in report
    assert report["zones"]["core_pixels"] > 0

    # The report on disk must be valid JSON with the same preservation result.
    on_disk = json.loads((tmp_path / "out" / "report.json").read_text())
    assert on_disk["preservation"]["ok"] is True
    assert on_disk["method_notes"]


def test_detect_only_writes_evidence_and_candidates(reference, tmp_path):
    payload = detect_only(reference, tmp_path / "det")
    assert (tmp_path / "det" / "candidates_evidence.png").exists()
    assert (tmp_path / "det" / "candidates_accepted.png").exists()
    assert "diagnostics" in payload and "candidates" in payload
    assert payload["diagnostics"]["limitation"]
    for c in payload["candidates"]:
        assert "reason" in c and "accepted" in c


def test_pipeline_writes_bit_identical_outside_and_reports_speculation(reference, tmp_path):
    """A wide painted gap must be delivered as speculation, not as 'recovered'."""
    img = load_image(reference)
    clean = img.to_float()
    mask = np.zeros(clean.shape[:2], bool)
    y, x = np.array(clean.shape[:2]) // 4
    mask[y:y + 24, x:x + 120] = True
    corrupted = corrupt(clean, mask, 0.5, np.full(clean.shape[2], 0.5, np.float32))
    inp = tmp_path / "wide.png"
    save_image(inp, (corrupted * 255).round().astype(np.uint8), is_gray=False)
    mpath = tmp_path / "wide_mask.png"
    save_image(mpath, mask.astype(np.uint8) * 255, is_gray=True)
    report = run(inp, mpath, tmp_path / "wide_out",
                 RunParams(band_px=2, patch=7, search_radius=30, speculative_width=12))
    assert report["preservation"]["ok"]
    assert report["speculation"]["speculative_pixels"] > 0
    assert report["speculation"]["statement"]
    st = report["speculation"]["statement"].lower()
    assert any(w in st for w in ("not", "guess", "plausib", "speculat"))
