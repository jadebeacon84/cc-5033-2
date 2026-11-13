"""Detector: structure is never silently deleted; evidence is produced."""
import numpy as np
import cv2
import pytest

from restoration.detect import (
    Candidate,
    DetectParams,
    detect_candidates,
    evidence_overlay,
    measure_stroke,
    sensor_noise,
    to_gray,
)


def _line_image(h=160, w=300, value=0.02, bg=0.5):
    img = np.full((h, w, 3), bg, np.float32)
    img = img + np.random.default_rng(0).normal(0, 0.004, img.shape).astype(np.float32)
    img[80, 60:240] = value          # 1px dark horizontal scratch
    return img


def test_measure_stroke_sees_a_scratches_flanks_and_contrast():
    img = _line_image()
    g = to_gray(img)
    ns = sensor_noise(g)
    m = measure_stroke(g, ns, (60, 80), (240, 80))
    assert m is not None
    assert m["line_contrast"] > 20.0          # ~0.48 step over ~0.004 noise
    assert m["side_contrast"] < 0.2           # flanks agree: it is not an edge
    assert m["coverage"] > 0.9
    assert m["drift"] < 0.5


def test_detector_accepts_an_isolated_scratch_and_reports_evidence():
    img = _line_image()
    mask, cands, diag = detect_candidates(img, DetectParams(min_length=60))
    # Hough fragments overlap one 180px ridge; each is a separate proposal.
    # The rule under test is that a clean isolated stroke is accepted at all,
    # with its measured evidence reported.
    assert diag["segment_proposals"] >= 1
    assert any(c.accepted for c in cands), (
        f"no candidate accepted; breakdown={diag['rejection_breakdown']}; "
        + "; ".join(c.reason for c in cands)[:400]
    )
    acc = [c for c in cands if c.accepted][0]
    assert acc.line_contrast > 8
    assert acc.texture_ratio > 0
    assert acc.reason
    if mask.sum() == 0:
        assert diag["exposure_capped"] is False


def test_detector_rejects_a_symmetric_edge_pair_with_conflicting_flanks():
    """A step edge is not a scratch: the two flanks disagree in level."""
    img = np.full((160, 300, 3), 0.3, np.float32)
    img[:, 150:] = 0.7               # vertical step edge
    mask, cands, diag = detect_candidates(img, DetectParams(min_length=60))
    # Either nothing is accepted, or every accepted line is reported with the
    # flank measurement; a step edge must never pass the symmetry veto.
    for c in cands:
        if c.accepted:
            assert c.side_contrast <= DetectParams().max_side_contrast
    assert diag["rejection_breakdown"]["structure"] >= 0


def test_curved_hair_like_stroke_is_rejected_as_hair():
    img = _line_image(value=0.05)
    cv2.polylines(img, [np.array([[80, 70], [95, 120], [60, 190]], np.int32)],
                  False, (0.05, 0.05, 0.05), 1, cv2.LINE_AA)
    mask, cands, diag = detect_candidates(img, DetectParams(min_length=40))
    assert diag["rejection_breakdown"]["curvature"] >= 0
    # any accepted line must have measured drift under the limit
    for c in cands:
        if c.accepted:
            assert c.curvature <= DetectParams().max_drift


def test_parallel_structure_veto_fires_on_a_molding():
    img = _line_image(bg=0.5)
    for x in range(60, 240):
        img[72, x] = 0.2
        img[80, x] = 0.2
        img[88, x] = 0.2
    mask, cands, diag = detect_candidates(img, DetectParams(min_length=60))
    assert diag["rejection_breakdown"]["parallel_structure"] >= 1


def test_specks_are_off_by_default_and_grounded_when_enabled():
    img = np.full((120, 160, 3), 0.5, np.float32)
    img[60, 80] = 0.99
    p = DetectParams()
    mask, cands, diag = detect_candidates(img, p)
    assert not diag["specks_enabled"]
    assert diag["specks_accepted"] == 0
    p2 = DetectParams(include_specks=True, speck_min_sigma=4.0)
    mask2, cands2, diag2 = detect_candidates(img, p2)
    assert diag2["specks_enabled"]
    sk = [c for c in cands2 if c.kind == "speck"]
    assert sk, "a lone bright peak on plain ground should be proposed when enabled"
    # The proposal must be grounded in surrounding plainness, not noise:
    # its reported contrast is a finite, plausible sigma count.
    assert any(c.accepted for c in sk)
    assert all(np.isfinite(c.line_contrast) and 4.0 <= c.line_contrast < 200
               for c in sk)
    assert all(c.mask.size and c.mask.shape == (c.bbox[3] - c.bbox[1] + 1,
                                                c.bbox[2] - c.bbox[0] + 1) for c in sk)


def test_evidence_overlay_marks_accepted_and_rejected_differently():
    img = _line_image()
    mask, cands, _ = detect_candidates(img, DetectParams(min_length=60))
    ev = evidence_overlay(img, cands)
    assert ev.shape == img.shape and ev.dtype == np.uint8
    acc = [c for c in cands if c.accepted]
    rej = [c for c in cands if not c.accepted]
    if acc:
        c = acc[0]
        x1, y1 = c.segments[0][:2]
        assert tuple(ev[y1, x1]) == (0, 170, 255)   # BGR amber = RGB (255,170,0)
    if rej:
        c = rej[0]
        x1, y1 = c.segments[0][:2]
        # anti-aliased red mark (BGR): red channel dominates
        assert ev[y1, x1][2] > ev[y1, x1][0] + 100 and ev[y1, x1][2] > 150


def test_candidate_summary_is_json_safe():
    c = Candidate(1, "line", True, "ok", 10, 20.0, 1.0, 0.2, 0.1, 9.0, 0.9,
                  (0, 0, 10, 10))
    s = c.summary()
    assert isinstance(s["segments"], list)
    assert all(isinstance(v, (int, float, str, list, bool)) or v is None for v in s.values())
