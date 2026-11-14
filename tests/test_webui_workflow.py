"""End-to-end checks for the graphical scratch-repair workflow.

The browser-facing app must keep the scratch-repair operation inside the image
correction UI: upload, paint a mask, repair, then view the real original and
real repair details.  Results always come from a live run - no fixed stand-in
data - and the processor's frozen mask is the one shown and exported.
"""
import json

import cv2
import numpy as np
import pytest

from restoration.imageio import save_image
from restoration.scratch import mask_identity
from restoration.webui import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path / "ui")
    app.config.update(TESTING=True)
    return app.test_client()


def _upload(client, tmp_path, h=120, w=160):
    rng = np.random.default_rng(11)
    img = (rng.random((h, w, 3), dtype=np.float32) * 0.5 + 0.25)
    # a visible dark scratch so there is something to repair
    img[60, 30:130] = 0.03
    p = tmp_path / "photo.png"
    save_image(p, (img * 255).round().astype(np.uint8), is_gray=False)
    data = {"image": (p.open("rb"), "photo.png")}
    res = client.post("/api/sessions", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    return res.get_json()["id"]


def _stroke(sid, client, *, mode="add", radius=3):
    body = {"strokes": [{
        "points": [[30, 60], [60, 60], [100, 60], [130, 60]],
        "radius": radius, "mode": mode,
    }]}
    res = client.post(f"/api/sessions/{sid}/mask", json=body)
    assert res.status_code == 200
    return res.get_json()


def _repair_body():
    return {
        "method": "hybrid", "band_px": 2, "texture_threshold": 0.0003,
        "patch": 5, "search_radius": 20, "seed": 1, "speculative_width": 4.0,
    }


def test_full_graphical_workflow_paint_undo_repair_and_real_media(client, tmp_path):
    sid = _upload(client, tmp_path)

    status = _stroke(sid, client)
    assert status["mask_pixels"] > 0 and status["revision"] == 1
    assert status["mask_id"] and status["can_undo"] is True

    # undo returns to the empty initial revision and rewrites mask.png
    undone = client.post(f"/api/sessions/{sid}/mask/undo").get_json()
    assert undone["revision"] == 0 and undone["mask_pixels"] == 0
    assert undone["can_undo"] is False

    # repaint, then repair
    status = _stroke(sid, client)
    assert status["revision"] == 1
    res = client.post(f"/api/sessions/{sid}/restore", json=_repair_body())
    assert res.status_code == 200, res.get_data(as_text=True)
    rep = res.get_json()

    # The run consumed the reviewed snapshot and reports its identity.
    assert rep["mask"]["mask_id"] == status["mask_id"]
    assert rep["mask"]["source_revision"] == 1
    assert rep["difference"]["cross_edge_changed_samples"] == 0
    assert rep["preservation"]["ok"] is True

    # Every media URL resolves and the repaired image is a real PNG, not a stub.
    for key, url in rep["media"].items():
        media = client.get(url)
        assert media.status_code == 200, (key, url)
        assert media.data[:8] == b"\x89PNG\r\n\x1a\n", key

    restored = cv2.imdecode(
        np.frombuffer(client.get(rep["media"]["restored"]).data, np.uint8),
        cv2.IMREAD_COLOR,
    )
    original = cv2.imdecode(
        np.frombuffer(client.get(rep["media"]["original"]).data, np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert restored is not None and original is not None
    # Repair changed pixels inside the scratch and kept the original elsewhere.
    assert not np.array_equal(restored, original)
    outside_changed = rep["preservation"]["changed_outside_allowed"]
    assert outside_changed == 0

    # Frozen-mask export is byte-consistent with the reviewed mask identity.
    frozen_bytes = client.get(rep["media"]["frozen_mask"]).data
    frozen = cv2.imdecode(np.frombuffer(frozen_bytes, np.uint8), cv2.IMREAD_GRAYSCALE) > 0
    assert mask_identity(frozen) == rep["mask"]["mask_id"]


def test_repair_refuses_empty_mask_instead_of_silently_succeeding(client, tmp_path):
    sid = _upload(client, tmp_path)
    res = client.post(f"/api/sessions/{sid}/restore", json=_repair_body())
    assert res.status_code == 400
    assert "empty" in res.get_json()["error"]


def test_repair_rejects_missing_parameters_without_defaulting(client, tmp_path):
    sid = _upload(client, tmp_path)
    _stroke(sid, client)
    body = _repair_body()
    del body["seed"]
    res = client.post(f"/api/sessions/{sid}/restore", json=body)
    assert res.status_code == 400
    assert "seed" in res.get_json()["error"]


def test_painting_after_repair_does_not_change_the_frozen_run(client, tmp_path):
    sid = _upload(client, tmp_path)
    status = _stroke(sid, client)
    rep = client.post(f"/api/sessions/{sid}/restore", json=_repair_body()).get_json()
    run_mask_id = rep["mask"]["mask_id"]

    # Further edits move the head forward...
    more = _stroke(sid, client, radius=5)
    assert more["revision"] == 2 and more["mask_id"] != run_mask_id
    # ...but the previous run's frozen snapshot and report stay on revision 1.
    report = client.get(f"/api/sessions/{sid}/report").get_json()
    assert report["mask"]["mask_id"] == run_mask_id
    frozen = cv2.imdecode(
        np.frombuffer(client.get(rep["media"]["frozen_mask"]).data, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ) > 0
    assert mask_identity(frozen) == run_mask_id


def test_detect_then_apply_candidates_links_proposal_identity(client, tmp_path):
    sid = _upload(client, tmp_path)
    det = client.post(f"/api/sessions/{sid}/detect", json={})
    assert det.status_code == 200
    proposal_id = det.get_json()["proposal_id"]
    applied = client.post(f"/api/sessions/{sid}/mask/apply_candidates")
    if applied.status_code == 400:
        pytest.skip("detector found no candidates on this small image")
    payload = applied.get_json()
    assert payload["revision"] == 1 and payload["mask_pixels"] >= 0
    # The revision's proposal link is visible in the session directory.
    manifest = json.loads((tmp_path / "ui" / sid / "revisions.json").read_text())
    assert manifest["revisions"][1]["based_on_proposal"] == proposal_id


def test_unknown_session_and_bad_media_paths_are_rejected(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.get("/media/nope/mask.png").status_code == 404
