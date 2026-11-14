"""Local review UI: load an image, review detection evidence, paint/erase a mask,
run the repair, and compare original / mask / result / difference side by side.

The UI owns the *reviewed selection* through a versioned
:class:`~restoration.scratch.MaskRevisionStore` (with undo).  Running a repair
freezes the head revision into a :class:`~restoration.scratch.FrozenMask`; the
repair processor receives only that snapshot and can never modify the working
selection.  Every preview and download of a run derives from the frozen mask.

The UI never heuristically erases anything on its own: painting is a deliberate
user action, and every rendered result comes from a real repair run on the
uploaded image - there are no fixed stand-in pixels.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from .detect import DetectParams, evidence_overlay
from .imageio import to_display_png
from .pipeline import RunParams
from .scratch import CandidateProposal, mask_identity
from .services import repair_with_frozen_mask
from .webui_store import FileMaskRevisionStore


class ScratchWorkflowError(RuntimeError):
    """A required scratch-workflow input is missing or inconsistent (HTTP 400)."""


def _session_dir(root: Path, sid: str) -> Path:
    d = root / sid
    if not (d / "session.json").exists():
        raise FileNotFoundError(f"unknown session: {sid}")
    return d


def create_app(out_dir: Path | str = Path("out/ui")) -> Flask:
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
    app.register_error_handler(ScratchWorkflowError, lambda e: (jsonify({"error": str(e)}), 400))

    # ---------------------------------------------------------------- helpers
    def load_session(sid: str) -> dict:
        d = _session_dir(root, sid)
        return json.loads((d / "session.json").read_text(encoding="utf-8"))

    def save_session(sid: str, meta: dict) -> None:
        # Bootstrap-safe: the first save creates the session record, so it
        # cannot go through _session_dir (which requires one to exist).
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def open_store(d: Path, meta: dict) -> FileMaskRevisionStore:
        if (d / "revisions.json").exists():
            return FileMaskRevisionStore.open(d)
        return FileMaskRevisionStore(d, (meta["height"], meta["width"]))

    def require_run_params(body: dict) -> RunParams:
        # Every parameter is required at this boundary: a missing field must be
        # reported, not silently replaced by a default that could mask a bug.
        required = ("method", "band_px", "texture_threshold", "patch",
                    "search_radius", "seed", "speculative_width")
        missing = [k for k in required if k not in body]
        if missing:
            raise ScratchWorkflowError(f"missing required repair parameter(s): {', '.join(missing)}")
        return RunParams(
            method=body["method"],
            band_px=int(body["band_px"]),
            min_area=int(body.get("min_area", 0)),
            texture_threshold=float(body["texture_threshold"]),
            patch=int(body["patch"]),
            search_radius=int(body["search_radius"]),
            seed=int(body["seed"]),
            speculative_width=float(body["speculative_width"]),
        )

    # ----------------------------------------------------------------- routes
    @app.get("/")
    def index():
        return send_from_directory(str(static_dir), "index.html")

    @app.post("/api/sessions")
    def create_session():
        if "image" not in request.files:
            return jsonify({"error": "missing image file field 'image'"}), 400
        upload = request.files["image"]
        sid = uuid.uuid4().hex[:12]
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        src = d / "original_upload"
        src.write_bytes(upload.read())

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            shutil.rmtree(d, ignore_errors=True)
            return jsonify({"error": "unreadable image"}), 400
        h, w = img.shape[:2]
        dtype = "uint16" if img.dtype == np.uint16 else "uint8"
        meta = {
            "id": sid, "width": w, "height": h, "dtype": dtype,
            "channels": 1 if img.ndim == 2 else img.shape[2],
            "source_name": upload.filename or "image",
            "original_path": str(d / "original"),
        }
        # Keep a lossless copy for processing; the upload may be anything OpenCV read.
        shutil.copyfile(src, d / "original")
        to_display_png(d / "original", d / "preview_original.png", max_side=1600)
        # Version 0 of the reviewed selection: an explicit empty revision.
        open_store(d, {"height": h, "width": w})
        save_session(sid, meta)
        return jsonify(meta)

    @app.get("/api/sessions/<sid>")
    def get_session(sid: str):
        try:
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        return jsonify(meta)

    @app.get("/media/<sid>/<path:fname>")
    def media(sid: str, fname: str):
        try:
            d = _session_dir(root, sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        if ".." in fname or fname.startswith("/"):
            return jsonify({"error": "bad path"}), 400
        p = d / fname
        if not p.exists():
            return jsonify({"error": "not found"}), 404
        return send_file(p)

    @app.post("/api/sessions/<sid>/detect")
    def detect(sid: str):
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        body = request.get_json(silent=True) or {}
        params = DetectParams(
            sensitivity=float(body.get("sensitivity", 0.55)),
            veto_parallel=bool(body.get("veto_parallel", True)),
            include_specks=bool(body.get("include_specks", False)),
        )
        img = cv2.imread(str(d / "original"), cv2.IMREAD_UNCHANGED)
        proposal = CandidateProposal.create(img, params)
        cv2.imwrite(str(d / "candidates_mask.png"),
                    proposal.mask.astype(np.uint8) * 255)
        overlay = evidence_overlay(
            img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0),
            list(proposal.candidates),
        )
        cv2.imwrite(str(d / "candidates_evidence.png"), overlay)
        payload = {
            "proposal_id": proposal.proposal_id,
            "diagnostics": proposal.diagnostics,
            "candidates": [c.summary() for c in proposal.candidates],
            "accepted_mask_pixels": proposal.pixel_count,
        }
        (d / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return jsonify(payload)

    @app.post("/api/sessions/<sid>/mask/apply_candidates")
    def apply_candidates(sid: str):
        """Add accepted detector candidates to the reviewed mask (one explicit action)."""
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        p = d / "candidates_mask.png"
        if not p.exists():
            return jsonify({"error": "run detection first"}), 400
        cand = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        proposal_id = None
        cj = d / "candidates.json"
        if cj.exists():
            proposal_id = json.loads(cj.read_text(encoding="utf-8")).get("proposal_id")
        store = open_store(d, meta)
        # Record the proposal's content identity on the new revision so the
        # link between candidate version and reviewed mask stays explicit.
        linked = CandidateProposal(
            mask=cand, candidates=(), diagnostics={},
            params=DetectParams(), proposal_id=proposal_id or mask_identity(cand),
        )
        rev = store.apply_proposal(linked)
        return _mask_status(store, rev)

    @app.post("/api/sessions/<sid>/mask")
    def edit_mask(sid: str):
        """Incremental mask editing: brush strokes in image coordinates.

        Body: {"strokes": [{"points": [[x,y],...], "radius": 4, "mode": "add"|"erase"}]}
        Each request becomes one new mask revision; no global clear unless explicit.
        """
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        body = request.get_json(silent=True) or {}
        store = open_store(d, meta)
        mask = store.head_mask.astype(np.uint8)
        strokes = body.get("strokes", [])
        origin = "stroke"
        for s in strokes:
            pts = np.array(s.get("points", []), dtype=np.int32)
            if pts.size < 2:
                continue
            r = int(s.get("radius", 4))
            mode = s.get("mode", "add")
            color = 1 if mode == "add" else 0
            if len(pts) == 2:
                pts = pts.reshape(2, 1, 2)
            cv2.polylines(mask, [pts], False, color, max(1, r * 2), cv2.LINE_8)
        if body.get("clear"):
            mask[:] = 0
            origin = "clear"
        if body.get("invert"):
            mask = 255 - mask
            origin = "invert"
        rev = store.commit(mask > 0, origin)
        return _mask_status(store, rev)

    @app.post("/api/sessions/<sid>/mask/undo")
    def undo_mask(sid: str):
        """Rewind one reviewed revision; the initial empty revision is the floor."""
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        store = open_store(d, meta)
        rev = store.undo()
        return _mask_status(store, rev)

    @app.get("/api/sessions/<sid>/mask_stats")
    def mask_stats(sid: str):
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        store = open_store(d, meta)
        return _mask_status(store, store.head)

    def _mask_status(store: FileMaskRevisionStore, rev) -> dict:
        mask = store.head_mask
        components = (
            cv2.connectedComponents(mask.astype(np.uint8))[0] - 1 if mask.any() else 0
        )
        return {
            "mask_pixels": int(mask.sum()),
            "mask_components": int(components),
            "revision": rev.revision,
            "mask_id": rev.mask_id,
            "can_undo": store.head_revision > 0,
        }

    @app.post("/api/sessions/<sid>/restore")
    def restore(sid: str):
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ScratchWorkflowError("request body must be a JSON object")
        params = require_run_params(body)
        store = open_store(d, meta)
        frozen = store.freeze()
        if frozen.pixel_count == 0:
            # An empty selection would silently "succeed" with nothing repaired.
            raise ScratchWorkflowError(
                "mask is empty: draw or apply candidates before running the repair"
            )
        run_dir = d / "run"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        # The processor gets only the frozen snapshot; the revision store and
        # mask.png are never passed in and cannot be modified by it.
        result = repair_with_frozen_mask(
            d / "original", frozen, run_dir, params,
            mask_source=f"reviewed revision {frozen.source_revision} ({frozen.mask_id})",
            compare_original=True,
        )
        # Sanity: the frozen export really is the snapshot the user reviewed.
        written = cv2.imread(str(run_dir / "mask_frozen.png"),
                             cv2.IMREAD_GRAYSCALE) > 0
        if mask_identity(written) != frozen.mask_id:
            raise ScratchWorkflowError("frozen mask export identity mismatch")
        meta["last_run"] = result.report
        save_session(sid, meta)
        return jsonify(_ui_report(result.report, sid))

    @app.get("/api/sessions/<sid>/report")
    def report(sid: str):
        try:
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        if "last_run" not in meta:
            return jsonify({"error": "no run yet"}), 400
        return jsonify(_ui_report(meta["last_run"], sid))

    @app.get("/api/sessions/<sid>/download/<kind>")
    def download(sid: str, kind: str):
        try:
            d = _session_dir(root, sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        mapping = {
            "restored": d / "run" / "restored.png",
            "mask": d / "mask.png",
            "frozen_mask": d / "run" / "mask_frozen.png",
            "report": d / "run" / "report.json",
            "candidates": d / "candidates.json",
            "evidence": d / "candidates_evidence.png",
            "original": d / "original",
        }
        p = mapping.get(kind)
        if p is None or not p.exists():
            return jsonify({"error": "not available"}), 404
        return send_file(p, as_attachment=True, download_name=f"{kind}{p.suffix}")

    return app


def _ui_report(report: dict, sid: str) -> dict:
    """Shape the report for the browser, including media URLs."""
    return {
        "media": {
            "original": f"/media/{sid}/preview_original.png",
            "mask": f"/media/{sid}/run/mask_used.png",
            "frozen_mask": f"/media/{sid}/run/mask_frozen.png",
            "restored": f"/media/{sid}/run/preview_restored.png",
            "difference": f"/media/{sid}/run/preview_difference.png",
            "boundary": f"/media/{sid}/run/preview_boundary.png",
            "speculative": f"/media/{sid}/run/preview_speculative.png",
        },
        "downloads": {
            "restored": f"/api/sessions/{sid}/download/restored",
            "mask": f"/api/sessions/{sid}/download/mask",
            "frozen_mask": f"/api/sessions/{sid}/download/frozen_mask",
            "report": f"/api/sessions/{sid}/download/report",
        },
        "mask": report["mask"],
        "difference": report["difference"],
        "preservation": report["preservation"],
        "zones": report["zones"],
        "parameters": report["parameters"],
        "regions": report["regions"],
        "speculation": report["speculation"],
        "method_notes": report["method_notes"],
        "timing_s": report["timing_s"],
        "inputs": report["inputs"],
    }
