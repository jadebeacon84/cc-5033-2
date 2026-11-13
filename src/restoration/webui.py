"""Local review UI: load an image, review detection evidence, paint/erase a mask,
run the repair, and compare original / mask / result / difference side by side.

State is kept per session in a small working directory so the mask can be
edited incrementally (a stroke at a time) over HTTP without re-uploading the
image.  The UI never heuristically erases anything on its own: painting is a
deliberate user action.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from .detect import DetectParams, detect_candidates, evidence_overlay
from .imageio import to_display_png
from .pipeline import RunParams, run


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

    def mask_path(d: Path) -> Path:
        return d / "mask.png"

    def read_mask(d: Path, meta: dict) -> np.ndarray:
        h, w = meta["height"], meta["width"]
        p = mask_path(d)
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == (h, w):
                return m > 0
        return np.zeros((h, w), dtype=bool)

    def write_mask(d: Path, mask: np.ndarray) -> None:
        cv2.imwrite(str(mask_path(d)), mask.astype(np.uint8) * 255)

    def mask_preview(d: Path, meta: dict) -> Path:
        """Mask as a translucent-ready PNG at display resolution + mask PNG."""
        mask = read_mask(d, meta)
        out = d / "preview_mask.png"
        cv2.imwrite(str(out), mask.astype(np.uint8) * 255)
        return out

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
        cv2.imwrite(str(mask_path(d)), np.zeros((h, w), np.uint8))
        meta["has_mask"] = True
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
        mask, candidates, diag = detect_candidates(img, params)
        cv2.imwrite(str(d / "candidates_mask.png"), mask.astype(np.uint8) * 255)
        overlay = evidence_overlay(
            img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0), candidates
        )
        cv2.imwrite(str(d / "candidates_evidence.png"), overlay)
        payload = {"diagnostics": diag, "candidates": [c.summary() for c in candidates],
                   "accepted_mask_pixels": int(mask.sum())}
        (d / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return jsonify(payload)

    @app.post("/api/sessions/<sid>/mask/apply_candidates")
    def apply_candidates(sid: str):
        """Add accepted detector candidates to the user's mask (one explicit action)."""
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        p = d / "candidates_mask.png"
        if not p.exists():
            return jsonify({"error": "run detection first"}), 400
        cand = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        mask = read_mask(d, meta) | cand
        write_mask(d, mask)
        mask_preview(d, meta)
        return jsonify({"mask_pixels": int(mask.sum())})

    @app.post("/api/sessions/<sid>/mask")
    def edit_mask(sid: str):
        """Incremental mask editing: brush strokes in image coordinates.

        Body: {"strokes": [{"points": [[x,y],...], "radius": 4, "mode": "add"|"erase"}]}
        Only the stroke is changed; no global clear unless mode explicit.
        """
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        body = request.get_json(silent=True) or {}
        mask = read_mask(d, meta).astype(np.uint8)
        strokes = body.get("strokes", [])
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
        if body.get("invert"):
            mask = 255 - mask
        out = mask > 0
        write_mask(d, out)
        mask_preview(d, meta)
        return jsonify({"mask_pixels": int(out.sum()), "mask_components": int(
            cv2.connectedComponents(out.astype(np.uint8))[0] - 1
        )})

    @app.get("/api/sessions/<sid>/mask_stats")
    def mask_stats(sid: str):
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        mask = read_mask(d, meta)
        return jsonify({"mask_pixels": int(mask.sum()),
                        "mask_components": int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
                        if mask.any() else 0})

    @app.post("/api/sessions/<sid>/restore")
    def restore(sid: str):
        try:
            d = _session_dir(root, sid)
            meta = load_session(sid)
        except FileNotFoundError:
            return jsonify({"error": "unknown session"}), 404
        body = request.get_json(silent=True) or {}
        params = RunParams(
            method=body.get("method", "hybrid"),
            band_px=int(body.get("band_px", 2)),
            min_area=int(body.get("min_area", 0)),
            texture_threshold=float(body.get("texture_threshold", 3.0e-4)),
            patch=int(body.get("patch", 7)),
            search_radius=int(body.get("search_radius", 40)),
            seed=int(body.get("seed", 0)),
            speculative_width=float(body.get("speculative_width", 4.0)),
        )
        mask_file = mask_path(d)
        report = run(d / "original", mask_file, d / "run", params, compare_original=True)
        meta["last_run"] = report
        save_session(sid, meta)
        return jsonify(_ui_report(report, sid))

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
            "restored": f"/media/{sid}/run/preview_restored.png",
            "difference": f"/media/{sid}/run/preview_difference.png",
            "boundary": f"/media/{sid}/run/preview_boundary.png",
            "speculative": f"/media/{sid}/run/preview_speculative.png",
        },
        "downloads": {
            "restored": f"/api/sessions/{sid}/download/restored",
            "mask": f"/api/sessions/{sid}/download/mask",
            "report": f"/api/sessions/{sid}/download/report",
        },
        "preservation": report["preservation"],
        "zones": report["zones"],
        "parameters": report["parameters"],
        "regions": report["regions"],
        "speculation": report["speculation"],
        "method_notes": report["method_notes"],
        "timing_s": report["timing_s"],
        "inputs": report["inputs"],
    }
