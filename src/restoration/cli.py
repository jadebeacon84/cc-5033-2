"""Command-line entry point.

    restoration restore  IMAGE [--mask M] [--out DIR] [--method hybrid] ...
    restoration detect   IMAGE [--out DIR]            # candidates + evidence
    restoration benchmark REF [REF ...] [--out DIR]   # injected-defect evaluation
    restoration serve    [--out DIR] [--port 8000]    # local review UI
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detect import DetectParams
from .evaluate import run_reference_suite
from .pipeline import RunParams, detect_only, run


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--method", default="hybrid", choices=["hybrid", "exemplar", "smooth"])
    p.add_argument("--band-px", type=int, default=2, dest="band_px",
                   help="transition band in px (0 = none; outside band is bit-identical)")
    p.add_argument("--min-area", type=int, default=0, dest="min_area")
    p.add_argument("--texture-threshold", type=float, default=3.0e-4, dest="texture_threshold")
    p.add_argument("--patch", type=int, default=7)
    p.add_argument("--search-radius", type=int, default=40, dest="search_radius")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speculative-width", type=float, default=4.0, dest="speculative_width")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="restoration", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("restore", help="restore one image using a reviewed mask")
    r.add_argument("image")
    r.add_argument("--mask", default=None)
    _add_common(r)

    d = sub.add_parser("detect", help="propose scratch/dust candidates with evidence")
    d.add_argument("image")
    d.add_argument("--out", default="out/detect")
    d.add_argument("--sensitivity", type=float, default=0.55, dest="sensitivity")
    d.add_argument("--no-parallel-veto", action="store_true", dest="no_parallel_veto")
    d.add_argument("--include-specks", action="store_true", dest="include_specks")

    b = sub.add_parser("benchmark", help="inject known defects into references and score")
    b.add_argument("references", nargs="+")
    b.add_argument("--out", default="out/benchmark")
    b.add_argument("--band-px", type=int, default=2)

    s = sub.add_parser("serve", help="start the local review UI")
    s.add_argument("--out", default="out/ui")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "restore":
        params = RunParams(
            method=args.method, band_px=args.band_px, min_area=args.min_area,
            texture_threshold=args.texture_threshold, patch=args.patch,
            search_radius=args.search_radius, seed=args.seed,
            speculative_width=args.speculative_width,
        )
        report = run(args.image, args.mask, args.out, params)
        print(json.dumps({"ok": True, "report": report["preservation"],
                          "restored": report["outputs"]["restored"],
                          "speculative_pixels": report["speculation"]["speculative_pixels"]},
                         indent=2))
    elif args.command == "detect":
        params = DetectParams(
            sensitivity=args.sensitivity,
            veto_parallel=not args.no_parallel_veto,
            include_specks=args.include_specks,
        )
        payload = detect_only(args.image, args.out, params)
        print(json.dumps(payload["diagnostics"], indent=2))
    elif args.command == "benchmark":
        agg = []
        for ref in args.references:
            out_dir = Path(args.out) / Path(ref).stem
            summary = run_reference_suite(Path(ref), out_dir, band_px=args.band_px)
            agg.append(summary)
        (Path(args.out) / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(json.dumps([{"reference": s["reference"], "cases": s["cases"]} for s in agg], indent=2))
    elif args.command == "serve":
        from .webui import create_app

        app = create_app(Path(args.out))
        print(f"serving on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
