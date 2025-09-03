#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

# Local modules
from src.io_utils import load_corr_npz, save_json, save_text, ensure_dir
from src.gevp_stability import run_gevp_stability
from src.plateau import detect_plateau, detect_plateau_from_s0, s0_trace, s0_wopt
from src.cross_estimator import run_cross_estimator
from src.excited_state_bound import run_excited_state_bound
from src.certificate import package_certificate, create_certificate_from_results
from src.report_md import render_certificate_md
from src.glueball_calibration import run_from_json as run_gluecalib_json


# ---------- helpers ----------

def _parse_int_pair(sp: List[str]) -> Tuple[int, int]:
    if len(sp) != 2:
        raise argparse.ArgumentTypeError("expected two integers: MIN MAX")
    a, b = int(sp[0]), int(sp[1])
    if b < a:
        raise argparse.ArgumentTypeError("MAX must be >= MIN")
    return a, b


def _print_json(obj) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    sys.stdout.flush()


# ---------- subcommands ----------

def cmd_load(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)
    meta = {
        "T": int(C.shape[0]),
        "ops": int(C.shape[1]),
        "file": os.path.abspath(args.corr),
        "t_min": int(ts[0]),
        "t_max": int(ts[-1]),
        "spd_hint": "assumed; matrices are symmetrized internally where needed",
    }
    _print_json(meta)
    return 0


def cmd_gevp(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)
    res = run_gevp_stability(
        C=C, ts=ts,
        t0=args.t0, dt=args.dt,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge,
        prune_rel=args.eigencut_rel,   # mild pruning uses same scale
        prune_abs=args.eigencut_abs
    )
    _print_json(res)
    return 0


def cmd_plateau(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)

    tmin, tmax = args.tmin, args.tmax
    if args.witness == "trace":
        S = s0_trace(C=C, t0=args.t0, ts=ts)
        witness = "trace"
    elif args.witness == "wopt":
        S = s0_wopt(
            C=C, ts=ts, t0=args.t0,
            t_star=args.tstar,
            eigencut_rel=args.eigencut_rel,
            eigencut_abs=args.eigencut_abs,
            keep_k=args.keep_k,
            ridge=args.ridge
        )
        witness = "wopt"
    else:
        raise ValueError("unknown witness (use 'trace' or 'wopt')")

    det = detect_plateau_from_s0(
        S0=S, ts=ts, t0=args.t0,
        tmin=tmin, tmax=tmax,
        threshold=args.threshold,
        slope_tol=args.slope_tol,
        witness=witness,
        keep_k=args.keep_k
    )
    out = {
        "t0": args.t0,
        "t_min": tmin,
        "t_max": tmax,
        "threshold": args.threshold,
        "slope_tol": args.slope_tol,
        "witness": witness,
        "passed": bool(det["passed"]),
        "plateau_window": det.get("window", None),
        "S0_min": det.get("S0_min", None),
        "S0_max": det.get("S0_max", None),
    }
    _print_json(out)
    return 0


def cmd_cross(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)
    res = run_cross_estimator(
        C=C, ts=ts,
        t0=args.t0,
        t_star=args.tstar,
        fit_tmin=args.fit_tmin,
        fit_tmax=args.fit_tmax,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge
    )
    _print_json(res)
    return 0


def cmd_bound(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)
    res = run_excited_state_bound(
        C=C, ts=ts,
        t0=args.t0, t=args.t,
        delta_override=args.delta,
        c_pref=args.cpref,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge,
        gap_q0=args.gap_q0,
        gap_q1=args.gap_q1
    )
    _print_json(res)
    return 0


def cmd_cert(args: argparse.Namespace) -> int:
    C, ts = load_corr_npz(args.corr)

    # 1) GEVP stability (informational; stored in certificate)
    gevp = run_gevp_stability(
        C=C, ts=ts,
        t0=args.t0, dt=args.dt,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge,
        prune_rel=args.eigencut_rel,
        prune_abs=args.eigencut_abs
    )

    # 2) Plateau, using selected witness
    if args.use_wopt:
        S = s0_wopt(
            C=C, ts=ts, t0=args.t0,
            t_star=args.tstar,
            fit_tmin=args.fit_tmin,
            fit_tmax=args.fit_tmax,
            eigencut_rel=args.eigencut_rel,
            eigencut_abs=args.eigencut_abs,
            keep_k=args.keep_k,
            ridge=args.ridge
        )
        witness = "wopt"
    else:
        S = s0_trace(C=C, t0=args.t0, ts=ts)
        witness = "trace"

    tmin, tmax = args.plateau_window
    plat = detect_plateau_from_s0(
        S0=S, ts=ts, t0=args.t0,
        tmin=tmin, tmax=tmax,
        threshold=args.threshold,
        slope_tol=args.slope_tol,
        witness=witness,
        keep_k=args.keep_k if args.use_wopt else None
    )
    plateau_selected = {
        "t0": args.t0,
        "t_min": tmin,
        "t_max": tmax,
        "threshold": args.threshold,
        "slope_tol": args.slope_tol,
        "witness": witness,
        "passed": bool(plat["passed"]),
        "plateau_window": plat.get("window", None),
        "S0_min": plat.get("S0_min", None),
        "S0_max": plat.get("S0_max", None),
        "K_kept": gevp.get("K_kept", None)
    }

    # 3) Cross-estimator agreement (in same window)
    cross = run_cross_estimator(
        C=C, ts=ts,
        t0=args.t0,
        t_star=args.tstar,
        fit_tmin=args.fit_tmin,
        fit_tmax=args.fit_tmax,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge
    )

    # 4) Excited-state smallness bound (data-driven Δ)
    bound = run_excited_state_bound(
        C=C, ts=ts,
        t0=args.t0, t=args.boundt,
        delta_override=args.delta,
        c_pref=args.cpref,
        eigencut_rel=args.eigencut_rel,
        eigencut_abs=args.eigencut_abs,
        keep_k=args.keep_k,
        ridge=args.ridge,
        gap_q0=args.gap_q0,
        gap_q1=args.gap_q1
    )

    # 5) Verdict + package + write files
    verdict = bool(
        plateau_selected.get("passed", False)
        and (cross.get("F0_fit_at_tmin", 0.0) >= args.threshold)
        and (bound.get("bound_1_minus_S0", 1.0) <= args.bound_thresh)
    )

    cert = create_certificate_from_results(
        t0=args.t0,
        dt=args.dt,
        plateau_selected=plateau_selected,
        cross_estimator=cross,
        bound=bound,
        gevp_snapshot=gevp,
        verdict=verdict
    )

    ensure_dir(args.out)
    json_path = os.path.join(args.out, "certificate.json")
    md_path = os.path.join(args.out, "report.md")

    save_json(cert, json_path)
    report_md = render_certificate_md(cert)
    save_text(report_md, md_path)

    _print_json({"out_dir": args.out, "verdict": verdict})
    return 0


def cmd_cert_batch(args: argparse.Namespace) -> int:
    inputs = [s for s in args.inputs.split(",") if s.strip()]
    summaries = []
    passes = 0

    for corr in inputs:
        try:
            C, ts = load_corr_npz(corr)

            # plateau detection (using new interface)
            tmin, tmax = args.plateau_window
            plat = detect_plateau(
                C=C, t0=args.t0,
                t_min=tmin, t_max=tmax,
                threshold=args.threshold,
                slope_tol=args.slope_tol,
                witness="wopt" if args.use_wopt else "trace",
                t_star=args.tstar,
                fit_tmin=args.fit_tmin,
                fit_tmax=args.fit_tmax,
                eigencut_rel=args.eigencut_rel,
                eigencut_abs=args.eigencut_abs,
                keep_k=args.keep_k,
                ridge=args.ridge
            )

            # cross-estimator
            cross = run_cross_estimator(
                C=C, ts=ts,
                t0=args.t0, t_star=args.tstar,
                fit_tmin=args.fit_tmin, fit_tmax=args.fit_tmax,
                eigencut_rel=args.eigencut_rel,
                eigencut_abs=args.eigencut_abs,
                keep_k=args.keep_k,
                ridge=args.ridge
            )

            # bound (data-driven Δ)
            bnd = run_excited_state_bound(
                C=C, ts=ts,
                t0=args.t0, t=args.boundt,
                delta_override=args.delta,
                c_pref=args.cpref,
                eigencut_rel=args.eigencut_rel,
                eigencut_abs=args.eigencut_abs,
                keep_k=args.keep_k,
                ridge=args.ridge,
                gap_q0=args.gap_q0,
                gap_q1=args.gap_q1
            )

            verdict = bool(
                plat.get("passed", False)
                and (cross.get("F0_fit_at_tmin", 0.0) >= args.threshold)
                and (bnd.get("bound_1_minus_S0", 1.0) <= args.bound_thresh)
            )

            passes += int(verdict)
            summaries.append({
                "corr": corr,
                "verdict": verdict,
                "plateau_passed": bool(plat.get("passed", False)),
                "F0_fit_at_tmin": float(cross.get("F0_fit_at_tmin", 0.0)),
                "S0_min": float(plat.get("S0_min", 0.0)) if plat.get("S0_min") is not None else None,
                "S0_max": float(plat.get("S0_max", 0.0)) if plat.get("S0_max") is not None else None,
                "bound_1_minus_S0": float(bnd.get("bound_1_minus_S0", 1.0)),
            })
        except Exception as e:
            summaries.append({
                "corr": corr,
                "error": str(e),
                "verdict": False
            })

    out = {
        "files": len(inputs),
        "passes": passes,
        "fraction_pass": (passes / max(len(inputs), 1)),
        "summaries": summaries
    }

    ensure_dir(args.out)
    save_json(out, os.path.join(args.out, "batch_summary.json"))
    _print_json(out)
    return 0


def cmd_gluecalib(args: argparse.Namespace) -> int:
    ensure_dir(args.out)
    json_path = os.path.join(args.out, "glue_calibration.json")
    md_path = os.path.join(args.out, "glue_calibration.md")
    cert = run_gluecalib_json(constants_path=args.constants, out_json=json_path, out_md=md_path)
    _print_json(cert)
    return 0


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(
        prog="constructive-cli",
        description="Constructive SU(3) 0++ overlap & calibration CLI"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # load
    p_load = sub.add_parser("load", help="Load and summarize correlator file.")
    p_load.add_argument("--corr", required=True)
    p_load.set_defaults(func=cmd_load)

    # common numerical knobs
    def add_common(m):
        m.add_argument("--eigencut-rel", type=float, default=1e-4)
        m.add_argument("--eigencut-abs", type=float, default=None)
        m.add_argument("--keep-k", type=int, default=None)
        m.add_argument("--ridge", type=float, default=0.0)

    # gevp
    p_gevp = sub.add_parser("gevp", help="Run GEVP stability check.")
    p_gevp.add_argument("--corr", required=True)
    p_gevp.add_argument("--t0", type=int, required=True)
    p_gevp.add_argument("--dt", type=int, required=True)
    add_common(p_gevp)
    p_gevp.set_defaults(func=cmd_gevp)

    # plateau
    p_plateau = sub.add_parser("plateau", help="Detect S0 plateau.")
    p_plateau.add_argument("--corr", required=True)
    p_plateau.add_argument("--t0", type=int, required=True)
    p_plateau.add_argument("--tmin", type=int, required=True)
    p_plateau.add_argument("--tmax", type=int, required=True)
    p_plateau.add_argument("--threshold", type=float, default=0.60)
    p_plateau.add_argument("--slope-tol", type=float, default=0.01)
    p_plateau.add_argument("--witness", choices=["trace", "wopt"], default="wopt")
    p_plateau.add_argument("--tstar", type=int, default=None)
    add_common(p_plateau)
    p_plateau.set_defaults(func=cmd_plateau)

    # cross
    p_cross = sub.add_parser("cross", help="Cross-estimator agreement.")
    p_cross.add_argument("--corr", required=True)
    p_cross.add_argument("--t0", type=int, required=True)
    p_cross.add_argument("--tstar", type=int, required=True)
    p_cross.add_argument("--fit-tmin", type=int, required=True)
    p_cross.add_argument("--fit-tmax", type=int, required=True)
    add_common(p_cross)
    p_cross.set_defaults(func=cmd_cross)

    # bound
    p_bound = sub.add_parser("bound", help="Excited-state smallness bound.")
    p_bound.add_argument("--corr", required=True)
    p_bound.add_argument("--t0", type=int, required=True)
    p_bound.add_argument("--t", type=int, required=True)
    p_bound.add_argument("--delta", type=float, default=None)
    p_bound.add_argument("--cpref", type=float, default=1.0)
    p_bound.add_argument("--gap-q0", type=float, default=0.10)
    p_bound.add_argument("--gap-q1", type=float, default=0.90)
    add_common(p_bound)
    p_bound.set_defaults(func=cmd_bound)

    # cert
    p_cert = sub.add_parser("cert", help="Run overlap pipeline and write certificate + report.")
    p_cert.add_argument("--corr", required=True)
    p_cert.add_argument("--out", required=True)
    p_cert.add_argument("--t0", type=int, required=True)
    p_cert.add_argument("--dt", type=int, required=True)
    p_cert.add_argument("--plateau", nargs=2, type=int, metavar=("TMIN", "TMAX"), required=True)
    p_cert.add_argument("--tstar", type=int, required=True)
    p_cert.add_argument("--fit", nargs=2, type=int, metavar=("TMIN", "TMAX"), required=True)
    p_cert.add_argument("--boundt", type=int, required=True)
    p_cert.add_argument("--threshold", type=float, default=0.60)
    p_cert.add_argument("--slope-tol", type=float, default=0.01)
    p_cert.add_argument("--use-wopt", action="store_true", default=False)
    p_cert.add_argument("--delta", type=float, default=None)
    p_cert.add_argument("--cpref", type=float, default=1.0)
    p_cert.add_argument("--gap-q0", type=float, default=0.10)
    p_cert.add_argument("--gap-q1", type=float, default=0.90)
    p_cert.add_argument("--bound-thresh", type=float, default=0.05)
    add_common(p_cert)
    p_cert.set_defaults(func=cmd_cert)

    # cert-batch
    p_cb = sub.add_parser("cert-batch", help="Batch certificate summary for multiple correlators.")
    p_cb.add_argument("--inputs", required=True, help="Comma-separated list of .npz correlators")
    p_cb.add_argument("--out", required=True)
    p_cb.add_argument("--t0", type=int, required=True)
    p_cb.add_argument("--dt", type=int, required=True)
    p_cb.add_argument("--plateau", nargs=2, type=int, metavar=("TMIN", "TMAX"), required=True)
    p_cb.add_argument("--tstar", type=int, required=True)
    p_cb.add_argument("--fit", nargs=2, type=int, metavar=("TMIN", "TMAX"), required=True)
    p_cb.add_argument("--boundt", type=int, required=True)
    p_cb.add_argument("--threshold", type=float, default=0.60)
    p_cb.add_argument("--slope-tol", type=float, default=0.01)
    p_cb.add_argument("--use-wopt", action="store_true", default=False)
    p_cb.add_argument("--delta", type=float, default=None)
    p_cb.add_argument("--cpref", type=float, default=1.0)
    p_cb.add_argument("--gap-q0", type=float, default=0.10)
    p_cb.add_argument("--gap-q1", type=float, default=0.90)
    p_cb.add_argument("--bound-thresh", type=float, default=0.05)
    add_common(p_cb)
    p_cb.set_defaults(func=cmd_cert_batch)

    # gluecalib
    p_gc = sub.add_parser("gluecalib", help="Calibrate τ/√σ, locality, and KP bound from a constants JSON.")
    p_gc.add_argument("--constants", required=True)
    p_gc.add_argument("--out", required=True)
    p_gc.set_defaults(func=cmd_gluecalib)

    args = p.parse_args()

    # expand tuple args
    if getattr(args, "plateau", None) is not None:
        args.plateau_window = _parse_int_pair([str(args.plateau[0]), str(args.plateau[1])])
    if getattr(args, "fit", None) is not None:
        args.fit_tmin, args.fit_tmax = _parse_int_pair([str(args.fit[0]), str(args.fit[1])])

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())