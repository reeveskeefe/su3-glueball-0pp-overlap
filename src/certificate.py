# src/certificate.py
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .io_utils import load_corr_npz, save_json, save_text, ensure_dir
from .plateau import s0_wopt, s0_trace, detect_plateau_from_s0
from .report_md import render_certificate_md  # assumed present


def _estimate_gap_quantile(
    C: np.ndarray,
    t0: int,
    ts: np.ndarray,
    q0: float = 0.10,
    q1: float = 0.90,
    eigencut_rel: float = 1e-3,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    window: Tuple[int, int] = (None, None),
) -> Dict[str, object]:
    """
    Quantile-based Δ estimator from whitened eigen-spectra:
      E0(t) = -ln(λ0(t+1)/λ0(t)), E1(t) analogously from λ1.
      Δ = median_{t in window} (E1 - E0) restricted to central [q0,q1] quantile.
    Returns dict with E0, E1 aggregates and Δ.
    """
    # Build whitened eigenvalues via the trace witness internals:
    # s0_trace computes eigenvalues but doesn't return them; reimplement minimal here.
    from .plateau import _whitener_from_C0, _make_Mt, _sorted_eigh_spd  # local helpers

    T = C.shape[0]
    C0 = C[t0]
    W, _, _ = _whitener_from_C0(C0, eigencut_rel=eigencut_rel, eigencut_abs=None, keep_k=keep_k, ridge=ridge)

    lam0 = []
    lam1 = []
    tins = []
    t_min, t_max = window
    if t_min is None:
        t_min = int(t0 + 1)
    if t_max is None:
        t_max = int(ts[-1])

    for t in range(max(t0 + 1, t_min), min(t_max, int(ts[-1])) + 1):
        M = _make_Mt(C[t], W)
        evals, _ = _sorted_eigh_spd(M)
        if evals.shape[0] >= 2 and np.isfinite(evals[0]) and np.isfinite(evals[1]) and evals[0] > 0.0 and evals[1] > 0.0:
            lam0.append(float(evals[0]))
            lam1.append(float(evals[1]))
            tins.append(t)
    lam0 = np.array(lam0, dtype=float)
    lam1 = np.array(lam1, dtype=float)
    tins = np.array(tins, dtype=int)

    if lam0.size < 2 or lam1.size < 2:
        return {"Delta": 0.0, "E0": 0.0, "E1": 0.0, "K_kept_max": None, "points": 0, "t0": int(t0), "t_window": [t_min, t_max]}

    # effective masses from adjacent ratios
    r0 = lam0[1:] / lam0[:-1]
    r1 = lam1[1:] / lam1[:-1]
    E0_t = -np.log(r0)
    E1_t = -np.log(r1)

    # central quantile trimming
    lo0, hi0 = np.quantile(E0_t[~np.isnan(E0_t)], [q0, q1])
    lo1, hi1 = np.quantile(E1_t[~np.isnan(E1_t)], [q0, q1])
    E0_trim = E0_t[(E0_t >= lo0) & (E0_t <= hi0)]
    E1_trim = E1_t[(E1_t >= lo1) & (E1_t <= hi1)]

    E0 = float(np.median(E0_trim)) if E0_trim.size else 0.0
    E1 = float(np.median(E1_trim)) if E1_trim.size else 0.0
    Delta = max(0.0, E1 - E0)

    return {
        "Delta": float(Delta),
        "E0": float(E0),
        "E1": float(E1),
        "K_kept_max": None,
        "per_t": {
            "E0_t": E0_t.tolist(),
            "E1_t": E1_t.tolist(),
            "t": tins[1:].tolist(),
        },
        "points": int(min(E0_t.size, E1_t.size)),
        "t0": int(t0),
        "t_window": [int(t_min), int(t_max)],
    }


def run_all(
    corr_path: str,
    t0: int,
    dt: int,
    plateau_tmin: int,
    plateau_tmax: int,
    tstar: Optional[int],
    fit_tmin: int,
    fit_tmax: int,
    excited_t_idx: int,
    eigencut_rel: float = 1e-3,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    use_wopt: bool = True,
    witness: Optional[str] = None,
    gap_q0: float = 0.10,
    gap_q1: float = 0.90,
    delta_override: Optional[float] = None,
    bound_thresh: float = 0.06,
) -> Dict[str, object]:
    """
    Minimal self-contained certificate pipeline using internal witnesses.
    Returns a JSON-serializable dict.
    """
    C, ts = load_corr_npz(corr_path)

    # Witness
    if witness is None:
        witness = "wopt" if use_wopt else "trace"

    if witness == "wopt":
        S = s0_wopt(C, t0, ts, eigencut_rel=eigencut_rel, eigencut_abs=None, keep_k=keep_k, ridge=ridge, t_star=tstar, fit_tmin=fit_tmin, fit_tmax=fit_tmax)
    elif witness == "trace":
        S = s0_trace(C, t0, ts, eigencut_rel=eigencut_rel, eigencut_abs=None, keep_k=keep_k, ridge=ridge)
    else:
        raise ValueError("witness must be 'trace' or 'wopt'")

    # Plateau
    plat = detect_plateau_from_s0(S, ts, t0=t0, tmin=plateau_tmin, tmax=plateau_tmax, threshold=0.60, slope_tol=0.01)
    plat_block = {
        "t0": int(t0),
        "t_min": int(plateau_tmin),
        "t_max": int(plateau_tmax),
        "threshold": 0.60,
        "slope_tol": 0.01,
        "plateau_window": list(plat["window"]) if plat["window"] else None,
        "S0_min": plat["S0_min"],
        "S0_max": plat["S0_max"],
        "passed": bool(plat["passed"]),
        "witness": witness,
    }

    # Cross-estimator (simple, robust fallback)
    # Use S0 at fit_tmin as a conservative fraction estimate.
    idx_fit = np.where(ts == int(fit_tmin))[0]
    F0_fit_at_tmin = float(S[idx_fit[0]]) if idx_fit.size and np.isfinite(S[idx_fit[0]]) else float("nan")
    cross_block = {
        "t0": int(t0),
        "t_star": int(tstar if tstar is not None else t0 + 1),
        "fit_window": [int(fit_tmin), int(fit_tmax)],
        "F0_fit_at_tmin": F0_fit_at_tmin,
    }

    # Gap / excited-state smallness bound
    if delta_override is not None:
        Delta = float(delta_override)
        meta = {"Delta": Delta, "override": True}
    else:
        meta = _estimate_gap_quantile(
            C=C,
            t0=t0,
            ts=ts,
            q0=gap_q0,
            q1=gap_q1,
            eigencut_rel=eigencut_rel,
            keep_k=keep_k,
            ridge=ridge,
            window=(plateau_tmin, plateau_tmin + 4),  # short early window for stability
        )
        Delta = float(meta.get("Delta", 0.0))

    dt_eff = int(excited_t_idx - t0)
    bound = math.exp(-Delta * max(0, dt_eff))
    bound_block = {
        "t0": int(t0),
        "t": int(excited_t_idx),
        "dt": int(dt_eff),
        "c_pref": 1.0,
        "Delta_used": float(Delta),
        "bound_1_minus_S0": float(bound),
        "delta_meta": meta,
    }

    # Verdict: plateau OK AND cross OK AND bound small enough
    cross_ok = (F0_fit_at_tmin >= 0.60) if np.isfinite(F0_fit_at_tmin) else False
    bound_ok = (bound <= float(bound_thresh))
    verdict = bool(plat_block["passed"] and cross_ok and bound_ok)

    return {
        "plateau_selected": plat_block,
        "cross_estimator": cross_block,
        "bound": bound_block,
        "verdict": verdict,
    }


def create_certificate_from_results(
    t0: int,
    dt: int,
    plateau_selected: Dict,
    cross_estimator: Dict,
    bound: Dict,
    gevp_snapshot: Dict,
    verdict: bool,
) -> Dict[str, object]:
    """
    Create a certificate from pre-computed results.
    """
    return {
        "t0": int(t0),
        "dt": int(dt),
        "plateau_selected": plateau_selected,
        "cross_estimator": cross_estimator,
        "bound": bound,
        "gevp_snapshot": gevp_snapshot,
        "verdict": verdict,
    }


def package_certificate(
    corr_path: str,
    out_dir: str,
    *,
    t0: int,
    dt: int,
    plateau_tmin: int,
    plateau_tmax: int,
    tstar: Optional[int],
    fit_tmin: int,
    fit_tmax: int,
    bound_t: int,
    eigencut_rel: float = 1e-3,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    use_wopt: bool = True,
    witness: Optional[str] = None,
    gap_q0: float = 0.10,
    gap_q1: float = 0.90,
    delta_override: Optional[float] = None,
    bound_thresh: float = 0.06,
    write_artifacts: bool = True,
) -> Dict[str, object]:
    """
    Wrapper that runs the pipeline and (optionally) writes:
      - outputs/certificate.json
      - outputs/report.md
    """
    ensure_dir(out_dir)
    cert = run_all(
        corr_path=corr_path,
        t0=t0,
        dt=dt,
        plateau_tmin=plateau_tmin,
        plateau_tmax=plateau_tmax,
        tstar=tstar,
        fit_tmin=fit_tmin,
        fit_tmax=fit_tmax,
        excited_t_idx=bound_t,
        eigencut_rel=eigencut_rel,
        keep_k=keep_k,
        ridge=ridge,
        use_wopt=use_wopt,
        witness=witness,
        gap_q0=gap_q0,
        gap_q1=gap_q1,
        delta_override=delta_override,
        bound_thresh=bound_thresh,
    )

    if write_artifacts:
        json_path = f"{out_dir.rstrip('/')}/certificate.json"
        md_path = f"{out_dir.rstrip('/')}/report.md"
        save_json(cert, json_path)
        try:
            md = render_certificate_md(cert)
            save_text(md, md_path)
        except Exception:
            # If report renderer is unavailable, still succeed with JSON.
            pass

    return cert