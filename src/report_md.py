# src/report_md.py
# Production-grade Markdown renderers for certificate + calibration reports.

from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple

def _fmt_bool(b: bool) -> str:
    return "✅ yes" if b else "❌ no"

def _fmt_float(x: float, prec: int = 6) -> str:
    return f"{x:.{prec}f}"

def _fmt_tuple_ii(t: Iterable[int]) -> str:
    a = list(t)
    if len(a) == 2:
        return f"[{a[0]},{a[1]}]"
    return "[" + ",".join(str(v) for v in a) + "]"

def _md_header(title: str, level: int = 1) -> str:
    return f"{'#' * level} {title}"

def render_certificate_md(payload: Dict[str, Any]) -> str:
    """
    Render the overlap 'certificate' payload (JSON dict) into a human-auditable Markdown report.
    Expected fields (robustly accessed):
      - gevp: { t0, dt, min_cos, cond_t0, K_kept? }
      - plateau_selected: { t0, t_min, t_max, threshold, slope_tol, plateau_window|None, passed, S0_min, S0_max, witness, K_kept? }
      - cross_estimator: { t0, t_star, fit_tmin, fit_tmax, F0_fit_at_tmin }
      - bound: { t0, t, dt, Delta_used, c_pref, bound_1_minus_S0, delta_meta? }
      - verdict: bool
      - meta: optional free-form
    """
    lines = []

    # Title
    lines.append(_md_header("0++ Overlap Certificate", 1))

    # GEVP
    gevp = payload.get("gevp", {})
    lines.append(_md_header("GEVP Stability", 2))
    lines.append(f"- t0: **{gevp.get('t0', 'NA')}**, dt: **{gevp.get('dt', 'NA')}**")
    if "K_kept" in gevp and gevp["K_kept"] is not None:
        lines.append(f"- rank kept (K): **{gevp['K_kept']}**")
    min_cos = gevp.get("min_cos", None)
    if min_cos is not None:
        lines.append(f"- min cos(angle(u0@t0, u0@t0±1)) = **{_fmt_float(min_cos, 6)}**")
    cond = gevp.get("cond_t0", None)
    if cond is not None:
        lines.append(f"- cond(C(t0)) = **{cond}**")

    # Plateau
    plat = payload.get("plateau_selected", {})
    lines.append("")
    lines.append(_md_header("Plateau (S0 ≥ threshold)", 2))
    lines.append(f"- threshold: **{plat.get('threshold','NA')}**, slope_tol: **{plat.get('slope_tol','NA')}**")
    if plat.get("plateau_window") is None:
        lines.append(f"- plateau window: **none**")
    else:
        tmin = plat.get("t_min", "NA")
        tmax = plat.get("t_max", "NA")
        lines.append(f"- plateau window: **{_fmt_tuple_ii((tmin, tmax))}**")
    lines.append(f"- witness: **{plat.get('witness','NA')}**")
    if "K_kept" in plat and plat["K_kept"] is not None:
        lines.append(f"- rank kept (K): **{plat['K_kept']}**")
    s0min = plat.get("S0_min", None)
    s0max = plat.get("S0_max", None)
    if s0min is not None and s0max is not None:
        lines.append(f"- S0 range: **{_fmt_float(s0min,6)} – {_fmt_float(s0max,6)}**")
    lines.append(f"- passed: **{_fmt_bool(bool(plat.get('passed', False)))}**")

    # Cross-estimator
    cross = payload.get("cross_estimator", {})
    lines.append("")
    lines.append(_md_header("Cross-Estimator (single-exp fraction)", 2))
    lines.append(f"- t0: **{cross.get('t0','NA')}**, t*: **{cross.get('t_star','NA')}**, fit window: **{_fmt_tuple_ii((cross.get('fit_tmin','NA'), cross.get('fit_tmax','NA')))}**")
    f0 = cross.get("F0_fit_at_tmin", None)
    if f0 is not None:
        lines.append(f"- F0_fit(tmin) = **{_fmt_float(float(f0),6)}**")

    # Excited-state bound
    bnd = payload.get("bound", {})
    lines.append("")
    lines.append(_md_header("Excited-State Bound", 2))
    dt = bnd.get("dt", None)
    s0_here = bnd.get("S0_at_t", None)
    if s0_here is not None:
        lines.append(f"- dt = **{dt}**, S0(t) = **{_fmt_float(float(s0_here),6)}**")
    else:
        if dt is not None:
            lines.append(f"- dt = **{dt}**")
    Delta_used = bnd.get("Delta_used", None)
    if Delta_used is not None:
        lines.append(f"- Δ_est = **{_fmt_float(float(Delta_used),6)}**")
    bound_val = bnd.get("bound_1_minus_S0", None)
    if bound_val is not None:
        lines.append(f"- bound: **{bound_val:.6e}**")
    if s0_here is not None and bound_val is not None:
        resid = 1.0 - float(s0_here)
        lines.append(f"- actual residual: **{resid:.6e}**")
        small_enough = resid <= float(bound_val)
        lines.append(f"- small_enough? **{_fmt_bool(small_enough)}**")
    else:
        # conservative textual line if some piece is missing
        lines.append(f"- small_enough? **{_fmt_bool(False)}**")

    # Verdict
    lines.append("")
    lines.append(_md_header("Verdict", 2))
    lines.append(f"- All checks pass? **{_fmt_bool(bool(payload.get('verdict', False)))}**")

    # Footer
    lines.append("")
    lines.append("---")
    lines.append("*Deterministic analysis; no RNG; all linear algebra hardened for SPD.*")

    return "\n".join(lines)

def render_gluecalib_md(calib: Dict[str, Any]) -> str:
    """
    Optional: render glueball calibration output.
    """
    lines = []
    lines.append(_md_header("Glueball 0++ Calibration", 1))

    rho = calib.get("rho_tau_over_sqrt_sigma", None)
    if rho is not None:
        lines.append(f"- τ/√σ = **{_fmt_float(float(rho),6)}**  (must be ≤ 3.405)")

    ell = calib.get("ell_phys", None)
    if ell is not None:
        lines.append(f"- ℓ_phys = **{_fmt_float(float(ell),6)} fm**")

    kstar = calib.get("k_star", None)
    if kstar is not None:
        lines.append(f"- k★ = **{kstar}**")

    bound_ratio = calib.get("m0_over_sqrtsigma_bound", None)
    if bound_ratio is not None:
        lines.append(f"- m_0 / √σ (bound) ≥ **{_fmt_float(float(bound_ratio),6)}**  (target 3.405)")

    verdict = calib.get("verdict", None)
    if verdict is not None:
        lines.append(f"- Consistency: **{_fmt_bool(bool(verdict))}**")

    return "\n".join(lines)