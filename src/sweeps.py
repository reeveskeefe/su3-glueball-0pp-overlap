from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from .plateau import detect_plateau
from .cross_estimator import run_cross_estimator
from .io_utils import load_corr_npz

def _drop_op(C: np.ndarray, j: int) -> np.ndarray:
    return np.delete(np.delete(C, j, axis=1), j, axis=2)

def run_loo_plateau_sweep(
    C: np.ndarray,
    ts: np.ndarray,
    *,
    t0: int,
    t_min: int,
    t_max: int,
    t_star: int,
    fit_tmin: int,
    fit_tmax: int,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Dict:
    N = C.shape[1]
    results: List[Dict] = []
    passes = 0
    for j in range(N):
        Cj = _drop_op(C, j)
        # wopt on reduced basis
        cross = run_cross_estimator(
            Cj, ts, t0=t0, t_star=t_star, fit_tmin=fit_tmin, fit_tmax=fit_tmax,
            keep_k=None if keep_k is None else max(1, min(keep_k, N-1)),
            eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, ridge=ridge
        )
        wopt = np.array(cross["weights"], dtype=float)
        plat = detect_plateau(
            Cj, ts, t0=t0, t_min=t_min, t_max=t_max, use_wopt=wopt
        )
        res = {
            "dropped_op": int(j),
            "passed": bool(plat["passed"]),
            "S0_min": float(plat["S0_min"]),
            "S0_max": float(plat["S0_max"]),
            "plateau_window": plat["plateau_window"],
            "F0_fit_tmin": float(cross["F0_fit_at_tmin"]),
        }
        results.append(res)
        passes += int(plat["passed"])
    return {
        "total_ops": int(N),
        "passes": int(passes),
        "fraction_pass": float(passes / max(N, 1)),
        "cases": results
    }

def run_cert_batch(
    corr_paths: List[str],
    *,
    t0: int,
    dt: int,
    plateau: Tuple[int,int],
    t_star: int,
    fit: Tuple[int,int],
    bound_t: int,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    use_wopt: bool = True,
    delta_override: Optional[float] = None,
    runner=None,
) -> Dict:
    from .certificate import run_all
    tmin_plat, tmax_plat = plateau
    fit_tmin, fit_tmax = fit
    summaries = []
    n_pass = 0
    for p in corr_paths:
        payload = run_all(
            corr_path=p, out_dir=str("."),
            t0_idx=t0, dt=dt,
            plateau_tmin=tmin_plat, plateau_tmax=tmax_plat,
            f0_tstar_idx=t_star, fit_tmin=fit_tmin, fit_tmax=fit_tmax,
            excited_t_idx=bound_t,
            eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
            keep_k=keep_k, ridge=ridge,
            use_wopt_for_plateau=use_wopt,
            delta_override=delta_override
        )
        summaries.append({
            "corr": p,
            "verdict": bool(payload["verdict"]),
            "plateau_passed": bool(payload["plateau_selected"]["passed"]),
            "F0_fit_at_tmin": float(payload["cross_estimator"]["F0_fit_at_tmin"]),
            "S0_min": float(payload["plateau_selected"]["S0_min"]),
            "S0_max": float(payload["plateau_selected"]["S0_max"]),
            "bound_1_minus_S0": float(payload["bound"]["bound_1_minus_S0"])
        })
        n_pass += int(payload["verdict"])
    return {
        "files": len(corr_paths),
        "passes": int(n_pass),
        "fraction_pass": float(n_pass / max(len(corr_paths), 1)),
        "summaries": summaries
    }