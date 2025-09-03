# src/cross_estimator.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import math
import numpy as np

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _whiten_from_B(
    B: np.ndarray,
    *,
    eigencut_rel: Optional[float],
    eigencut_abs: Optional[float],
    keep_k: Optional[int],
    ridge: float,
) -> Tuple[np.ndarray, int]:
    B = _sym(B)
    ew, EV = np.linalg.eigh(B)
    idx = np.argsort(ew)[::-1]
    ew = ew[idx]
    EV = EV[:, idx]

    mask = np.ones_like(ew, dtype=bool)
    if eigencut_rel is not None:
        vmax = float(np.max(ew))
        mask &= ew >= (eigencut_rel * vmax)
    if eigencut_abs is not None:
        mask &= ew >= eigencut_abs

    kept_idx = np.nonzero(mask)[0]
    if keep_k is not None:
        kept_idx = kept_idx[: int(keep_k)]
    if kept_idx.size == 0:
        kept_idx = np.array([0], dtype=int)

    ew_kept = ew[kept_idx]
    EV_kept = EV[:, kept_idx]

    denom = np.sqrt(np.maximum(ew_kept + float(ridge), 0.0))
    inv_sqrt = np.diag(1.0 / denom)
    W = EV_kept @ inv_sqrt
    return W, int(len(ew_kept))

def _principal_vec_gep(A: np.ndarray, B: np.ndarray, W: np.ndarray) -> np.ndarray:
    S = _sym(W.T @ A @ W)
    ew, EV = np.linalg.eigh(S)
    u_sub = EV[:, -1]
    u = W @ u_sub
    denom = float(np.sqrt(max(u.T @ _sym(B) @ u, 0.0)))
    if denom > 0:
        u = u / denom
    return u

def _project_series(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    T = C.shape[0]
    out = np.empty(T, dtype=float)
    for i in range(T):
        out[i] = float(w.T @ _sym(C[i]) @ w)
    return out

def _ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = x.astype(float)
    y = y.astype(float)
    n = x.size
    xm = float(np.mean(x)); ym = float(np.mean(y))
    xc = x - xm; yc = y - ym
    ss_xx = float(np.dot(xc, xc))
    if ss_xx == 0.0:
        return ym, 0.0, 0.0
    b = float(np.dot(xc, yc) / ss_xx)
    a = ym - b * xm
    y_hat = a + b * x
    ss_tot = float(np.dot(y - ym, y - ym))
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    R2 = 0.0 if ss_tot <= 0.0 else max(0.0, 1.0 - ss_res / ss_tot)
    return a, b, R2

# --- wrapper to use the exact same w_opt as plateau (no import cycle) ---
def _build_opt_channel(
    C: np.ndarray,
    t0: int,
    t_star: int,
    keep_k: Optional[int] = None,
    eigencut_rel: float = 0.0,
    eigencut_abs: Optional[float] = 0.0,
    ridge: float = 0.0,
) -> np.ndarray:
    from .plateau import _build_wopt  # local import to avoid circular dependency
    return _build_wopt(
        C, t0, t_star,
        keep_k=keep_k,
        eigencut_rel=eigencut_rel,
        eigencut_abs=eigencut_abs,
        ridge=ridge,
    )

def run_cross_estimator(
    *,
    C: np.ndarray,
    ts: np.ndarray,
    t0: int,
    t_star: int,
    fit_tmin: int,
    fit_tmax: int,
    keep_k: Optional[int] = None,
    eigencut_rel: Optional[float] = None,
    eigencut_abs: Optional[float] = None,
    ridge: float = 0.0,
) -> Dict[str, float]:
    if C.ndim != 3 or C.shape[1] != C.shape[2]:
        raise ValueError(f"C must have shape (T,N,N); got {C.shape}")
    T, N, _ = C.shape
    t_to_idx = {int(ts[i]): i for i in range(len(ts))}
    if int(t0) not in t_to_idx or int(t_star) not in t_to_idx:
        raise ValueError(f"t0={t0} or t*={t_star} not in times {ts.tolist()}")
    if not (fit_tmin <= fit_tmax):
        raise ValueError(f"fit_tmin={fit_tmin} must be <= fit_tmax={fit_tmax}")

    i0 = t_to_idx[int(t0)]
    is_ = t_to_idx[int(t_star)]
    B = _sym(C[i0])

    # Report K_kept consistently with plateau whitening
    _, K = _whiten_from_B(
        B,
        eigencut_rel=eigencut_rel,
        eigencut_abs=eigencut_abs,
        keep_k=keep_k,
        ridge=ridge,
    )

    # Exact same optimized channel as plateau
    w = _build_opt_channel(
        C, t0, t_star,
        keep_k=keep_k,
        eigencut_rel=(0.0 if eigencut_rel is None else float(eigencut_rel)),
        eigencut_abs=(0.0 if eigencut_abs is None else float(eigencut_abs)),
        ridge=float(ridge),
    )

    Cw = _project_series(C, w)

    tmask = []
    for t in range(fit_tmin, fit_tmax + 1):
        idx = t_to_idx.get(int(t))
        if idx is None:
            continue
        if Cw[idx] > 0.0 and np.isfinite(Cw[idx]):
            tmask.append(t)
    if len(tmask) < 2:
        return {
            "t0": int(t0),
            "t_star": int(t_star),
            "fit_tmin": int(fit_tmin),
            "fit_tmax": int(fit_tmax),
            "F0_fit_at_tmin": 0.0,
            "K_kept": int(K),
        }

    xs = np.array(tmask, dtype=float)
    ys = np.array([math.log(Cw[t_to_idx[int(t)]]) for t in tmask], dtype=float)

    a, slope, R2 = _ols_line(xs, ys)
    F0_fit_at_tmin = float(max(0.0, min(1.0, R2)))

    return {
        "t0": int(t0),
        "t_star": int(t_star),
        "fit_tmin": int(fit_tmin),
        "fit_tmax": int(fit_tmax),
        "F0_fit_at_tmin": F0_fit_at_tmin,
        "K_kept": int(K),
    }