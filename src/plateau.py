# src/plateau.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


# ------------------------- small, local LA utilities -------------------------

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _sorted_eigh_spd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decompose a symmetric matrix; return (evals_desc, evecs_columns_sorted).
    """
    S = _sym(A)
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]  # descending
    return w[idx], V[:, idx]


def _whitener_from_C0(
    C0: np.ndarray,
    *,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Build a whitening map W so that W^T C0 W ≈ I on the kept subspace.

    Returns:
      W          : (N x K) whitening matrix
      K_kept     : effective dimension K
      lam_kept   : kept eigenvalues (desc)
      V_kept     : kept eigenvectors (cols)
    """
    lam, V = _sorted_eigh_spd(C0)
    if eigencut_abs is None:
        eigencut_abs = float(lam[0]) * float(eigencut_rel)

    mask = lam > max(0.0, float(eigencut_abs))
    if not np.any(mask):
        # fall back to the leading mode to avoid empty subspace
        mask = np.zeros_like(lam, dtype=bool)
        mask[0] = True

    lam_k = lam[mask]
    V_k = V[:, mask]

    if keep_k is not None:
        k = max(1, min(int(keep_k), lam_k.shape[0]))
        lam_k = lam_k[:k]
        V_k = V_k[:, :k]

    denom = np.sqrt(lam_k + float(ridge))
    W = V_k / denom[None, :]
    return W, int(lam_k.shape[0]), lam_k, V_k


def _make_Mt(Ct: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Whitened correlator M(t) = W^T C(t) W, symmetrized.
    """
    M = W.T @ _sym(Ct) @ W
    return _sym(M)


def _build_wopt(
    C: np.ndarray,
    t0: int,
    t_star: int,
    *,
    keep_k: Optional[int] = None,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Build the optimal channel vector for cross-correlation in the ORIGINAL basis.
    Returns a vector w (column) normalized in the C(t0)-metric.
    """
    C0 = C[t0]
    W, _, _, _ = _whitener_from_C0(
        C0, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge
    )
    M_star = _make_Mt(C[t_star], W)
    _, U = _sorted_eigh_spd(M_star)
    u0 = U[:, 0]                 # top in whitened space
    w = W @ u0                   # map back
    # normalize in B-metric: w^T C0 w = 1
    denom = float(np.sqrt(max(w.T @ _sym(C0) @ w, 0.0)))
    if denom > 0:
        w = w / denom
    return w


# --------------------------- S0 witnesses (trace/wopt) -----------------------

def s0_trace_internal(
    C: np.ndarray,
    t0: int,
    t_min: int,
    t_max: int,
    *,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    S0_trace(t) = λ_max(M(t)) / Tr(M(t)) in the whitened subspace built from C(t0).
    Returns (S0 array over [t_min..t_max], K_kept).
    """
    C0 = C[t0]
    W, K, _, _ = _whitener_from_C0(
        C0, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge
    )

    S0 = np.full(t_max - t_min + 1, np.nan, dtype=float)
    for i, t in enumerate(range(t_min, t_max + 1)):
        if t < t0:
            continue
        M = _make_Mt(C[t], W)
        evals, _ = _sorted_eigh_spd(M)
        tr = float(np.sum(evals))
        S0[i] = (float(evals[0]) / tr) if tr > 0.0 and np.isfinite(tr) else np.nan
    return S0, K


def s0_wopt_internal(
    C: np.ndarray,
    t0: int,
    t_min: int,
    t_max: int,
    *,
    t_star: Optional[int] = None,
    fit_tmin: Optional[int] = None,
    fit_tmax: Optional[int] = None,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Channel-normalized S0 using the same optimal channel as cross estimator.
    Build w from C(t0), get E0 from OLS fit on ln(Cw) over [fit_tmin, fit_tmax],
    then S0(t) = (Cw(t) / Cw(t0)) * exp(E0 * (t - t0)).
    Returns (S0 array over [t_min..t_max], K_kept).
    """
    if t_star is None:
        t_star = t0 + 1
    t_star = int(max(t0 + 1, t_star))

    # Default fit range if not specified
    if fit_tmin is None:
        fit_tmin = t_star
    if fit_tmax is None:
        fit_tmax = t_star + 6

    C0 = C[t0]
    W, K, _, _ = _whitener_from_C0(
        C0, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge
    )

    # Build optimal channel vector w in original basis, normalized in C(t0) metric
    w = _build_wopt(
        C, t0, t_star,
        eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge
    )
    
    # Collect Cw values and compute E0 from OLS fit on ln(Cw) vs t over [fit_tmin, fit_tmax]
    fit_ts = []
    fit_vals = []
    for t in range(int(fit_tmin), int(fit_tmax) + 1):
        if t < len(C):
            Cw_t = float(w.T @ _sym(C[t]) @ w)
            if Cw_t > 0.0 and np.isfinite(Cw_t):
                fit_ts.append(t)
                fit_vals.append(np.log(Cw_t))
    
    # Compute E0 from linear fit: ln(Cw) ~ a - E0 * t, so E0 = -slope
    E0 = 0.0
    if len(fit_ts) >= 2:
        _, slope = _linfit(np.array(fit_ts, dtype=float), np.array(fit_vals, dtype=float))
        E0 = -slope if np.isfinite(slope) else 0.0
    
    # Compute S0(t) = (Cw(t) / Cw(t0)) * exp(E0 * (t - t0))
    Cw_t0 = float(w.T @ _sym(C0) @ w)  # Should be ≈ 1
    
    S0 = np.full(t_max - t_min + 1, np.nan, dtype=float)
    for i, t in enumerate(range(t_min, t_max + 1)):
        if t < len(C):
            Cw_t = float(w.T @ _sym(C[t]) @ w)
            if np.isfinite(Cw_t) and Cw_t0 > 0:
                S0[i] = (Cw_t / Cw_t0) * np.exp(E0 * (t - t0))
    
    return S0, K


# ----------------------------- linear fit helper ----------------------------

def _linfit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares fit y ~ a + b x; returns (a, b).
    Ignores non-finite y.
    """
    m = np.isfinite(y)
    xm = x[m]
    ym = y[m]
    if xm.size < 2:
        return (np.nan, np.nan)
    X = np.vstack([np.ones_like(xm), xm]).T
    beta, *_ = np.linalg.lstsq(X, ym, rcond=None)
    a, b = float(beta[0]), float(beta[1])
    return a, b


# ------------------------------- main entrypoint ----------------------------

def detect_plateau(
    C: np.ndarray,
    t0: int,
    t_min: int,
    t_max: int,
    *,
    threshold: float = 0.60,
    slope_tol: float = 0.01,
    witness: str = "wopt",
    t_star: Optional[int] = None,
    fit_tmin: Optional[int] = None,
    fit_tmax: Optional[int] = None,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    # Backward compatibility for batch processing
    S: Optional[np.ndarray] = None,
    ts: Optional[np.ndarray] = None,
    **_
) -> Dict[str, object]:
    """
    Compute S0(t) on [t_min, t_max] and decide plateau:
      - S0 >= threshold for all t in the window
      - |slope| <= slope_tol where slope is from a linear fit S0 ~ a + b t

    Returns a dict ready to be embedded in certificate.json under "plateau_selected".
    
    For backward compatibility, if S (precomputed S0) is provided, use that instead.
    """
    if witness not in ("wopt", "trace"):
        raise ValueError("witness must be 'wopt' or 'trace'")

    # Backward compatibility: if S is provided, use precomputed S0
    if S is not None:
        S0 = np.asarray(S, dtype=float)
        K_kept = keep_k  # Use provided value or None
    else:
        # Normal path: compute S0 from correlator
        if witness == "wopt":
            S0, K_kept = s0_wopt_internal(
                C, t0, t_min, t_max,
                t_star=t_star, fit_tmin=fit_tmin, fit_tmax=fit_tmax,
                eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
                keep_k=keep_k, ridge=ridge,
            )
        else:
            S0, K_kept = s0_trace_internal(
                C, t0, t_min, t_max,
                eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
                keep_k=keep_k, ridge=ridge,
            )

    # window coordinates & fit
    ts = np.arange(int(t_min), int(t_max) + 1, dtype=float)
    has_finite = np.isfinite(S0).any()
    
    # Clip S0 for decision logic only (variance-share overshoots >1 are numerical drift)
    S_eval = np.clip(S0, 0.0, 1.0)
    Smin = float(np.nanmin(S_eval)) if has_finite else np.nan
    Smax = float(np.nanmax(S_eval)) if has_finite else np.nan
    _, slope = _linfit(ts, S_eval)

    slope_ok = (np.isfinite(slope) and abs(slope) <= float(slope_tol))
    thresh_ok = (np.isfinite(Smin) and Smin >= float(threshold))
    passed = bool(slope_ok and thresh_ok)

    # assemble exactly how certificate expects it
    result = {
        "t0": int(t0),
        "t_min": int(t_min),
        "t_max": int(t_max),
        "threshold": float(threshold),
        "slope_tol": float(slope_tol),
        "witness": str(witness),
        "passed": passed,
        "plateau_window": [int(t_min), int(t_max)] if passed else None,
        "S0_min": float(Smin) if np.isfinite(Smin) else None,
        "S0_max": float(Smax) if np.isfinite(Smax) else None,
        "K_kept": int(K_kept) if keep_k is not None else None,
    }
    return result


# ------------------------- CLI compatibility wrappers -------------------------

def s0_trace(
    C: np.ndarray,
    t0: int,
    ts: np.ndarray,
    *,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    CLI compatibility wrapper: returns S0 array for (integer) time indices in ts.
    """
    ts = np.asarray(ts, dtype=int)
    t_min = int(ts.min())
    t_max = int(ts.max())
    S0_full, _ = s0_trace_internal(
        C, t0, t_min, t_max,
        eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
        keep_k=keep_k, ridge=ridge,
    )
    indices = [int(t - t_min) for t in ts if t_min <= t <= t_max]
    return S0_full[indices]


def s0_wopt(
    C: np.ndarray,
    ts: np.ndarray,
    t0: int,
    *,
    t_star: Optional[int] = None,
    fit_tmin: Optional[int] = None,
    fit_tmax: Optional[int] = None,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    CLI compatibility wrapper: returns S0 array for (integer) time indices in ts.
    """
    ts = np.asarray(ts, dtype=int)
    t_min = int(ts.min())
    t_max = int(ts.max())
    S0_full, _ = s0_wopt_internal(
        C, t0, t_min, t_max,
        t_star=t_star, fit_tmin=fit_tmin, fit_tmax=fit_tmax,
        eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
        keep_k=keep_k, ridge=ridge,
    )
    indices = [int(t - t_min) for t in ts if t_min <= t <= t_max]
    return S0_full[indices]


def detect_plateau_from_s0(
    S0: np.ndarray,
    ts: np.ndarray,
    t0: int,
    tmin: int,
    tmax: int,
    threshold: float = 0.60,
    slope_tol: float = 0.01,
    witness: str = "wopt",
    keep_k: Optional[int] = None,
) -> Dict[str, object]:
    """
    Detect plateau from a pre-computed S0(ts) series (compat layer).
    """
    ts = np.asarray(ts, dtype=float)
    S0 = np.asarray(S0, dtype=float)
    tmin = int(tmin); tmax = int(tmax)

    # Filter to only include the plateau range [tmin, tmax]
    mask = (ts >= tmin) & (ts <= tmax)
    ts_plateau = ts[mask]
    S0_plateau = S0[mask]

    has_finite = np.isfinite(S0_plateau).any()
    Smin = float(np.nanmin(S0_plateau)) if has_finite else np.nan
    Smax = float(np.nanmax(S0_plateau)) if has_finite else np.nan
    _, slope = _linfit(ts_plateau, S0_plateau)

    slope_ok = np.isfinite(slope) and abs(slope) <= float(slope_tol)
    thresh_ok = np.isfinite(Smin) and (Smin >= float(threshold))
    passed = bool(slope_ok and thresh_ok)

    return {
        "t0": int(t0),
        "t_min": int(tmin),
        "t_max": int(tmax),
        "threshold": float(threshold),
        "slope_tol": float(slope_tol),
        "witness": str(witness),
        "passed": passed,
        "plateau_window": [int(tmin), int(tmax)] if passed else None,
        "S0_min": float(Smin) if np.isfinite(Smin) else None,
        "S0_max": float(Smax) if np.isfinite(Smax) else None,
        "K_kept": int(keep_k) if keep_k is not None else None,
        "slope": float(slope) if np.isfinite(slope) else None,
    }


def s0_series_via_gevp(
    C: np.ndarray,
    t0: int,
    ts: np.ndarray,
    *,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Historical alias used by TOOLcli. We route to wopt for a sensible default.
    """
    return s0_wopt(
        C, ts, t0,
        t_star=None, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
        keep_k=keep_k, ridge=ridge,
    )