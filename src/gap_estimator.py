from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from .lin_alg import generalized_eigs

def _safe_energy_from_lambda(lmbda: np.ndarray, dt: np.ndarray) -> np.ndarray:
    eps = 1e-300
    lam = np.clip(lmbda, eps, 1.0 - 1e-15)
    E = -np.log(lam) / np.maximum(dt.astype(float), 1.0)
    E[~np.isfinite(E)] = 0.0
    return E

def estimate_delta_from_multit_gevp(
    C: np.ndarray,
    t0: int,
    *,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = 3,
    ridge: float = 0.0,
    max_points: int = 5,
    q0: float = 0.10,
    q1: float = 0.90,
) -> Dict:
    T = C.shape[0]
    t_start = t0 + 1
    t_end = min(t0 + max_points, T - 1)
    if t_end < t_start:
        return {"t0": int(t0), "t_window": [t_start, t_end], "E0": 0.0, "E1": 0.0, "Delta": 0.0, "points": 0}

    Ct0 = 0.5 * (C[t0] + C[t0].T)
    ts = np.arange(t_start, t_end + 1, dtype=int)
    dt = ts - int(t0)

    lam0, lam1, kept_max = [], [], 0
    for t in ts:
        evals_t, _, K = generalized_eigs(
            C[t], Ct0,
            keep_k=keep_k, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, ridge=ridge
        )
        if evals_t.size == 0:
            lam0.append(1.0); lam1.append(1.0)
            continue
        e = np.sort(np.asarray(evals_t, dtype=float))[::-1]
        lam0.append(float(e[0]))
        lam1.append(float(e[1] if e.size >= 2 else e[0]))
        kept_max = max(kept_max, min(K, e.size))

    lam0 = np.array(lam0, dtype=float)
    lam1 = np.array(lam1, dtype=float)

    E0_t = _safe_energy_from_lambda(lam0, dt)
    E1_t = _safe_energy_from_lambda(lam1, dt)

    # Quantile pooling across t to resist bias (low q for E0, high q for E1)
    E0 = float(np.quantile(E0_t, q0))
    E1 = float(np.quantile(E1_t, q1))
    Delta = float(max(E1 - E0, 0.0))

    return {
        "t0": int(t0),
        "K_kept_max": int(kept_max),
        "t_window": [int(ts[0]), int(ts[-1])],
        "E0": E0,
        "E1": E1,
        "Delta": Delta,
        "points": int(ts.size),
        "per_t": {
            "t": ts.tolist(),
            "lambda0": lam0.tolist(),
            "lambda1": lam1.tolist(),
            "E0_t": E0_t.tolist(),
            "E1_t": E1_t.tolist(),
        },
        "quantiles": {"q0": float(q0), "q1": float(q1)},
    }