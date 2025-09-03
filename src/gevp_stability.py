# src/gevp_stability.py
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import numpy as np

def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _whiten_from_B(
    B: np.ndarray,
    *,
    prune_rel: Optional[float],
    prune_abs: Optional[float],
    keep_k: Optional[int],
    ridge: float,
) -> Tuple[np.ndarray, int]:
    B = _sym(B)
    ew, EV = np.linalg.eigh(B)
    idx = np.argsort(ew)[::-1]
    ew = ew[idx]
    EV = EV[:, idx]

    mask = np.ones_like(ew, dtype=bool)
    if prune_rel is not None:
        vmax = float(np.max(ew))
        mask &= (ew >= prune_rel * vmax)
    if prune_abs is not None:
        mask &= (ew >= prune_abs)

    if keep_k is not None:
        mask_idx = np.nonzero(mask)[0][: int(keep_k)]
    else:
        mask_idx = np.nonzero(mask)[0]

    if mask_idx.size == 0:
        mask_idx = np.array([0], dtype=int)

    ew_kept = ew[mask_idx]
    EV_kept = EV[:, mask_idx]

    denom = np.sqrt(np.maximum(ew_kept + float(ridge), 0.0))
    inv_sqrt = np.diag(1.0 / denom)
    W = EV_kept @ inv_sqrt
    return W, int(len(ew_kept))

def _principal_generalized_eigvec(A: np.ndarray, B: np.ndarray, W: np.ndarray) -> np.ndarray:
    S = _sym(W.T @ A @ W)
    ew, EV = np.linalg.eigh(S)
    u_sub = EV[:, -1]
    u = W @ u_sub
    denom = float(np.sqrt(u.T @ B @ u))
    if denom > 0:
        u = u / denom
    return u

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return abs(float(np.dot(u, v) / (nu * nv)))

def _cond2(A: np.ndarray) -> float:
    s = np.linalg.svd(_sym(A), compute_uv=False)
    smax = float(np.max(s))
    smin = float(np.min(s))
    if smin <= 0.0:
        return math.inf
    return smax / smin

def run_gevp_stability(
    *,
    C: np.ndarray,
    ts: np.ndarray,
    t0: int,
    dt: int,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
    eigencut_rel: Optional[float] = None,
    eigencut_abs: Optional[float] = None,
    prune_rel: Optional[float] = None,
    prune_abs: Optional[float] = None,
) -> Dict[str, object]:
    """
    Accepts either (eigencut_rel/eigencut_abs) or (prune_rel/prune_abs) for eigenvalue pruning on C(t0).
    Returns: {"t0","dt","min_cos","cond_t0","K_kept"}.
    """
    if C.ndim != 3 or C.shape[1] != C.shape[2]:
        raise ValueError(f"C must have shape (T,N,N); got {C.shape}")
    T, N, _ = C.shape

    # Alias CLI names to internal names (CLI passes eigencut_*)
    if prune_rel is None:
        prune_rel = eigencut_rel
    if prune_abs is None:
        prune_abs = eigencut_abs

    t_to_idx = {int(ts[i]): i for i in range(len(ts))}
    if int(t0) not in t_to_idx or int(t0 + dt) not in t_to_idx:
        raise ValueError(f"(t0, t0+dt)=({t0},{t0+dt}) not in times {ts.tolist()}")

    i0 = t_to_idx[int(t0)]
    i1 = t_to_idx[int(t0 + dt)]
    B0 = _sym(C[i0])
    A0 = _sym(C[i1])
    cond_t0 = _cond2(B0)

    W0, K0 = _whiten_from_B(B0, prune_rel=prune_rel, prune_abs=prune_abs, keep_k=keep_k, ridge=ridge)
    u0 = _principal_generalized_eigvec(A0, B0, W0)

    cosines = []
    if int(t0 - 1) in t_to_idx and int(t0 - 1 + dt) in t_to_idx:
        il0 = t_to_idx[int(t0 - 1)]
        il1 = t_to_idx[int(t0 - 1 + dt)]
        Bl = _sym(C[il0]); Al = _sym(C[il1])
        Wl, _ = _whiten_from_B(Bl, prune_rel=prune_rel, prune_abs=prune_abs, keep_k=keep_k, ridge=ridge)
        ul = _principal_generalized_eigvec(Al, Bl, Wl)
        cosines.append(_cosine(u0, ul))
    if int(t0 + 1) in t_to_idx and int(t0 + 1 + dt) in t_to_idx:
        ir0 = t_to_idx[int(t0 + 1)]
        ir1 = t_to_idx[int(t0 + 1 + dt)]
        Br = _sym(C[ir0]); Ar = _sym(C[ir1])
        Wr, _ = _whiten_from_B(Br, prune_rel=prune_rel, prune_abs=prune_abs, keep_k=keep_k, ridge=ridge)
        ur = _principal_generalized_eigvec(Ar, Br, Wr)
        cosines.append(_cosine(u0, ur))

    min_cos = float(min(cosines)) if cosines else 0.0
    return {"t0": int(t0), "dt": int(dt), "min_cos": min_cos, "cond_t0": float(cond_t0), "K_kept": int(K0)}