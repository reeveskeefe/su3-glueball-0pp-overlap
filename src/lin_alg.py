# src/lin_alg.py
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def sorted_eigh(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen-decompose a symmetric matrix, return (evals_desc, evecs_cols).
    """
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def condition_number(A: np.ndarray, eps: float = 0.0) -> float:
    """
    Spectral condition number κ = λ_max / λ_min for SPD-ish A.
    """
    w, _ = sorted_eigh(A)
    w = np.maximum(w, eps)
    top = float(w[0])
    bot = float(w[-1])
    return float(top / bot) if bot > 0.0 and np.isfinite(bot) else float("inf")


def rayleigh_ratio(A: np.ndarray, x: np.ndarray) -> float:
    """
    ρ(A, x) = (x^T A x) / (x^T x), safe for x ≠ 0.
    """
    x = np.asarray(x, dtype=float)
    num = float(x.T @ (0.5 * (A + A.T)) @ x)
    den = float(x.T @ x)
    return num / den if den > 0.0 and np.isfinite(den) else np.nan


def whiten(
    C0: np.ndarray,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return W such that M = W^T C W is well-conditioned in a rank-k subspace.
    """
    lam, V = sorted_eigh(C0)
    if eigencut_abs is None:
        eigencut_abs = float(lam[0]) * float(eigencut_rel)
    mask = lam > max(0.0, float(eigencut_abs))
    if not np.any(mask):
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
    return W, lam_k, V_k


def principal_generalized_eigvec(
    A: np.ndarray,
    B: np.ndarray,
    *,
    eigencut_rel: float = 1e-6,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Solve the top generalized eigenvector of A v = λ B v by whitening B:
      - Compute W ≈ B^{-1/2} on a stable subspace.
      - Form S = W^T A W (symmetric).
      - Take leading eigenvector u of S.
      - Map back v = W u, normalize in the B-inner-product: v^T B v = 1.
    Returns v (shape (N,)).
    """
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    W, _, _ = whiten(B, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge)
    S = W.T @ A @ W
    ew, EV = sorted_eigh(S)
    u = EV[:, 0]  # leading
    v = W @ u
    # B-normalize
    norm2 = float(v.T @ B @ v)
    if norm2 <= 0.0 or not np.isfinite(norm2):
        # Euclidean fallback
        norm2 = float(v.T @ v)
    v = v / np.sqrt(norm2) if norm2 > 0.0 and np.isfinite(norm2) else v
    return v


# --- keep your existing imports and functions above ---

def generalized_eigvals(
    A: np.ndarray,
    B: np.ndarray,
    *,
    eigencut_rel: float = 1e-6,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> np.ndarray:
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    W, _, _ = whiten(B, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge)
    S = W.T @ A @ W
    ew, _ = sorted_eigh(S)
    return ew

# Backwards-compat alias expected by gap_estimator.py
def generalized_eigs(
    A: np.ndarray,
    B: np.ndarray,
    *,
    eigencut_rel: float = 1e-6,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    W, _, _ = whiten(B, eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs, keep_k=keep_k, ridge=ridge)
    S = W.T @ A @ W
    ew, ev = sorted_eigh(S)
    K = W.shape[1]  # Number of kept dimensions from whitening
    return ew, ev, K