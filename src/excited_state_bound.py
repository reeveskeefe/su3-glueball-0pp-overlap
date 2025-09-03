from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from math import exp
from .gap_estimator import estimate_delta_from_multit_gevp

def run_excited_state_bound(
    C: np.ndarray,
    ts: np.ndarray,
    *,
    t0: int,
    t: int,
    delta_override: Optional[float] = None,
    c_pref: float = 1.0,
    eigencut_rel: float = 1e-4,
    eigencut_abs: Optional[float] = None,
    keep_k: Optional[int] = 3,
    ridge: float = 0.0,
    gap_q0: float = 0.10,
    gap_q1: float = 0.90,
) -> Dict:
    if t <= t0:
        raise ValueError("t must be > t0 for the bound")

    if delta_override is None:
        est = estimate_delta_from_multit_gevp(
            C, t0,
            eigencut_rel=eigencut_rel, eigencut_abs=eigencut_abs,
            keep_k=keep_k, ridge=ridge, max_points=5,
            q0=gap_q0, q1=gap_q1
        )
        Delta = float(est["Delta"])
        delta_meta = est
    else:
        Delta = float(delta_override)
        delta_meta = {"override": True, "Delta": Delta}

    dt = int(ts[t] - ts[t0])
    bound = float(c_pref * exp(-max(Delta, 0.0) * dt))
    return {
        "t0": int(t0),
        "t": int(t),
        "dt": int(dt),
        "Delta_used": float(Delta),
        "c_pref": float(c_pref),
        "bound_1_minus_S0": float(bound),
        "delta_meta": delta_meta,
    }