from __future__ import annotations
from dataclasses import dataclass
from math import log, log2, ceil, isfinite
from typing import List

@dataclass(frozen=True)
class KPResult:
    eta_seq: List[float]
    sum_eta: float
    collar_product: float
    k_star_iter: int
    k_star_cf: int

def iterate_kp(eta0: float, A: float, C: float, eta_star: float, max_steps: int = 256) -> KPResult:
    if not (0 < eta0 < 1 and A > 0 and 0 < C < 1 and 0 < eta_star < 1):
        raise ValueError("Invalid KP parameters.")

    seq = [float(eta0)]
    prod = 1.0
    s = float(eta0)
    k_star_iter = -1

    for k in range(max_steps):
        term = 1.0 - C * seq[-1]
        prod = 0.0 if term <= 0.0 else prod * term

        if k_star_iter < 0 and seq[-1] <= eta_star:
            k_star_iter = k

        nxt = float(A * (seq[-1] ** 2))
        seq.append(nxt)
        s += nxt
        if nxt == 0.0 or not isfinite(nxt):
            break

    k_star_cf = _kstar_closed_form(eta0, A, eta_star)
    return KPResult(eta_seq=seq, sum_eta=s, collar_product=prod, k_star_iter=k_star_iter, k_star_cf=k_star_cf)

def _kstar_closed_form(eta0: float, A: float, eta_star: float) -> int:
    num = log(max(1e-300, (1.0 / eta_star) * A))
    den = log(max(1e-300, (1.0 / eta0) * A))
    if den <= 0.0:
        return 0
    val = num / den
    return int(max(0, ceil(log2(max(1.0, val)))))
