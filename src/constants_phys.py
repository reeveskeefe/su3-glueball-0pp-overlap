# src/constants_phys.py
"""
Physics constants, units, and validated inputs for glueball calibration.

This module defines:
- GEV_TO_FM (ħc) and conversions between fm and GeV^{-1}
- A strict Inputs dataclass with validation for all parameters needed by the
  glueball calibration (tau / sqrt(sigma), locality window, KP recursion).
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from math import isfinite

# Natural units: ħc = 0.1973269804 GeV·fm
GEV_TO_FM = 0.1973269804
FM_TO_GEV_INV = 1.0 / GEV_TO_FM  # 1 fm ≈ 5.0677307 GeV^{-1}

REF_RATIO_0PP = 3.405  # m_{0++} / sqrt(sigma) continuum value (dimensionless)

@dataclass(frozen=True)
class Inputs:
    # String tension and geometry
    sigma_phys_GeV2: float   # physical string tension σ in GeV^2
    l_phys_fm: float         # tile width ℓ_phys in fm
    c_area: float            # area-law → tube-cost prefactor in (0,1]

    # KP / RG recursion
    b: int                   # block factor (≥1 integer)
    A: float                 # KP contraction envelope (>0)
    C: float                 # collar loss coefficient in (0,1)
    eta0: float              # seed smallness in (0,1)
    eta_star: float          # target smallness where cluster bounds comfortably hold (e.g. 1e-3)
    c_KP: float              # prefactor in m_lower bound (dimensionless >0)

    # Locality check
    kappa: float             # locality multiplier in (0,1]; require ℓ_phys ≤ κ ξ_{0++}

    @staticmethod
    def load(path: str) -> "Inputs":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        req = ["sigma_phys_GeV2","l_phys_fm","c_area","b","A","C","eta0","eta_star","c_KP","kappa"]
        for k in req:
            if k not in obj:
                raise ValueError(f"Missing '{k}' in {path}")
        x = Inputs(
            sigma_phys_GeV2=float(obj["sigma_phys_GeV2"]),
            l_phys_fm=float(obj["l_phys_fm"]),
            c_area=float(obj["c_area"]),
            b=int(obj["b"]),
            A=float(obj["A"]),
            C=float(obj["C"]),
            eta0=float(obj["eta0"]),
            eta_star=float(obj["eta_star"]),
            c_KP=float(obj["c_KP"]),
            kappa=float(obj["kappa"]),
        )
        _validate(x)
        return x

def _validate(x: Inputs) -> None:
    if not (x.sigma_phys_GeV2 > 0 and isfinite(x.sigma_phys_GeV2)):
        raise ValueError("sigma_phys_GeV2 must be finite and > 0")
    if not (x.l_phys_fm > 0):
        raise ValueError("l_phys_fm must be > 0")
    if not (0 < x.c_area <= 1.0):
        raise ValueError("c_area must be in (0,1]")
    if not (x.b >= 1 and int(x.b) == x.b):
        raise ValueError("b must be integer >= 1")
    if not (x.A > 0):
        raise ValueError("A must be > 0")
    if not (0 < x.C < 1):
        raise ValueError("C must be in (0,1)")
    if not (0 < x.eta0 < 1):
        raise ValueError("eta0 must be in (0,1)")
    if not (0 < x.eta_star < 1):
        raise ValueError("eta_star must be in (0,1)")
    if not (x.c_KP > 0):
        raise ValueError("c_KP must be > 0")
    if not (0 < x.kappa <= 1):
        raise ValueError("kappa must be in (0,1]")

def fm_to_GeV_inv(fm: float) -> float:
    return fm * FM_TO_GEV_INV

def GeV_inv_to_fm(x: float) -> float:
    return x * GEV_TO_FM

def sqrt_sigma(sigma_GeV2: float) -> float:
    return sigma_GeV2 ** 0.5