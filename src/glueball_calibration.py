# src/glueball_calibration.py
"""
Glueball 0++ calibration:
- Ratio sanity: rho := tau / sqrt(sigma) ≤ REF_RATIO_0PP (≈ 3.405)
- Locality window: ℓ_phys ≤ κ * ξ_{0++}, with ξ_{0++} = ħc / m_{0++}
- KP bound: compute k_star, then m_lower ≥ c_KP / (b^{k_star} ℓ_phys)

Outputs a JSON-friendly dict and a concise Markdown report snippet.
"""

from __future__ import annotations
from typing import Dict, Any
from math import exp

from .constants_phys import (
    Inputs, fm_to_GeV_inv, GeV_inv_to_fm, sqrt_sigma, REF_RATIO_0PP, GEV_TO_FM
)
from .kp_recursion import iterate_kp

def compute_calibration(inp: Inputs) -> Dict[str, Any]:
    # √σ and τ (GeV)
    root_sigma = sqrt_sigma(inp.sigma_phys_GeV2)  # GeV
    l_GeV_inv = fm_to_GeV_inv(inp.l_phys_fm)      # GeV^{-1}
    tau_GeV = inp.c_area * inp.sigma_phys_GeV2 * l_GeV_inv  # GeV
    rho = float(tau_GeV / root_sigma)  # dimensionless

    # Reference m_{0++}
    m0pp_ref = float(REF_RATIO_0PP * root_sigma)  # GeV
    xi_GeV_inv = float(1.0 / m0pp_ref)            # GeV^{-1}
    xi_fm = GeV_inv_to_fm(xi_GeV_inv)             # fm

    # Locality requirement
    locality_ok = bool(inp.l_phys_fm <= inp.kappa * xi_fm)

    # KP recursion
    kp = iterate_kp(eta0=inp.eta0, A=inp.A, C=inp.C, eta_star=inp.eta_star, max_steps=256)
    k_star = kp.k_star_iter if kp.k_star_iter >= 0 else kp.k_star_cf
    # Mass lower bound from plateau scale
    m_lower = float(inp.c_KP / ( (inp.b ** k_star) * l_GeV_inv ))  # GeV
    ratio_lower = float(m_lower / root_sigma)  # m_lower / √σ (dimensionless)

    # Basic acceptance logic (conservative)
    checks = {
        "ratio_ok": rho <= (REF_RATIO_0PP + 1e-12),
        "locality_ok": locality_ok,
        "kp_sum_finite": kp.sum_eta < 1e9,
        "kp_collar_positive": kp.collar_product > 0.0,
        "kp_kstar_defined": k_star >= 0,
        # we *prefer* m_lower ≤ m0pp_ref for comfortable slack
        "kp_lower_not_exceed_ref": m_lower <= m0pp_ref + 1e-12,
    }
    passed = all(checks.values())

    return {
        "inputs": {
            "sigma_phys_GeV2": inp.sigma_phys_GeV2,
            "l_phys_fm": inp.l_phys_fm,
            "c_area": inp.c_area,
            "b": inp.b,
            "A": inp.A,
            "C": inp.C,
            "eta0": inp.eta0,
            "eta_star": inp.eta_star,
            "c_KP": inp.c_KP,
            "kappa": inp.kappa,
        },
        "computed": {
            "sqrt_sigma_GeV": root_sigma,
            "tau_GeV": tau_GeV,
            "rho_tau_over_sqrtsigma": rho,
            "m0pp_ref_GeV": m0pp_ref,
            "xi_GeV_inv": xi_GeV_inv,
            "xi_fm": xi_fm,
            "locality_limit_fm": inp.kappa * xi_fm,
            "kp": {
                "eta_sequence": kp.eta_seq,
                "sum_eta": kp.sum_eta,
                "collar_product": kp.collar_product,
                "k_star_iter": kp.k_star_iter,
                "k_star_cf": kp.k_star_cf,
                "k_star_used": k_star,
            },
            "m_lower_GeV": m_lower,
            "m_lower_over_sqrtsigma": ratio_lower,
            "ref_ratio_0pp_over_sqrtsigma": REF_RATIO_0PP,
        },
        "checks": checks,
        "pass": passed,
    }

def to_markdown(cert: Dict[str, Any]) -> str:
    I = cert["inputs"]; C = cert["computed"]; K = C["kp"]; CH = cert["checks"]
    def yn(b: bool) -> str: return "✅ yes" if b else "❌ no"
    md = []
    md.append("# Glueball 0++ Calibration\n")
    md.append("## Ratio sanity\n")
    md.append(f"- √σ = **{C['sqrt_sigma_GeV']:.6f} GeV**, τ = **{C['tau_GeV']:.6f} GeV**, ρ := τ/√σ = **{C['rho_tau_over_sqrtsigma']:.6f}**\n")
    md.append(f"- Reference m₀⁺⁺/√σ = **{C['ref_ratio_0pp_over_sqrtsigma']}** → ratio ok? **{yn(CH['ratio_ok'])}**\n")
    md.append("\n## Locality window\n")
    md.append(f"- ξ₀⁺⁺ = **{C['xi_fm']:.6f} fm**; κ = **{I['kappa']}** ⇒ κ ξ = **{C['locality_limit_fm']:.6f} fm**\n")
    md.append(f"- ℓ_phys = **{I['l_phys_fm']:.6f} fm** → locality ok? **{yn(CH['locality_ok'])}**\n")
    md.append("\n## KP / RG lower bound\n")
    md.append(f"- k* (iter) = **{K['k_star_iter']}**, k* (closed-form) = **{K['k_star_cf']}**, used = **{K['k_star_used']}**\n")
    md.append(f"- m_lower = **{C['m_lower_GeV']:.6f} GeV** (i.e. **{C['m_lower_over_sqrtsigma']:.6f} × √σ**), ≤ m₀⁺⁺(ref)? **{yn(CH['kp_lower_not_exceed_ref'])}**\n")
    md.append(f"- Ση_k finite? **{yn(CH['kp_sum_finite'])}**; collar product positive? **{yn(CH['kp_collar_positive'])}**\n")
    md.append("\n## Verdict\n")
    md.append(f"- All calibration checks pass? **{yn(cert['pass'])}**\n")
    md.append("\n---\n*Deterministic; no RNG; units and guards enforced.*\n")
    return "".join(md)

def run_from_json(constants_path: str, out_json: str | None = None, out_md: str | None = None) -> Dict[str, Any]:
    inp = Inputs.load(constants_path)
    cert = compute_calibration(inp)
    if out_json:
        import os, json
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2, sort_keys=True)
    if out_md:
        import os
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(to_markdown(cert))
    return cert