#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import numpy as np
from src.io_utils import save_json
from src.plateau import s0_series_via_gevp
from src.cross_estimator import run_cross_estimator

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _synth_corr(T: int, ops: int, energies: list[float], noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = np.arange(T, dtype=int)
    N = ops
    C = np.zeros((T, N, N), dtype=float)
    # random overlaps for each state into N operators
    S = np.stack([rng.normal(0, 1, size=N) for _ in energies], axis=0)  # (K,N)
    for t in range(T):
        Ct = np.zeros((N, N), dtype=float)
        for En, sn in zip(energies, S):
            amp = np.exp(-En * t)
            Ct += amp * np.outer(sn, sn)
        Ct += noise * np.eye(N)
        C[t] = 0.5 * (Ct + Ct.T)
    return C, ts

def cmd_synth(args) -> int:
    _ensure_dir(args.out)
    C, ts = _synth_corr(args.T, args.ops, [float(x) for x in args.energies.split(",")], args.noise, args.seed)
    np.savez(os.path.join(args.out, "synth_corr.npz"), C=C, ts=ts)
    meta = dict(T=args.T, ops=args.ops, energies=[float(x) for x in args.energies.split(",")], noise=args.noise, seed=args.seed, file=os.path.join(args.out, "synth_corr.npz"))
    print(json.dumps(meta, indent=2))
    save_json(meta, os.path.join(args.out, "synth_meta.json"))
    return 0

def cmd_charts(args) -> int:
    import matplotlib.pyplot as plt
    data = np.load(args.corr)
    C, ts = data["C"], data["ts"]
    t0 = args.t0
    tmin, tmax = args.tmin, args.tmax
    t_range = range(max(t0+1, tmin), min(tmax, C.shape[0]-1)+1)

    s0_vals, K = s0_series_via_gevp(
        C, t0, t_range,
        eigencut_rel=args.eigencut_rel, eigencut_abs=None, keep_k=args.keep_k, ridge=args.ridge
    )
    plt.figure()
    plt.plot(ts[list(t_range)], s0_vals, marker="o")
    plt.xlabel("t")
    plt.ylabel("S0 (GEVP)")
    plt.title(f"S0 via GEVP (t0={t0}, K={K}, ridge={args.ridge}, cut={args.eigencut_rel})")
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(args.corr), "s0_gevp.png")
    plt.savefig(out_png, dpi=180)
    print(json.dumps({"png": out_png, "K_kept": K, "S0_min": float(np.min(s0_vals)), "S0_max": float(np.max(s0_vals))}, indent=2))
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description="Independent Research Tools for SU(3) 0++ experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("synth", help="Generate synthetic correlators.")
    sp.add_argument("--T", type=int, required=True)
    sp.add_argument("--ops", type=int, required=True)
    sp.add_argument("--energies", type=str, required=True)
    sp.add_argument("--noise", type=float, default=0.0)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", type=str, default="outputs")
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("charts", help="Produce plateau and C_opt charts for a correlator file.")
    sp.add_argument("--corr", required=True)
    sp.add_argument("--t0", type=int, required=True)
    sp.add_argument("--tmin", type=int, required=True)
    sp.add_argument("--tmax", type=int, required=True)
    sp.add_argument("--eigencut-rel", type=float, default=1e-4)
    sp.add_argument("--keep-k", type=int, default=None)
    sp.add_argument("--ridge", type=float, default=0.0)
    sp.set_defaults(func=cmd_charts)

    args = p.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())