import json, subprocess, sys, os, numpy as np
def run(cmd): 
    subprocess.run(cmd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
def test_end_to_end(tmp_path):
    out = tmp_path/"outputs"; out.mkdir()
    run(f"researcher-tools synth --T 32 --ops 6 --energies 0.35,0.90,1.50 --noise 0.001 --seed 42 --out {out}")
    run(f"constructive-cli cert --corr {out}/synth_corr.npz --t0 6 --dt 1 --plateau 7 20 --tstar 7 --fit 8 20 --boundt 14 --eigencut-rel 1e-3 --keep-k 3 --ridge 1e-12 --use-wopt --delta 0.55 --out {out}")
    j = json.load(open(out/'certificate.json'))
    assert j['verdict'] is True
    w = j['plateau_selected']
    assert w['passed'] is True and w['witness']=='wopt'
    assert w['S0_min'] >= 0.70
