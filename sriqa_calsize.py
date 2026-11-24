"""SRIQA-Bench: effect of the calibration-set size (n_cal = 20, 30, 50, 70 of 100 contents; the rest is test).
CRC (expected-risk control) and LTT (high-probability control), 500 random splits each; grid from the unlabelled
test part. Splits re-divide the same 100 contents, so they are not independent datasets."""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs_v3 as M
W = M.W
out = {}
for st in ("A_ref_RealESRNet", "B_ref_GT"):
    D = dict(np.load(W + f"sriqa_scores_{st}.npz")); p = D.pop("p"); C = len(p); ar = np.arange(C)
    for q in ("hcbon", "afine", "liqe"):
        k = D[q].argmax(1); s, loss = D[q][ar, k], 1 - p[ar, k]
        for n in (20, 30, 50, 70):
            for alpha in (0.10, 0.20):
                for mode in ("crc", "ltt"):
                    rng = np.random.default_rng(0); r, d = [], []
                    for _ in range(500):
                        perm = rng.permutation(C); cal, te = perm[:n], perm[n:]
                        g = s[te] > M.thresholds(s, loss, cal, te, alpha, mode=mode, name=q)
                        r.append(np.mean(g * loss[te])); d.append(g.mean())
                    out[f"{st}|{q}|{n}|{alpha}|{mode}"] = {"risk": float(np.mean(r)), "viol": float(np.mean(np.array(r) > alpha)),
                                                            "deploy": float(np.mean(d))}
json.dump(out, open(W + "sriqa_calsize.json", "w"), indent=1)
for k, v in out.items():
    if "|hcbon|" in k or "|afine|" in k:
        print(f"{k:38s} risk={v['risk']:.3f} viol={v['viol']:.2f} dep={v['deploy']:.2f}")
