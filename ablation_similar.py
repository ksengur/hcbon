"""Ablation: count 'similar' as harm (p' = share of 'better' votes) instead of 1/2. Same selection (m-hat, c=0),
loss 1 - p' in calibration and evaluation; LTT, delta = 0.1, 100 content-disjoint splits of DiffIQA test."""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs_v3 as M
pb = (M.votes == 1).mean(2)                                    # similar counted as harm
out = {}
for alpha in (0.10, 0.20, 0.30):
    row = {}
    for q in ("hcbon", "afine", "clipiqa", "liqe"):
        k = M.S[q].argmax(1)
        s = M.S[q][M.ar, k]
        for name, P in (("half", M.p), ("harm", pb)):
            loss = 1 - P[M.ar, k]
            r, d = [], []
            for cal, te, _ in M.splits(100):
                t = M.thresholds(s, loss, cal, te, alpha, name=q)
                g = s[te] > t
                r.append(np.mean(g * loss[te])); d.append(g.mean())
            row[f"{q}|{name}"] = {"risk": float(np.mean(r)), "viol": float(np.mean(np.array(r) > alpha)), "deploy": float(np.mean(d))}
    out[str(alpha)] = row
json.dump(out, open(M.W + "ablation_similar.json", "w"), indent=1)
for a, row in out.items():
    print("alpha", a)
    for k, v in row.items(): print(f"  {k:14s} risk={v['risk']:.3f} viol={v['viol']:.2f} deploy={v['deploy']:.3f}")
print("mean p (half) %.3f  mean p' (harm) %.3f" % (M.p.mean(), pb.mean()))
