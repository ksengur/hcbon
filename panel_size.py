"""Empirical sensitivity to the calibration panel size r on DiffIQA test (review request).
Calibration losses use the mean of r in {1,2,3} randomly chosen votes per output; the risk on the held-out half is
always evaluated with the full three-vote mean (the best available unbiased estimate of the population preference).
Same selector (HC-BoN, c=0), fixed grid, alpha, delta and 100 content-disjoint splits as Table I."""
from paths import WORK  # noqa: E402
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from bon_diffiqa import calibrate, GRID01
W = WORK
T, Pd = np.load(f"{W}/table_test.npz"), np.load(f"{W}/pred3_test.npz")
p, votes, m = T["p"], T["votes"], Pd["m"]
C = p.shape[0]; ar = np.arange(C)
k = np.argmax(m, 1); s = m[ar, k]; loss_full = 1 - p[ar, k]; v_sel = votes[ar, k]          # [C,3]
rng = np.random.default_rng(0)
out = {}
for alpha in (0.10, 0.20):
    for r in (1, 2, 3):
        for mode in ("ltt", "crc"):
            R, D = [], []
            for rep in range(100):
                perm = rng.permutation(C); cal, te = perm[: C // 2], perm[C // 2:]
                pick = np.argsort(rng.random((len(cal), 3)), 1)[:, :r]
                vbar = np.take_along_axis(v_sel[cal], pick, 1).mean(1)
                t = calibrate(s[cal], 1 - vbar, GRID01, alpha, 0.1, mode)
                g = s[te] > t
                R.append(float(np.mean(g * loss_full[te]))); D.append(float(g.mean()))
            key = f"a{alpha:.2f}|r{r}|{mode}"
            out[key] = {"risk": [np.mean(R), np.std(R)], "deploy": [np.mean(D), np.std(D)],
                        "viol": float(np.mean(np.array(R) > alpha))}
            print(key, {a: (round(b[0], 4), round(b[1], 4)) if isinstance(b, list) else b for a, b in out[key].items()})
json.dump(out, open(f"{W}/panel_size.json", "w"), indent=1)
