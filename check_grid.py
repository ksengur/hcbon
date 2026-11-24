"""Validity check: calibrate with a grid fixed in advance (401 points on [0,1], independent of all data) instead of
quantiles of the unlabelled test half, which depend (weakly) on the test point and break exact CRC exchangeability."""
import json, os, sys
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs_v3 as M
FIX = np.concatenate([np.linspace(0, 1, 401), [np.inf]])
def thr(s, loss, cal, alpha, mode, delta=0.1):
    sc, lc = s[cal], loss[cal]; o = np.argsort(sc); ss = sc[o]
    suf = np.concatenate([np.cumsum(lc[o][::-1])[::-1], [0.0]]); n = len(sc)
    R = suf[np.searchsorted(ss, FIX, "right")] / n
    if mode == "crc":
        ok = n / (n + 1) * R + 1 / (n + 1) <= alpha
        return FIX[np.argmax(ok)] if ok.any() else np.inf
    pv = np.where(R >= alpha, 1.0, np.exp(-2 * n * np.clip(alpha - R, 0, None) ** 2))
    rej = (pv <= delta)[::-1]; k = len(rej) if rej.all() else int(np.argmax(~rej))
    return FIX[len(FIX) - k] if k else np.inf
k = M.m_hat.argmax(1); s, loss = M.m_hat[M.ar, k], 1 - M.p[M.ar, k]
out = {}
for alpha in (0.05, 0.10, 0.20, 0.30):
    for mode in ("ltt", "crc"):
        r, d = [], []
        for cal, te, _ in M.splits(100):
            g = s[te] > thr(s, loss, cal, alpha, mode); r.append(np.mean(g * loss[te])); d.append(g.mean())
        out[f"{alpha}|{mode}"] = {"risk": float(np.mean(r)), "viol": float(np.mean(np.array(r) > alpha)), "deploy": float(np.mean(d))}
        print(alpha, mode, {k_: round(v, 4) for k_, v in out[f"{alpha}|{mode}"].items()})
json.dump(out, open(M.W + "check_fixed_grid.json", "w"), indent=1)
