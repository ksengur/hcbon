"""Code-level check of the guarantees: run the exact calibration routine used in the paper (bon_diffiqa.calibrate,
fixed grid of 401 points in [0,1] plus +inf) on a synthetic population whose true risk is known, many times.

Population (per input): selected score s ~ Beta(2,2) (plays m-hat of the selected candidate); population preference
p = clip(s + N(0, 0.15^2), 0, 1); a panel of r votes, each better / similar / worse with probabilities
(p - h/2, h, 1 - p - h/2), h = 0.2 min(p, 1-p) (vote values 1, 1/2, 0; vote mean is unbiased for p); observed loss
1 - vote mean if deployed. True risk R(lambda) = E[1{s > lambda} (1 - p)] is computed on 2,000,000 fresh draws.

Checks:  CRC   E[R(lambda_hat)] <= alpha        (expectation over calibration sets)
         LTT   P(R(lambda_hat) > alpha) <= delta
for panel sizes r = 1, 3, 10 and calibration sizes n = 100, 500, 3000.

  <cqsr> verify_calibration.py -> pilot/work_bon/verify_calibration.json
"""
from paths import WORK  # noqa: E402
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bon_diffiqa as B  # noqa: E402

GRID = np.concatenate([np.linspace(0, 1, 401), [np.inf]])
ALPHA, DELTA = 0.10, 0.10


def draw(n, r, rng):
    s = rng.beta(2, 2, n)
    p = np.clip(s + rng.normal(0, 0.15, n), 0, 1)
    h = 0.2 * np.minimum(p, 1 - p)
    u = rng.random((n, r))
    votes = np.where(u < p[:, None] - h[:, None] / 2, 1.0, np.where(u < p[:, None] + h[:, None] / 2, 0.5, 0.0))
    return s, p, votes.mean(1)


rng = np.random.default_rng(0)
s_pop, p_pop, _ = draw(2_000_000, 1, rng)
o = np.argsort(s_pop)
s_sorted = s_pop[o]
suf = np.concatenate([np.cumsum((1 - p_pop[o])[::-1])[::-1], [0.0]]) / len(s_pop)
true_R = lambda lam: suf[np.searchsorted(s_sorted, lam, "right")]        # E[1{s > lam}(1 - p)]
dep_of = lambda lam: 1 - np.searchsorted(s_sorted, lam, "right") / len(s_sorted)

out = {}
for r in (1, 3, 10):
    for n in (100, 500, 3000):
        reps = 20000 if n <= 500 else 5000
        R_crc, R_ltt, dep = [], [], []
        for _ in range(reps):
            s, p, vbar = draw(n, r, rng)
            loss = 1 - vbar                                              # observed (vote-level) loss
            R_crc.append(true_R(B.calibrate(s, loss, GRID, ALPHA, DELTA, "crc")))
            t = B.calibrate(s, loss, GRID, ALPHA, DELTA, "ltt")
            R_ltt.append(true_R(t))
            dep.append(float(dep_of(t)))
        R_crc, R_ltt = np.array(R_crc), np.array(R_ltt)
        key = f"r={r}|n={n}"
        out[key] = {"reps": reps, "crc_mean_true_risk": float(R_crc.mean()),
                    "crc_mean_se": float(R_crc.std() / np.sqrt(reps)),
                    "ltt_violation_freq": float(np.mean(R_ltt > ALPHA)), "ltt_mean_true_risk": float(R_ltt.mean()),
                    "ltt_deploy": float(np.mean(dep))}
        print(key, {k: round(v, 4) for k, v in out[key].items()}, flush=True)
out["note"] = f"alpha={ALPHA}, delta={DELTA}; grid: 401 points in [0,1] + inf; routine: bon_diffiqa.calibrate"
json.dump(out, open(WORK + "/verify_calibration.json", "w"), indent=1)
