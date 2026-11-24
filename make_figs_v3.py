"""All figures of the v3 paper (draft/figs/*.png + .pdf).

  fig_pipeline     method overview (wide)
  fig_examples     DiffIQA test examples: reference + 6 outputs, human preference, picks and the gate (wide)
  fig_bon_diffiqa  (a) best-of-N curves, (b) deployment vs alpha, (c) per-split realised risk (wide)
  fig_majority     majority-vote bias phi_r(p) - p (single column)
  fig_sriqa        (a) proxy efficiency vs N, (b) zero-shot transfer efficiency (wide)
  fig_ablation     (a) predictor inputs (CV), (b) per-split deployment gain over A-FINE (single column)

Only DiffIQA images are shown (academic-use licence); SRIQA-Bench images are not reproduced.
  <cqsr python> make_figs_v3.py
"""
from paths import DATA, FIGS, WORK  # noqa: E402
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = WORK + "/"
OUT = FIGS + "/"

INK, MUTED, GRID, SURF = "#0b0b0b", "#6b6a64", "#e4e3df", "#ffffff"
COL = {"hcbon": "#2a78d6", "afine": "#8a63d2", "liqe": "#eb6834", "musiq": "#1baf7a", "clipiqa": "#eda100",
       "dists": "#e87ba4", "topiq_nr": "#3fa7c9", "maniqa": "#9b7b4f", "lpips": "#b8b6ae"}
MK = {"hcbon": "o", "afine": "P", "liqe": "s", "musiq": "^", "clipiqa": "D", "dists": "v", "topiq_nr": "X",
      "maniqa": "h", "lpips": "<"}
NAME = {"hcbon": "HC-BoN", "afine": "A-FINE", "liqe": "LIQE", "musiq": "MUSIQ", "clipiqa": "CLIP-IQA",
        "dists": "DISTS", "topiq_nr": "TOPIQ", "maniqa": "MANIQA", "lpips": "LPIPS"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.titlesize": 9, "legend.fontsize": 7.5, "mathtext.fontset": "stix"})


def save(fig, name):
    fig.savefig(OUT + name + ".png", dpi=450, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(OUT + name)


# ------------------------------------------------------------------ data
T = np.load(W + "table_test.npz")
p, votes, paths = T["p"], T["votes"], T["paths"]
C = len(p)
ar = np.arange(C)
m_hat = np.load(W + "pred3_test.npz")["m"]
X = dict(np.load(W + "proxies_test.npz"))
X.pop("idx")
X["afine"] = np.load(W + "afine_test.npz")["afine"]
S = {"hcbon": m_hat} | X
maj = (np.sum(votes > 0.5, 2) > np.sum(votes < 0.5, 2)).astype(float)


GRID01 = np.concatenate([np.linspace(0, 1, 401), [np.inf]])
_val = dict(np.load(W + "proxies_val.npz"))
_val.pop("idx")
_val["afine"] = np.load(W + "afine_val.npz")["afine"]
VALGRID = {q: np.unique(np.concatenate([np.quantile(v.max(1), np.linspace(0, 1, 401)), [np.inf]]))
           for q, v in _val.items()}                     # fixed before calibration, from the training split only
TEST_GRID = False                                        # True: earlier grid (quantiles of the unlabelled test half)


def thresholds(s, loss, cal, te, alpha, delta=0.1, mode="ltt", name=None, grid=None):
    """Vectorised version of bon_diffiqa.calibrate (identical result). The grid is fixed before calibration and
    independent of calibration and test data: explicit `grid`, else the training-split grid of selector `name`,
    else [0, 1] in 401 steps for predicted preferences."""
    if TEST_GRID:
        grid = np.unique(np.concatenate([np.quantile(s[te], np.linspace(0, 1, 400)), [np.inf]]))
    elif grid is None:
        if name in VALGRID:
            grid = VALGRID[name]
        else:
            assert s.min() >= 0 and s.max() <= 1, f"no fixed grid for selector {name}"
            grid = GRID01
    sc, lc = s[cal], loss[cal]
    o = np.argsort(sc)
    ss, suf = sc[o], np.concatenate([np.cumsum(lc[o][::-1])[::-1], [0.0]])
    n = len(sc)
    R = suf[np.searchsorted(ss, grid, "right")] / n
    if mode == "crc":
        ok = n / (n + 1) * R + 1 / (n + 1) <= alpha
        return grid[np.argmax(ok)] if ok.any() else np.inf
    pv = np.where(R >= alpha, 1.0, np.exp(-2 * n * np.clip(alpha - R, 0, None) ** 2))
    rej = (pv <= delta)[::-1]
    k = len(rej) if rej.all() else int(np.argmax(~rej))
    return grid[len(grid) - k] if k else np.inf


def splits(reps=100, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        perm = rng.permutation(C)
        cal, te = perm[: C // 2], perm[C // 2:]
        kr = rng.integers(0, 6, C)                                   # as in evaluate (selection-unaware)
        yield cal, te, kr


# ------------------------------------------------------------------ fig_pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x, y, w, h, txt, fc="#f4f3ef", ec=MUTED, bold=False, size=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x0, y0, x1, y1, txt=None, col=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col,
                                     linestyle=ls))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.9, txt, ha="center", va="bottom", fontsize=6.5, color=col)

    # thumbnails of one test content (example (a) of fig_examples)
    c = 5755
    ims = [Image.open(f).convert("RGB").resize((128, 128)) for f in paths[c]]
    ax.imshow(np.asarray(ims[6]), extent=(0.3, 8.3, 18.0, 26.0))
    ax.text(4.3, 17.2, r"reference $y_0$", ha="center", va="top", fontsize=6.6)
    for j in range(3):
        x0 = 12.6 + j * 1.9
        ax.imshow(np.asarray(ims[j]), extent=(x0, x0 + 7.0, 19.0 - j * 1.2, 26.0 - j * 1.2), zorder=3 + j)
    ax.text(17.4, 15.8, r"candidates $y_1,\dots,y_N$", ha="center", va="top", fontsize=6.6)
    arrow(8.6, 22.3, 12.3, 22.3)
    ax.text(10.4, 26.6, "generator", ha="center", fontsize=6.3, color=MUTED)
    nl = "\n"
    box(25.0, 18.2, 18.0, 8.0, nl.join(["IQA scores", "MUSIQ, CLIP-IQA,", "TOPIQ, LIQE, LPIPS,",
                                        "DISTS, A-FINE"]), size=6.2)
    arrow(22.3, 22.2, 24.7, 22.2)
    box(46.0, 18.2, 12.5, 8.0, nl.join(["preference", "predictor", r"$\hat m_k\approx p_k$"]), fc="#e3edf9",
        ec=COL["hcbon"], size=6.6)
    arrow(43.3, 22.2, 45.7, 22.2)
    box(61.5, 18.2, 12.0, 8.0, nl.join(["select", r"$\hat k=\arg\max_k\,\hat m_k$"]), fc="#e3edf9", ec=COL["hcbon"],
        size=6.6)
    arrow(58.8, 22.2, 61.2, 22.2)
    box(76.5, 18.2, 9.5, 8.0, nl.join(["gate", r"$\hat m_{\hat k}>\hat\lambda$ ?"]), fc="#e3edf9", ec=COL["hcbon"],
        size=6.6)
    arrow(73.8, 22.2, 76.2, 22.2)
    box(91.0, 23.0, 8.5, 3.4, r"show $y_{\hat k}$", fc="#ffffff", size=6.4)
    box(91.0, 17.6, 8.5, 3.4, r"keep $y_0$", fc="#ffffff", size=6.4)
    arrow(86.3, 23.6, 90.7, 24.7)
    arrow(86.3, 20.8, 90.7, 19.3)
    ax.text(88.0, 25.3, "yes", fontsize=6.2, color=MUTED, ha="center")
    ax.text(88.0, 17.6, "no", fontsize=6.2, color=MUTED, ha="center")
    # calibration lane
    ax.add_patch(Rectangle((23.5, 0.4), 76.3, 12.6, fc="#faf9f6", ec=GRID, lw=0.8))
    ax.text(24.4, 12.3, "calibration on a labelled set (once)", fontsize=6.4, color=MUTED, va="top", style="italic")
    box(25.0, 1.8, 21.5, 6.8, nl.join([r"panels of $r$ raters per output", r"vote mean $\bar v_k$, unbiased for $p_k$"]),
        size=6.0)
    box(50.0, 1.8, 21.5, 6.8, nl.join(["loss of the realised decision",
                                       r"$\tilde L_i(\lambda)=g_\lambda(x_i)\,(1-\bar v_{i\hat k})$"]), size=6.0)
    box(75.0, 1.8, 24.0, 6.8, nl.join(["Learn-then-Test, fixed grid",
                                       r"$\mathbb{P}\{R(\hat\lambda)\leq\alpha\}\geq 1-\delta$"]),
        size=6.0, fc="#e3edf9", ec=COL["hcbon"])
    arrow(46.8, 5.2, 49.7, 5.2)
    arrow(71.8, 5.2, 74.7, 5.2)
    arrow(81.25, 8.9, 81.25, 17.9, col=COL["hcbon"])
    ax.text(82.0, 13.2, r"$\hat\lambda$", fontsize=7.5, color=COL["hcbon"], va="center")
    save(fig, "fig_pipeline")


# ------------------------------------------------------------------ fig_examples
def fig_examples():
    cal, te, _ = next(splits())
    lam = {}
    for q in ("hcbon", "afine"):
        k = S[q].argmax(1)
        lam[q] = thresholds(S[q][ar, k], 1 - p[ar, k], cal, te, 0.10, name=q)
    pick = {q: S[q].argmax(1) for q in ("hcbon", "afine", "musiq")}
    dep = {q: S[q][ar, pick[q]] > lam[q] for q in ("hcbon", "afine")}
    ph = p[ar, pick["hcbon"]]
    best = p.max(1)
    tes = np.zeros(C, bool)
    tes[te] = True
    conf = m_hat.max(1)
    cases = [
        ("(a) NR proxy picks an output viewers reject; HC-BoN deploys the preferred one",
         tes & dep["hcbon"] & (ph >= 5 / 6) & (p[ar, pick["musiq"]] <= 1 / 6)),
        ("(b) No output beats the reference: HC-BoN keeps it, the A-FINE selector deploys",
         tes & ~dep["hcbon"] & dep["afine"] & (best <= 1 / 3)),
        ("(c) HC-BoN and A-FINE disagree; viewers side with HC-BoN",
         tes & dep["hcbon"] & (ph >= 5 / 6) & (p[ar, pick["afine"]] <= 1 / 3)),
        ("(d) Failure: HC-BoN deploys an output viewers find worse than the reference",
         tes & dep["hcbon"] & (ph <= 1 / 6)),
    ]
    rows = []
    for title, mask in cases:
        idx = np.where(mask)[0]
        c = idx[np.argmax(conf[idx])] if title.startswith(("(a)", "(c)", "(d)")) else idx[np.argmin(conf[idx])]
        rows.append((title, c))
    cw = 7
    fig, axes = plt.subplots(len(rows), cw, figsize=(7.16, 1.38 * len(rows) + 0.15),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.62})
    for r, (title, c) in enumerate(rows):
        ims = [np.asarray(Image.open(f).convert("RGB"), np.float32) for f in paths[c]]
        st = np.stack(ims[:6])
        var = np.abs(st - st.mean(0)).mean((0, -1))                         # where the outputs differ
        h = 200
        sm = np.cumsum(np.cumsum(var, 0), 1)
        best_sc, y0, x0 = -1, 0, 0
        for yy in range(0, 512 - h + 1, 16):
            for xx in range(0, 512 - h + 1, 16):
                v = sm[yy + h - 1, xx + h - 1] - (sm[yy - 1, xx + h - 1] if yy else 0) - \
                    (sm[yy + h - 1, xx - 1] if xx else 0) + (sm[yy - 1, xx - 1] if yy and xx else 0)
                if v > best_sc:
                    best_sc, y0, x0 = v, yy, xx
        order = [6] + list(range(6))
        for j, k in enumerate(order):
            ax = axes[r, j]
            ax.imshow(ims[k][y0:y0 + h, x0:x0 + h].astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(GRID)
                sp.set_linewidth(0.6)
            if k == 6:
                lab = "reference $y_0$" + ("  (kept)" if not dep["hcbon"][c] else "")
                ax.set_xlabel(lab, fontsize=6.6, labelpad=2)
                if not dep["hcbon"][c]:
                    for sp in ax.spines.values():
                        sp.set_color(COL["hcbon"])
                        sp.set_linewidth(2.2)
                continue
            vt = votes[c, k]
            nb, ns, nw = int((vt == 1).sum()), int((vt == 0.5).sum()), int((vt == 0).sum())
            star = "*" if p[c, k] == best[c] else ""
            ax.set_xlabel(f"$p$={p[c, k]:.2f}{star}  ({nb}/{ns}/{nw})", fontsize=6.4, labelpad=2)
            if k == pick["hcbon"][c]:
                for sp in ax.spines.values():
                    sp.set_color(COL["hcbon"])
                    sp.set_linewidth(2.2)
                    sp.set_linestyle("-" if dep["hcbon"][c] else (0, (2, 1.5)))
            tags = [("H", COL["hcbon"])] if k == pick["hcbon"][c] else []
            tags += [("A", COL["afine"])] if k == pick["afine"][c] else []
            tags += [("M", COL["musiq"])] if k == pick["musiq"][c] else []
            for t_i, (t, col) in enumerate(tags):
                ax.text(0.04 + 0.2 * t_i, 0.95, t, transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                        color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"))
        axes[r, 0].text(0, 1.06, title, transform=axes[r, 0].transAxes, ha="left", va="bottom", fontsize=7.2,
                        color=INK)
    save(fig, "fig_examples")
    return [int(c) for _, c in rows]


# ------------------------------------------------------------------ fig_bon_diffiqa
def fig_bon(R1):
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    # (a)
    a = ax[0]
    Ns = list(range(1, 7))
    orc = [R1["oracle_p"][str(n)] for n in Ns]
    a.plot(Ns, orc, color=MUTED, lw=1.1, ls="--")
    a.text(6.1, orc[-1], "panel oracle\n(same votes)", color=MUTED, va="center", fontsize=6.8)
    # held-out panel oracle: select by two raters, evaluate with the third (unbiased for the selection)
    rng_o = np.random.default_rng(3)
    ho = []
    for n in Ns:
        vals = []
        for _ in range(10):
            sub = np.argsort(rng_o.random((C, 6)), 1)[:, :n]
            vs = np.take_along_axis(votes, sub[:, :, None], 1)
            for j in range(3):
                oth = [i for i in range(3) if i != j]
                kk = (vs[:, :, oth].mean(2) + rng_o.random(vs.shape[:2]) * 1e-6).argmax(1)
                vals.append(vs[ar, kk, j].mean())
        ho.append(float(np.mean(vals)))
    a.plot(Ns, ho, color=INK, lw=1.1, ls=":")
    a.text(6.1, ho[-1] - 0.006, "2-rater panel\n(held-out)", color=INK, va="center", fontsize=6.8)
    for k in ["hcbon", "afine", "liqe", "musiq", "clipiqa", "dists"]:
        y = [R1["goodhart"][k][str(n)]["p_selected"] for n in Ns]
        a.plot(Ns, y, color=COL[k], lw=1.5, marker=MK[k], ms=3.5, mec=SURF, mew=0.6, label=NAME[k], zorder=3)
    a.axhline(orc[0], color=GRID, lw=1, zorder=0)
    a.set_xlabel("candidates $N$")
    a.set_ylabel("preference of selected output")
    a.set_xticks(Ns)
    a.set_xlim(0.8, 8.0)
    a.grid(axis="y", color=GRID, lw=0.6)
    pass  # shared legend drawn in panel (b)
    a.set_ylim(0.47, 0.75)
    a.set_title("(a) Best-of-$N$ selection", loc="left")
    # (b) deployment vs alpha (LTT)
    b = ax[1]
    alphas = np.round(np.arange(0.05, 0.301, 0.025), 3)
    sp = list(splits(50))
    for q in ["hcbon", "afine", "liqe", "musiq", "clipiqa", "dists"]:
        k = S[q].argmax(1)
        s, loss = S[q][ar, k], 1 - p[ar, k]
        dep = [np.mean([np.mean(s[te] > thresholds(s, loss, cal, te, al, name=q)) for cal, te, _ in sp]) for al in alphas]
        b.plot(alphas, dep, color=COL[q], lw=1.5, marker=MK[q], ms=3, mec=SURF, mew=0.6, label=NAME[q])
    b.set_xlabel("guaranteed dissatisfaction $\\alpha$")
    b.set_ylabel("fraction of images enhanced")
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_title("(b) Enhancement at a guaranteed risk (LTT)", loc="left")
    b.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.3, columnspacing=0.6, fontsize=7, borderaxespad=0.1)
    b.set_ylim(0, 1.1)
    # (c) per-split realised risk at alpha = 0.10
    c = ax[2]
    k = m_hat.argmax(1)
    s, loss = m_hat[ar, k], 1 - p[ar, k]
    res = {"LTT": [], "CRC": [], "selection-\nunaware": [], "majority\nvote": []}
    for cal, te, kr in splits(100):
        for mode, key in (("ltt", "LTT"), ("crc", "CRC")):
            t = thresholds(s, loss, cal, te, 0.10, mode=mode)
            res[key].append(np.mean((s[te] > t) * loss[te]))
        sr, lr = m_hat[ar, kr], 1 - p[ar, kr]
        t = thresholds(sr, lr, cal, te, 0.10, mode="crc")
        res["selection-\nunaware"].append(np.mean((s[te] > t) * loss[te]))
        lm = 1 - maj[ar, k]
        t = thresholds(s, lm, cal, te, 0.10, mode="crc")
        res["majority\nvote"].append(np.mean((s[te] > t) * loss[te]))
    rng = np.random.default_rng(1)
    for i, (key, v) in enumerate(res.items()):
        v = np.array(v)
        col = COL["hcbon"] if i < 2 else MUTED
        c.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=5, color=col, alpha=0.55, lw=0)
        c.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=1.1)
        c.text(i, max(v.max(), 0.1) + 0.006, f"{np.mean(v > 0.1):.0%}", ha="center", fontsize=6.5, color=INK)
    c.axhline(0.10, color=INK, lw=0.8, ls="--")
    c.text(3.45, 0.101, "$\\alpha$", fontsize=7, va="bottom")
    c.set_xticks(range(4))
    c.set_xticklabels(list(res), fontsize=6.6)
    c.set_ylabel("realised dissatisfaction (test half)")
    c.set_ylim(0.06, 0.168)
    c.grid(axis="y", color=GRID, lw=0.6)
    c.set_title("(c) Realised risk over 100 splits", loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_bon_diffiqa")
    return {k_: [float(np.mean(v)), float(np.mean(np.array(v) > 0.1))] for k_, v in res.items()}


# ------------------------------------------------------------------ fig_majority
def fig_majority():
    from scipy.stats import binom
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    pp = np.linspace(0, 1, 401)
    for r, col, ls in ((3, COL["hcbon"], "-"), (5, COL["liqe"], "--"), (9, COL["musiq"], "-.")):
        phi = binom.sf(r // 2, r, pp)                                     # P(majority of r says better)
        ax.plot(pp, phi - pp, color=col, lw=1.4, ls=ls, label=f"panel $r={r}$")
    ax.axhline(0, color=MUTED, lw=0.7)
    pk = (3 + np.sqrt(3)) / 6
    ax.annotate("$r=3$: max 0.096 at $p\\approx0.79$", (pk, pk * (1 - pk) * (2 * pk - 1)), (0.55, -0.19),
                fontsize=6.3, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.fill_between(pp, -0.26, 0.26, where=pp > 0.5, color=GRID, alpha=0.45, lw=0)
    ax.text(0.985, 0.245, "typically deployed\n($p>1/2$)", ha="right", va="top", fontsize=6.3, color=MUTED)
    ax.set_xlabel("population preference $p$")
    ax.set_ylabel("$\\varphi_r(p)-p$")
    ax.set_ylim(-0.26, 0.26)
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, "fig_majority")


# ------------------------------------------------------------------ fig_sriqa
def fig_sriqa():
    g = json.load(open(DATA + "/goodhart_sriqa.json"))
    tr = json.load(open(W + "sriqa_transfer_v3.json"))["settings"]
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [1, 1.35]})
    a = ax[0]
    Ns = sorted(int(n) for n in g["by_N"])
    for q in ["musiq", "liqe", "topiq_nr", "clipiqa", "maniqa", "dists"]:
        y = [g["by_N"][str(n)]["efficiency"][q] for n in Ns]
        a.plot(Ns, y, color=COL[q], lw=1.4, marker=MK[q], ms=3, mec=SURF, mew=0.5, label=NAME[q])
    a.set_xlabel("number of SR methods $N$")
    a.set_ylabel("selection efficiency")
    a.set_xticks(Ns)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_ylim(0.2, 0.86)
    a.legend(frameon=False, ncol=3, loc="lower left", handlelength=1.4, columnspacing=0.8)
    a.set_title("(a) SRIQA-Bench: proxy selection vs $N$", loc="left")
    b = ax[1]
    qs = ["hcbon", "afine", "liqe", "musiq", "topiq_nr", "clipiqa", "maniqa"]
    for j, (st, off, alpha_) in enumerate((("A_ref_RealESRNet", -0.19, 0.5), ("B_ref_GT", 0.19, 1.0))):
        v = [tr[st]["select"][q]["efficiency"] for q in qs]
        b.bar(np.arange(len(qs)) + off, v, width=0.36, color=[COL[q] for q in qs], alpha=alpha_, edgecolor=SURF, lw=1)
        for i in (0,):
            b.text(i + off, v[i] + 0.015, f"{v[i]:.2f}", ha="center", fontsize=6.3)
    b.set_xticks(range(len(qs)))
    b.set_xticklabels([NAME[q] for q in qs], fontsize=6.8)
    b.set_ylabel("best-of-$K$ efficiency")
    b.set_ylim(0, 1.1)
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.text(0.99, 0.98, "light: reference = RealESRNet ($K$=9)\ndark: reference = GT ($K$=10)", transform=b.transAxes,
           ha="right", va="top", fontsize=6.3, color=MUTED)
    b.set_title("(b) Transfer without retraining", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_sriqa")


# ------------------------------------------------------------------ fig_ablation
def fig_ablation():
    tj = json.load(open(W + "pred3_test_train.json"))
    fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.1), gridspec_kw={"hspace": 0.75})
    a = ax[0]
    best = {}
    for g_ in tj["grid"]:
        best[g_["fset"]] = max(best.get(g_["fset"], 0), g_["cv_bo6_p"])
    lab = {"dino": "DINOv2", "proxy": "proxies", "dino+proxy": "DINOv2+proxies", "dino+proxy+afine": "DINOv2+prox.+A-FINE",
           "proxy+afine": "proxies+A-FINE"}
    keys = ["dino", "proxy", "dino+proxy", "dino+proxy+afine", "proxy+afine"]
    vals = [best[k] for k in keys] + [tj["train_proxy_selectors"]["afine"]["bo6_p"]]
    names = [lab[k] for k in keys] + ["A-FINE alone"]
    cols = [COL["hcbon"] if k == "proxy+afine" else "#9fc1ea" for k in keys] + [COL["afine"]]
    a.barh(range(len(vals)), vals, color=cols, height=0.62, edgecolor=SURF)
    for i, v in enumerate(vals):
        a.text(v + 0.0007, i, f"{v:.3f}", va="center", fontsize=6.3)
    a.set_yticks(range(len(vals)))
    a.set_yticklabels(names, fontsize=6.6)
    a.invert_yaxis()
    a.set_xlim(0.565, 0.61)
    a.set_xlabel("CV best-of-6 preference (training split)")
    a.grid(axis="x", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title("(a) Predictor inputs", loc="left")
    b = ax[1]
    diffs = {}
    for q in ("hcbon", "afine"):
        k = S[q].argmax(1)
        s, loss = S[q][ar, k], 1 - p[ar, k]
        diffs[q] = np.array([np.mean(s[te] > thresholds(s, loss, cal, te, 0.10, name=q)) for cal, te, _ in splits(100)])
    d = 100 * (diffs["hcbon"] - diffs["afine"])
    b.hist(d, bins=np.arange(-1, 6.01, 0.5), color=COL["hcbon"], edgecolor=SURF, lw=1)
    b.axvline(0, color=INK, lw=0.8)
    b.set_xlabel("deployment gain over A-FINE at $\\alpha$=0.10 (points)")
    b.set_ylabel("splits")
    b.text(0.98, 0.95, f"mean {d.mean():+.1f}; >0 in {np.mean(d > 0):.0%}", transform=b.transAxes, ha="right",
           va="top", fontsize=6.5)
    b.set_title("(b) Paired comparison over 100 splits", loc="left")
    save(fig, "fig_ablation")
    return float(d.mean()), float(np.mean(d > 0))


if __name__ == "__main__":
    R1 = json.load(open(W + "results_v3_a0.10.json"))
    info = {"examples": fig_examples()}
    fig_pipeline()
    info["risk_c"] = fig_bon(R1)
    fig_majority()
    fig_sriqa()
    info["ablation_gain"] = fig_ablation()
    json.dump(info, open(W + "figs_v3_info.json", "w"), indent=1)
    print(info)
