# Selection-Aware Calibration for Best-of-N Generative Restoration

Code for the paper *Selection-Aware Calibration for Best-of-N Generative Restoration with Population-Level
Human-Preference Guarantees* by Abdulkadir Sengur, Massimo Salvi, Yan Tian and Rajendra Acharya
(submitted to IEEE Transactions on Image Processing).

HC-BoN chooses among N generated candidates or keeps the reference, and calibrates the whole select-then-deploy
decision with Learn-then-Test (LTT) or Conformal Risk Control (CRC) on a vote-level loss. The result is a
finite-sample bound on the expected fraction of viewers who would have preferred the reference, valid for any panel
size and any selector.

**No images, human labels, vote tables or proxy scores are included.** DiffIQA, SRIQA-Bench and FGRestore have to be
obtained from their authors; SRIQA-Bench's licence forbids redistribution. `results/` holds the JSON results behind
every number in the paper.

## Contents

| File | What it does |
|---|---|
| `bon_diffiqa.py` | Main pipeline on DiffIQA: `table`, `features2`, `proxies`, `train`, `evaluate` (calibration, baselines, best-of-N curves) |
| `diffiqa_labels.py`, `unzip_split.py` | Read the DiffIQA label archives; extract split zip archives |
| `afine_scores.py` | A-FINE scores (overall, fidelity, naturalness) via IQA-PyTorch |
| `cv_c.py`, `paired_vs_afine.py`, `ablation_similar.py`, `panel_size.py`, `check_grid.py` | Selection constant, paired comparison with A-FINE, "similar" ablation, panel-size sensitivity, grid check |
| `review_analyses.py`, `timing.py`, `verify_calibration.py` | Controlled comparison and bootstrap intervals, run time, simulation check of the calibration code |
| `sriqa_labels.py`, `sriqa_proxies.py`, `goodhart_sriqa.py`, `sriqa_transfer.py`, `sriqa_calsize.py` | SRIQA-Bench: labels, proxies, selection efficiency, transfer without retraining, calibration-size study |
| `fgrestore_score.py`, `fgrestore_eval.py` | External selection-only validation on FGRestore |
| `make_figs_v3.py` | All figures of the paper |
| `paths.py` | Data / work / figure locations (environment variables `HCBON_DATA`, `HCBON_WORK`, `HCBON_FIGS`) |
| `results/` | JSON results used in the paper |

## Setup

Python 3.12, PyTorch 2.6 (CUDA 12.4), timm 1.0, pyiqa 0.1.16 (IQA-PyTorch).

```bash
pip install -r requirements.txt
```

Data, under `data/` by default:

- DiffIQA (Chen *et al.*, CVPR 2025, A-FINE): validation and test images and labels.
- SRIQA-Bench (same authors): images and pairwise comparisons.
- FGRestore (Sheng *et al.*, AAAI 2026): only for the external validation.

## Reproducing the paper

```bash
W=work
python diffiqa_labels.py --zips data/DiffIQA/_dl/testlabel.zip:test data/DiffIQA/_dl/validationlabel.zip:val --out data/diffiqa
python bon_diffiqa.py table --root data/DiffIQA --panel data/diffiqa_panel.csv --out $W
for s in val test; do
  python bon_diffiqa.py proxies --table $W/table_$s.npz --out $W/proxies_$s.npz
  python bon_diffiqa.py proxies --ref_only --table $W/table_$s.npz --out $W/proxies_ref_$s.npz
  python afine_scores.py diffiqa --table $W/table_$s.npz --out $W/afine_$s.npz
  python bon_diffiqa.py features2 --table $W/table_$s.npz --out $W/feat2_$s.npz
done
# predictor: trained and selected on the DiffIQA validation split only (content-grouped 5-fold CV)
python bon_diffiqa.py train --feat_tr $W/feat2_val.npz --table_tr $W/table_val.npz --feat_te $W/feat2_test.npz \
  --out $W/pred3_test.npz --k 256 512 --hidden 128 --wd 1e-2 --epochs 30 \
  --fsets dino dino+proxy dino+proxy+afine proxy proxy+afine \
  --prox_tr $W/proxies_val.npz --ref_tr $W/proxies_ref_val.npz --afine_tr $W/afine_val.npz \
  --prox_te $W/proxies_test.npz --ref_te $W/proxies_ref_test.npz --afine_te $W/afine_test.npz
# calibration and evaluation on 100 content-disjoint halves of the DiffIQA test split (Table I, Fig. 3)
for a in 0.10 0.20; do
  python bon_diffiqa.py evaluate --table $W/table_test.npz --pred $W/pred3_test.npz \
    --proxies $W/proxies_test.npz --afine $W/afine_test.npz \
    --proxies_val $W/proxies_val.npz --afine_val $W/afine_val.npz \
    --c 0 --alpha $a --reps 100 --out $W/results_v3_a$a.json
done
# SRIQA-Bench: labels, proxy and A-FINE scores, then transfer without retraining (Fig. 5)
python sriqa_labels.py --root data/SRIQA-Bench --out data/sriqa_pairs.csv
python sriqa_proxies.py --root data/SRIQA-Bench --out data/sriqa_proxies.csv
python afine_scores.py sriqa --out data/sriqa_afine.csv
python goodhart_sriqa.py --pairs data/sriqa_pairs.csv --proxies data/sriqa_proxies.csv
python sriqa_transfer.py --train_json $W/pred3_test_train.json --c 0 --out $W/sriqa_transfer_v3.json
python make_figs_v3.py
```

The remaining analysis scripts (`review_analyses.py`, `paired_vs_afine.py`, `panel_size.py`, `fgrestore_eval.py`, …)
read the files in `work/` and write the corresponding JSON in `results/`.

Rules that keep the guarantee valid, and that the code follows:

1. The predictor and the selection constant are fitted on the DiffIQA validation split only.
2. Calibration and test are content-disjoint random halves of the DiffIQA test split.
3. The threshold grid is fixed before calibration (401 equally spaced values in [0, 1] for predicted preferences,
   401 quantiles of the validation-split scores for other selectors).

## Licence

Code: MIT (see `LICENSE`). The datasets are subject to their own licences and are not included.

## Citation

```bibtex
@article{hcbon2026,
  title   = {Selection-Aware Calibration for Best-of-N Generative Restoration with Population-Level Human-Preference Guarantees},
  author  = {Sengur, Abdulkadir and Salvi, Massimo and Tian, Yan and Acharya, Rajendra},
  journal = {IEEE Transactions on Image Processing},
  note    = {submitted},
  year    = {2026}
}
```
