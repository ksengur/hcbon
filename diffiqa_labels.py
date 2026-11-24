"""DiffIQA raw per-rater labels (read directly from the released label zips) -> long CSV + panel table.

Long CSV rows: split, rater, version (01..06), image (generated file), reference, label, seconds
  label in {better, similar, worse}  (Positive / Similar / Negative for the generated image vs reference)
Panel CSV (one row per generated image): votes as score 1 / 0.5 / 0, n_raters, mean (unbiased for p),
  majority label (as used by DiffIQA / A-FINE).

  py -3 diffiqa_labels.py --zips data/DiffIQA/_dl/testlabel.zip:test \
        data/DiffIQA/_dl/validationlabel.zip:val --out data/diffiqa
"""
import argparse
import collections
import csv
import json
import os
import zipfile

MAP = {"Positive": "better", "Similar": "similar", "Negative": "worse"}
SCORE = {"better": 1.0, "similar": 0.5, "worse": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="+", required=True, help="path.zip:splitname")
    ap.add_argument("--out", required=True, help="output prefix")
    a = ap.parse_args()
    rows, bad = [], []
    for spec in a.zips:
        path, split = spec.rsplit(":", 1)
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) != 4 or parts[0] != "labels" or not n.endswith(".json"):
                continue
            try:
                d = {k.lower(): v for k, v in json.loads(z.read(n).decode("utf-8-sig")).items()}
                ai, ref = d["picture_ai"], d["picture_original"]     # key case varies across files
                lab = MAP[ai["Label"]]
            except (ValueError, KeyError, TypeError):
                bad.append(n)
                continue
            rows.append({"split": split, "rater": parts[1], "version": parts[2],
                         "image": ai["Name"], "reference": ref["Name"],
                         "label": lab, "seconds": d.get("time_cost", {}).get("Total", "")})
    with open(a.out + "_long.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["split"], r["image"], r["reference"])].append(SCORE[r["label"]])
    panel = []
    for (split, img, ref), v in sorted(by.items()):
        cnt = collections.Counter(v)
        top = cnt.most_common()
        maj = "tie" if len(top) > 1 and top[0][1] == top[1][1] else {1.0: "better", 0.5: "similar", 0.0: "worse"}[top[0][0]]
        panel.append({"split": split, "image": img, "reference": ref, "n": len(v),
                      "mean": sum(v) / len(v), "majority": maj, "votes": " ".join(f"{x:g}" for x in v)})
    with open(a.out + "_panel.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(panel[0]))
        w.writeheader()
        w.writerows(panel)
    print(f"{len(rows)} judgements, {len(panel)} generated images; skipped {len(bad)}")


if __name__ == "__main__":
    main()
