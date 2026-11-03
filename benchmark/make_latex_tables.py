"""Write report/tables.tex (main results, speed, per-item AP) from benchmark/results/*.csv.

    python benchmark/make_latex_tables.py
    python benchmark/make_latex_tables.py --provisional "OWLv2 B/16" "YOLO-World v2-S"   # mark rows with a dagger
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parent
RES = BENCH / "results"
OUT = BENCH.parent / "report" / "tables.tex"
GT = json.load(open(BENCH / "ground_truth.json"))

ORDER = ["YOLO11s (baseline)", "Faster R-CNN R50-FPN v2", "RT-DETR-L", "YOLO-World v2-S",
         "YOLOE-11S", "YOLOE-11S @1280px", "YOLOE-11L", "OWLv2 B/16"]
SHORT = {"YOLO11s (baseline)": "YOLO11s (baseline)", "Faster R-CNN R50-FPN v2": "Faster R-CNN",
         "RT-DETR-L": "RT-DETR-L", "YOLO-World v2-S": "YOLO-World-S", "YOLOE-11S": "YOLOE-11S",
         "YOLOE-11S @1280px": "YOLOE-11S @1280", "YOLOE-11L": "YOLOE-11L", "OWLv2 B/16": "OWLv2"}
HEAD = {"YOLO11s (baseline)": "YOLO11s", "Faster R-CNN R50-FPN v2": "\\makecell{Faster\\\\R-CNN}",
        "RT-DETR-L": "\\makecell{RT-\\\\DETR}", "YOLO-World v2-S": "\\makecell{YOLO-\\\\World}",
        "YOLOE-11S": "\\makecell{YOLOE\\\\-S}", "YOLOE-11S @1280px": "\\makecell{YOLOE-S\\\\@1280}",
        "YOLOE-11L": "\\makecell{YOLOE\\\\-L}", "OWLv2 B/16": "OWLv2"}


def bold_best(values, fmt="{:.2f}", higher=True):
    best = max(values) if higher else min(values)
    return [("\\textbf{" + fmt.format(v) + "}") if v == best and v != 0 else fmt.format(v) for v in values]


def main(provisional):
    s = pd.read_csv(RES / "summary.csv").set_index("Model")
    models = [m for m in ORDER if m in s.index]
    s = s.loc[models]
    dag = lambda m: SHORT[m] + ("$^\\dagger$" if m in provisional else "")
    out = []

    # --- main results -------------------------------------------------------
    cols = ["mAP (all 16)", "mAP COCO (8)", "mAP non-COCO (8)", "Recall COCO", "Recall non-COCO",
            "Precision", "Items found (/16)", "CPU ms/img"]
    cells = {c: bold_best(list(s[c]), "{:d}" if c in ("Items found (/16)", "CPU ms/img") else "{:.2f}",
                          higher=(c != "CPU ms/img")) for c in cols}
    out += [r"\begin{table}[h]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Main results on the shared 9-image benchmark (best per column in bold).}",
            r"\label{tab:main}",
            r"\begin{tabular}{@{}l ccc cc c c r@{}}", r"\toprule",
            r" & \multicolumn{3}{c}{mAP} & \multicolumn{2}{c}{Recall} & & Items & CPU \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
            r"Model & all & COCO & non-COCO & COCO & non-COCO & Precision & found & ms/img \\", r"\midrule"]
    for i, m in enumerate(models):
        if m == "YOLO-World v2-S":
            out.append(r"\addlinespace")
        out.append(dag(m) + " & " + " & ".join(cells[c][i] for c in cols) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if provisional:
        out += [r"\\[2pt]{\footnotesize $^\dagger$ Provisional: some boxes from this model are still being "
                r"hand-verified, so its numbers may rise slightly.}"]
    out += [r"\end{table}", ""]

    # --- per-item AP --------------------------------------------------------
    ap = pd.read_csv(RES / "per_item_ap.csv", index_col=0)[models]
    pos = {it: sum(1 for lab in GT["labels"].values() if lab[it] == 1) for it in GT["items"]}
    out += [r"\begin{table}[h]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{Per-item AP. Pos.\ = number of images containing the item. Best per row in bold.}",
            r"\label{tab:peritem}",
            r"\begin{tabular}{@{}l c " + "c" * len(models) + r"@{}}", r"\toprule",
            "Item & Pos. & " + " & ".join(HEAD[m] for m in models) + r" \\", r"\midrule"]
    for grp, flag in (("Inside COCO", True), ("Outside COCO", False)):
        out.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (len(models) + 2, grp))
        for it in [i for i in GT["items"] if GT["items"][i]["coco"] == flag]:
            out.append(f"\\quad {it} & {pos[it]} & " + " & ".join(bold_best(list(ap.loc[it]))) + r" \\")
        if flag:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--provisional", nargs="*", default=[])
    main(a.parse_args().provisional)
