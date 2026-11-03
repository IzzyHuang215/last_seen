# benchmark/

Detector comparison for LastSeen. Write-up: [COMPARISON.md](COMPARISON.md). Colab runner: `../LastSeen_Model_Comparison.ipynb`.

| File | What it is |
|---|---|
| `compare_models.py` | Runs each detector on `images/`, caches raw detections, scores them, writes `results/*.csv` |
| `make_figures.py` | Per-item AP heatmap + side-by-side example figure |
| `ground_truth.json` | The 16 query items (COCO mapping + text prompt) and per-image presence labels (1 / 0 / -1 = ambiguous) |
| `verified_boxes.json` | Hand-checked correct / wrong boxes per image and item, used for the IoU ≥ 0.5 localization check |
| `results/summary.csv` | Main comparison table |
| `results/per_item.csv`, `results/per_item_ap.csv` | Per-item AP / recall / false positives |
| `results/raw/*.json` | Cached detections (all scores ≥ 0.01), so models can be run on different machines and merged |

```bash
python benchmark/compare_models.py                  # all models
python benchmark/compare_models.py --models owlv2   # one model
python benchmark/compare_models.py --report-only    # rebuild tables from cached results
python benchmark/make_figures.py
```

Weights are looked up in the repo root, then `benchmark/weights/`, then downloaded automatically.
