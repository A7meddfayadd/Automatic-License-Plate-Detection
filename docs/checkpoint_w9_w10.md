# Week 9-10 checkpoint evidence

Course: CSE281 Digital Image Processing. Project: Car License Plate
Localization. Team size: 5-6 (fill names in the README roles table).

## Week 9 - topic settled

| Requirement | Evidence | Status |
|---|---|---|
| Problem and dataset chosen | Car License Plate Detection, 433 photos, 471 boxes, CC0 1.0; `README.md`, `data/README.md` | Done |
| Dataset checked against criteria | 299 unique scenes, full ground truth, four measured image defects, Colab-feasible; README criteria table and notebook section 2 | Done |
| Primary metric fixed | per-image box IoU, mean +/- std and hit@0.5; count error secondary; justified in `README.md` | Done |
| Roles assigned | six roles in `README.md` (names still `TBD`) | Done |
| Repository created | modular layout, `.gitignore`, `requirements.txt`, `configs/baselines.yaml` | Done |

## Week 10 - baselines ready

| Requirement | Evidence | Status |
|---|---|---|
| Image analysis finished | `notebooks/01_image_analysis.ipynb` (executed, outputs included); seven figures in `experiments/figures/`; findings in `experiments/week10_findings.md` | Done |
| Development/test split with ground truth | `data/splits/dev_test.json`: 209 dev / 90 test unique scenes, seed 42, duplicate groups kept together | Done |
| Metric functions tested | `tests/test_metrics.py`: 7 hand-worked cases pass (`python -m unittest discover -s tests -v`) | Done |
| Both baselines measured | `python -m src.evaluate`; results in `experiments/baselines_dev_results.md`, per-image rows in `baselines_dev.csv`, log in `experiment_log.csv` | Done |

## Baseline numbers (development set, 209 unique scenes)

| Pipeline | Mean IoU | Std | Median | Hit@0.5 | Count error | ms/image |
|---|---:|---:|---:|---:|---:|---:|
| Fixed threshold 127 | 0.0579 | 0.1606 | 0.0115 | 0.033 | 0.105 | 1.54 |
| Otsu + 3x3 opening | 0.0523 | 0.1470 | 0.0100 | 0.029 | 0.105 | 1.05 |

Best: `Cars109.png` 0.968 (trivial), `Cars37.png` 0.950 (simple).
Worst: `Cars107.png`, `Cars100.png` (0.000).
Interpretation: raw intensity selects bright backgrounds, exactly why the
natural pipeline for this dataset starts from vertical plate edges.

## Image-analysis outcomes used as design constraints

- Loader normalises 16-bit RGBA, 8-bit RGBA and 8-bit RGB to uint8.
- Median plate is 69 x 25 px (width p25/p75 38/114), so pixel thresholds must
  scale with the image and candidates are filtered by area/aspect later.
- Contrast p05 = 47.4, illumination block range p95 = 244, noise sigma p95 =
  21.4, blur score p05 = 341 -> enhancement, adaptive threshold and median
  filtering all need evidence-based comparison.
- Lowest passing scale 0.75; lowest passing gray-level count 4. Native
  resolution is kept and no downscale below 0.75 is allowed.
- Duplicate audit: 433 files -> 299 scenes; label jitter median 0.825, 3.4 %
  below IoU 0.5. Split is group-aware; evaluation uses one canonical copy.
- Multi-plate scenes: 23 unique scenes (up to 6 plates), so the pipeline must
  emit multiple candidates and report count error.

## Reproduce

```bash
pip install -r requirements.txt
python data/get_dataset.py --archive /path/to/car-license-plate-DatasetNinja.tar
python -m unittest discover -s tests -v
python -m src.evaluate
python -m nbconvert --to notebook --execute --inplace notebooks/01_image_analysis.ipynb
```

## Open items before Week 14

- Replace `TBD` team names in `README.md`; update the `author` field in
  `configs/baselines.yaml` and the two rows already written to
  `experiments/experiment_log.csv`.
- Two duplicate groups disagree on the plate count; the canonical copy is used
  and the disagreement is reported as label noise.
- The representative "noisy" scene is a screenshot of an image-editing
  application; it is retained as a genuine dataset difficulty, not a synthetic
  sample.
- The test split has not been evaluated by any command; keep it that way until
  Week 13.
- Weeks 11-12 must add Pipelines A/B/C, the remaining modules, NumPy
  implementations with hand-worked 3x3/5x5 checks, tuning and four ablations.
