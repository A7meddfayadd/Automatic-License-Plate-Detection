# PlateFinder - Car License Plate Localization (CSE281)

Classical image-processing pipeline that locates the license plate in a photo
of a car. Course project for **CSE281 Digital Image Processing**, Weeks 9-14.

**Status:** Week 9 and Week 10 checkpoints complete. Pipeline A/B/C,
experiments, test-set evaluation and demo follow in Weeks 11-14.

## Problem statement

A traffic-monitoring technician receives photos from fixed or handheld cameras
and needs the license-plate region isolated before a human reads the plate.
The pipeline takes one raw photo and returns the plate bounding box (or
several boxes when a scene contains more than one car) together with the plate
count. A missed plate forces a reviewer to inspect the whole image; a false box
wastes review time and may crop a sensitive region. The **primary metric is
per-image bounding-box IoU** against the best reference box, summarized as
mean +/- standard deviation and hit rate at IoU >= 0.5. IoU grades both
position and size errors continuously, matches the cost of cropping the wrong
region, and is the standard localization measure. **Absolute count error** is
the secondary metric because 23 unique scenes contain 2-6 plates.

- **Input:** RGB `uint8` 0-255 at native resolution (225-600 x 141-531 px);
  grayscale `uint8` for the Week-10 baselines.
- **Output:** `x1, y1, x2, y2` plate box in native pixels, plus plate count.
- **Scope:** classical processing only (no learned models), NumPy + OpenCV +
  scikit-image + SciPy + Matplotlib, runnable on free Colab CPU.

## Dataset

**Car License Plate Detection** - 433 real street/car photos with 471
reference boxes, **CC0 1.0**.

- Kaggle: <https://www.kaggle.com/datasets/andrewmvd/car-plate-detection>
- DatasetNinja conversion: <https://datasetninja.com/car-license-plate-detection>
- Local preparation: `python data/get_dataset.py --archive <archive>`

Verified content and the duplicate/label-quality analysis are in
[data/README.md](data/README.md). The archive is not committed.

### Dataset criteria check

| Criterion | Requirement | Evidence | Status |
|---|---|---|---|
| Size | >= 100 real images | 299 unique scenes (433 files) | Met |
| Ground truth | reference answer for every image | 471 boxes, every image annotated | Met |
| Task difficulty | needs object/background separation | box localization, 23 multi-plate scenes | Met |
| Real-world mess | >= 2 of low contrast / uneven light / noise / blur | contrast p05 47.4; noise sigma p95 21.4; illumination range p95 244; blur p05 341 | Met |
| Licence | academic use | CC0 1.0 | Met |
| Feasibility | < 10 min on free Colab CPU | baselines: 209 scenes in a few seconds | Met |
| Not a toy | no sample/synthetic data, not one filter | 299 real photos; both baselines fail (IoU about 0.05) | Met |

### Data-quality finding

The 433 files collapse to **299 unique scenes** (85 duplicate groups, 219
files). Copies carry slightly different boxes: matched-box IoU median 0.825,
10th percentile 0.671, 3.4 % below 0.5. The split therefore operates on
duplicate groups, one canonical copy per group is evaluated, and 134 redundant
files are excluded. This imposes a label-noise ceiling of about 0.17 IoU on any
detector and is reported with the results.

## Week 9-10 results

### Split

| Set | Unique scenes | Purpose |
|---|---:|---|
| Development | 209 | all tuning and baseline measurement |
| Test | 90 | untouched until Week 13 |

Frozen in `data/splits/dev_test.json`, seed 42. No duplicate group straddles
the two sets.

### Baselines on the development set

Both baselines follow the specification: threshold raw grayscale intensity,
keep the largest connected component, and score its box. Full per-image rows
are in `experiments/baselines_dev.csv`.

| Pipeline | Mean IoU | Std IoU | Median IoU | Hit@0.5 | Mean count error | ms/image |
|---|---:|---:|---:|---:|---:|---:|
| Trivial: fixed threshold 127 | 0.0579 | 0.1606 | 0.0115 | 0.033 | 0.105 | 1.54 |
| Simple: Otsu + 3x3 opening | 0.0523 | 0.1470 | 0.0100 | 0.029 | 0.105 | 1.05 |

Best cases: `Cars109.png` (0.968, trivial), `Cars37.png` (0.950, simple).
Worst cases: `Cars107.png` and `Cars100.png` (0.000) where the largest bright
component is the sky, road or car body. This is an honest baseline failure and
motivates the gradient-based Pipeline A.

Example outputs: `experiments/figures/week10_baselines.png`.
Summary: `experiments/baselines_dev_results.md`.
Experiment log: `experiments/experiment_log.csv`.

### Image analysis (Week 10)

Executed notebook: `notebooks/01_image_analysis.ipynb` (outputs included).

| Question | Finding | Design decision |
|---|---|---|
| Image formats | 315 RGBA uint16, 110 RGB uint8, 8 RGBA uint8 | one loader: 16->8 bit, drop alpha, RGB/grayscale |
| Plate scale | median 69 x 25 px, width p25/p75 38/114 px | multi-scale candidates, area/aspect filtering |
| Contrast tail | gray std p05 = 47.4 | test gamma and CLAHE before edges |
| Illumination | smoothed block range p95 = 244 | background correction / adaptive threshold |
| Noise | residual sigma p95 = 21.4 | compare mean, Gaussian, median under controlled noise |
| Spectra | strong horizontal/vertical background energy | direction-specific Sobel/LoG, wide horizontal closing |
| Resolution | lowest passing scale 0.75 | never downscale below 0.75 |
| Gray levels | lowest passing count 4 | keep >= 4 levels |
| Baselines | mean IoU about 0.05 | start Pipeline A from gradients + morphology |
| Duplicates | 299 scenes, label jitter median 0.825 | group split, canonical copy, report label noise |

Figures are in `experiments/figures/`; conclusions and decisions are also
saved in `experiments/week10_findings.md`.

## Repository layout

```
configs/baselines.yaml          Week-10 baseline configuration
data/get_dataset.py             extract and validate the dataset
data/README.md                  source, licence, duplicate audit
data/splits/dev_test.json       frozen duplicate-aware split
notebooks/01_image_analysis.ipynb   executed Week-10 analysis
src/data.py                     consistent loader, boxes, split
src/metrics.py                  MSE, PSNR, mask/box IoU, count error
src/pipeline.py                 the two baselines
src/evaluate.py                 dev evaluation, log, figure
tests/test_metrics.py           hand-worked metric checks
experiments/                    log, per-image CSV, summary, findings, figures
docs/checkpoint_w9_w10.md       evidence for both checkpoints
app/                            demo app (Weeks 13-14)
report/                         report PDF and slides (Weeks 13-14)
```

Weeks 11-12 add `enhancement.py`, `restoration.py`, `frequency.py`,
`segmentation.py`, `morphology.py`, `run_pipeline.py` and Pipelines A/B/C.

## Setup and commands

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/get_dataset.py --archive /path/to/car-license-plate-DatasetNinja.tar
python -m unittest discover -s tests -v
python -m src.evaluate
python -m nbconvert --to notebook --execute --inplace notebooks/01_image_analysis.ipynb
```

On Colab, clone/upload the repository and dataset, run the same commands, and
skip installing OpenCV if it is already present. Pinned versions are in
`requirements.txt`; the local verification used
`opencv-python-headless==4.10.0.84` (same `cv2` API, no GUI).

Reproducibility: one seed (42), one frozen split file, and one command per
result. `python -m src.evaluate` rewrites the baseline table, CSV, log rows and
figure deterministically. The test split is read by no Week-9/10 command.

## Team roles 

Teams of six; for teams of five, merge the Tech Lead and Data/Baseline roles.

| # | Role | Name | Work package (Weeks 9-10) |
|---|---|---|---|
| 1 ahmed | Tech Lead | TBD | problem framing, repository, config, integration, review |
| 2 jooks | Data & Baseline Engineer | TBD | loader, duplicate audit, split, metrics, both baselines |
| 3 haneen | Enhancement Engineer | TBD | histograms, contrast, resolution/gray-level study |
| 4 judy | Restoration & Frequency Engineer | TBD | noise model analysis, spectra, illumination study |
| 5 malak | Segmentation Engineer | TBD | criteria audit, baseline/threshold review |
| 6 habiba | Morphology & Evaluation Engineer | TBD | failure-case review, evaluation plan, later demo |

Every member still owes at least two NumPy implementations and one experiment,
as required by the specification. Replace `TBD` before the Week-14 submission.

### Contribution table template (report)

| Member | Work package | Files | Experiments run |
|---|---|---|---|
| TBD | Tech Lead | README, configs, review | split/baseline reproduction |
| TBD | Data & Baseline | data.py, metrics.py, evaluate.py, tests | baseline dev run |
| TBD | Enhancement | notebook sections 3, 6, 7 | resolution/gray-level study |
| TBD | Restoration & Frequency | notebook sections 4-6 | noise/spectrum study |
| TBD | Segmentation | analysis review | criteria audit |
| TBD | Morphology & Evaluation | evaluation review | baseline failure review |

## Roadmap

- **Week 9 - done:** topic and dataset settled, metric fixed, roles assigned,
  repository and config created.
- **Week 10 - done:** image analysis frozen, dev/test split with ground truth,
  metric functions tested by hand, both baselines measured.
- **Weeks 11-12:** Pipelines A/B/C; NumPy implementations checked against
  OpenCV; kernel/cutoff/threshold/structuring-element tuning; four ablations.
- **Week 13:** single test-set evaluation, failure analysis of worst images,
  frozen results, report draft.
- **Week 14:** tag `v1.0`, report PDF and slides in `report/`, live demo,
  individual defense.
