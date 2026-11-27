# Data

## Source

**Car License Plate Detection** (license plates on cars, 433 photos).

- Kaggle: <https://www.kaggle.com/datasets/andrewmvd/car-plate-detection>
- DatasetNinja conversion: <https://datasetninja.com/car-license-plate-detection>
- Licence: **CC0 1.0** (public domain dedication). See `raw/LICENSE.md`
  after extraction.

The archive is not committed. Download it and run:

```bash
python data/get_dataset.py --archive /path/to/car-license-plate-DatasetNinja.tar
```

Expected layout:

```
data/raw/
  LICENSE.md
  README.md
  meta.json
  ds/
    img/Cars0.png, Cars1.png, ...
    ann/Cars0.png.json, Cars1.png.json, ...
```

## Verified content (27 Nov 2026)

| Item | Value |
|---|---|
| Image files | 433 PNG |
| Annotation files | 433 JSON (Supervisely rectangles, class `license plate`) |
| Reference boxes | 471 |
| Images with 2 plates | 18 |
| Images with 4 plates | 5 |
| Images with 6 plates | 1 |
| Image sizes | 128 distinct sizes, 225-600 px wide, 141-531 px high |
| Pixel formats | 110 x RGB uint8, 8 x RGBA uint8, 315 x RGBA uint16 |

## Duplicate scenes and label jitter

A pixel hash of the normalised grayscale image shows **299 unique scenes**:
85 groups contain duplicate files (2-5 copies, 219 files in total). The copies
carry slightly different reference boxes; matched-box IoU between copies has
median 0.825, 10th percentile 0.671, and 3.4 % of pairs fall below 0.5.

Consequences, implemented in `src/data.py`:

1. The development/test split operates on duplicate groups, so one scene never
   appears in both sets.
2. One canonical copy (lexicographically first name) represents each group in
   evaluation; 134 duplicate copies are excluded to avoid double counting.
3. Label noise of roughly 0.17 IoU is expected even for a perfect detector and
   is discussed in the report.

The generated split is `data/splits/dev_test.json` (209 dev scenes, 90 test
scenes, seed 42). Delete it to regenerate with the same settings.
