# Week 10 findings and design decisions

## Mixed PNG encodings

**Evidence:** 315 RGBA uint16, 110 RGB uint8, 8 RGBA uint8

**Decision:** Single loader: 16->8 bit, drop alpha, RGB or grayscale uint8 0-255

## Duplicate scenes with label jitter

**Evidence:** 433 files -> 299 scenes; matched-box median IoU 0.825, 3.4% below 0.5

**Decision:** Split duplicate groups, keep one canonical copy, report the label-noise ceiling

## Plates are small and variable

**Evidence:** median 69 x 25 px; width p25/p75 38/114 px

**Decision:** Multi-scale candidates, area and aspect filtering, no fixed pixel thresholds

## Low contrast in a tail of images

**Evidence:** gray std p05 = 47.4

**Decision:** Test gamma and CLAHE enhancement before edge detection

## Uneven illumination

**Evidence:** 16x16 smoothed block range p95 = 244 gray levels

**Decision:** Background correction or adaptive thresholding; do not rely on one global threshold

## Visible noise

**Evidence:** 5x5 median residual sigma p95 = 21.4

**Decision:** Compare mean, Gaussian and median smoothing under controlled Gaussian and salt-and-pepper noise

## Strong high-frequency background structure

**Evidence:** spectra show energy on both axes from road, cars and vegetation

**Decision:** Use direction-specific Sobel/LoG edges and wide horizontal closing to isolate plate character rows

## Resolution and gray-level limits

**Evidence:** lowest passing scale 0.75; lowest passing gray levels 4

**Decision:** Process at native resolution, never below 0.75 scale; keep at least 4 gray levels

## Raw-intensity baselines fail

**Evidence:** dev mean IoU 0.058 (fixed threshold) and 0.052 (Otsu + opening)

**Decision:** Pipeline A starts from gradient magnitude and morphology, not global intensity

## Multiple plates per scene

**Evidence:** 23 unique scenes contain 2-6 plates

**Decision:** Pipeline must return several candidates; use greedy matching and count error as secondary metric
