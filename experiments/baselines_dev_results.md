# Week 10 - baseline results (development split)

Generated: 2026-11-27T03:54:52+02:00  
Split seed: 42  
Development scenes: 209; test scenes: 90; derived from 299 unique scenes among 433 files (duplicate groups kept together)  
Metric: bounding-box IoU of the single predicted plate candidate against the best reference box; hit rate at IoU >= 0.5.

| Pipeline | Mean IoU | Std IoU | Median IoU | Hit@0.5 | Mean count error | ms/image |
|---|---:|---:|---:|---:|---:|---:|
| trivial | 0.0579 | 0.1606 | 0.0115 | 0.033 | 0.105 | 1.54 |
| simple | 0.0523 | 0.1470 | 0.0100 | 0.029 | 0.105 | 1.05 |

Best and worst cases:

- trivial: best Cars109.png (IoU 0.968), worst Cars107.png (IoU 0.000)
- simple: best Cars37.png (IoU 0.950), worst Cars100.png (IoU 0.000)

Both baselines threshold raw grayscale intensity, as specified for segmentation tasks, and then keep the largest connected component. This is deliberately naive for plate localization: bright road, sky or car-body regions often dominate the mask. The result motivates the gradient-based Pipeline A.

Configuration: `configs\baselines.yaml`.
