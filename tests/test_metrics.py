"""Hand-worked checks for the shared metrics.

Run from the repository root with::

    python -m unittest discover -s tests -v

Hand calculations used below:

* MSE: A = [[0, 2], [4, 6]], B = [[1, 2], [3, 6]].  Squared differences are
  1, 0, 1, 0, so MSE = 2 / 4 = 0.5.
* PSNR: 10 log10(255^2 / 0.5) = 10 log10(130050) = 51.1280 dB.
* Mask IoU: 3 x 3 masks with 3 foreground pixels each and 2 shared pixels;
  intersection 2, union 4, IoU = 0.5.
* Box IoU: (0, 0, 10, 10) and (5, 0, 15, 10) overlap 5 x 10 = 50 px;
  union = 100 + 100 - 50 = 150 px; IoU = 1/3.
"""

import math
import unittest

import numpy as np

from src import metrics


class MetricsTest(unittest.TestCase):
    def test_mse_hand_example(self):
        reference = np.array([[0, 2], [4, 6]], dtype=np.uint8)
        test = np.array([[1, 2], [3, 6]], dtype=np.uint8)
        self.assertAlmostEqual(metrics.mse(reference, test), 0.5)

    def test_psnr_hand_example(self):
        reference = np.array([[0, 2], [4, 6]], dtype=np.uint8)
        test = np.array([[1, 2], [3, 6]], dtype=np.uint8)
        expected = 10.0 * math.log10(255.0**2 / 0.5)
        self.assertAlmostEqual(metrics.psnr(reference, test), expected, places=6)
        self.assertEqual(metrics.psnr(reference, reference), float("inf"))

    def test_mask_iou_hand_example(self):
        prediction = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        truth = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        self.assertAlmostEqual(metrics.iou_mask(prediction, truth), 0.5)

    def test_mask_iou_edge_cases(self):
        empty = np.zeros((3, 3), dtype=np.uint8)
        full = np.ones((3, 3), dtype=np.uint8)
        self.assertEqual(metrics.iou_mask(empty, empty), 1.0)
        self.assertEqual(metrics.iou_mask(empty, full), 0.0)

    def test_box_iou_hand_example(self):
        prediction = (0, 0, 10, 10)
        truth = (5, 0, 15, 10)
        self.assertAlmostEqual(metrics.iou_box(prediction, truth), 1.0 / 3.0)
        self.assertEqual(metrics.iou_box(None, truth), 0.0)

    def test_best_box_iou_selects_matching_box(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 70, 70]], dtype=np.float32)
        value, index = metrics.best_box_iou((52, 52, 68, 68), boxes)
        self.assertEqual(index, 1)
        # 16 x 16 prediction inside a 20 x 20 box: 256 / 400 = 0.64.
        self.assertAlmostEqual(value, 0.64)

    def test_count_error(self):
        self.assertEqual(metrics.count_error(3, 2), 1)
        self.assertEqual(metrics.count_error(0, 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
