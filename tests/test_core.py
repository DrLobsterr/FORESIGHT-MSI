import random
import unittest

import numpy as np
import torch

from foresight.masks import exact_foreground_block_mask, legacy_v4_block_mask
from foresight.metrics import calculate_metrics
from foresight.model import FORESIGHT


class ModelTests(unittest.TestCase):
    def test_shape_and_preserved_pixels(self):
        model = FORESIGHT(hidden_dim=8, num_blocks=1, modes=2, dropout=0.0).eval()
        image = torch.randn(1, 3, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:8, 5:9] = 1
        with torch.inference_mode():
            result = model(image, mask)
        self.assertEqual(tuple(result.shape), tuple(image.shape))
        preserved = (1.0 - mask).repeat(1, 3, 1, 1).bool()
        self.assertTrue(torch.equal(result[preserved], image[preserved]))


class MaskTests(unittest.TestCase):
    def test_exact_ratio(self):
        foreground = np.zeros((32, 32), dtype=bool)
        foreground[4:28, 4:28] = True
        mask, stats = exact_foreground_block_mask(foreground, 0.4, 42)
        expected = int(round(np.count_nonzero(foreground) * 0.4))
        self.assertEqual(int(np.count_nonzero(mask.astype(bool) & foreground)), expected)
        self.assertAlmostEqual(stats["realized_foreground_ratio"], expected / 576)

    def test_legacy_is_deterministic_with_explicit_rng(self):
        foreground = np.ones((16, 16), dtype=bool)
        first, _ = legacy_v4_block_mask(foreground, 0.4, random.Random(7))
        second, _ = legacy_v4_block_mask(foreground, 0.4, random.Random(7))
        self.assertTrue(np.array_equal(first, second))


class MetricTests(unittest.TestCase):
    def test_full_and_region_scopes_differ(self):
        original = np.zeros((16, 16, 3), dtype=np.float32)
        candidate = original.copy()
        original[4:8, 4:8] = 1.0
        repair = np.zeros((16, 16), dtype=np.float32)
        repair[4:8, 4:8] = 1.0
        foreground = repair.astype(bool)
        values = calculate_metrics(original, candidate, repair, foreground)
        self.assertAlmostEqual(values["mse_full"], 0.0625)
        self.assertAlmostEqual(values["mse_repair_region"], 1.0)


if __name__ == "__main__":
    unittest.main()
