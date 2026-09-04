import os
import unittest
from pathlib import Path

import torch

from foresight.checkpoint import load_checkpoint


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_archived_checkpoint_loads_strictly(self):
        value = os.environ.get("FORESIGHT_CHECKPOINT")
        if not value:
            self.skipTest("FORESIGHT_CHECKPOINT is not set")
        model, config, metadata = load_checkpoint(Path(value), torch.device("cpu"))
        self.assertEqual(config["hidden_dim"], 128)
        self.assertEqual(config["num_blocks"], 4)
        self.assertEqual(config["modes"], 8)
        self.assertIn("epoch", metadata)
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 3, 256, 256),
                torch.ones(1, 1, 256, 256),
            )
        self.assertEqual(tuple(output.shape), (1, 3, 256, 256))


if __name__ == "__main__":
    unittest.main()
