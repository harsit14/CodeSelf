from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import GRPOSmokeTrainer as ExportedGRPOTrainer  # noqa: E402
from codeself.training import PPOSmokeTrainer as ExportedPPOTrainer  # noqa: E402
from codeself.training.grpo_train import GRPOSmokeTrainer as LegacyGRPOTrainer  # noqa: E402
from codeself.training.ppo_train import PPOSmokeTrainer as LegacyPPOTrainer  # noqa: E402
from codeself.training.smoke import GRPOSmokeTrainer, PPOSmokeTrainer  # noqa: E402


class TrainingSmokeNamespaceTests(unittest.TestCase):
    def test_training_exports_point_to_smoke_trainers(self) -> None:
        self.assertIs(ExportedGRPOTrainer, GRPOSmokeTrainer)
        self.assertIs(ExportedPPOTrainer, PPOSmokeTrainer)

    def test_legacy_training_modules_remain_compatible(self) -> None:
        self.assertIs(LegacyGRPOTrainer, GRPOSmokeTrainer)
        self.assertIs(LegacyPPOTrainer, PPOSmokeTrainer)


if __name__ == "__main__":
    unittest.main()
