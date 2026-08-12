import sys
import unittest
from unittest.mock import patch

from scripts.args import parse_train_args


class TrainSeedArgsTests(unittest.TestCase):
    def test_seed_defaults_to_42(self):
        argv = ["train.py", "--config", "config.py"]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertEqual(args.seed, 42)

    def test_seed_accepts_explicit_value(self):
        argv = ["train.py", "--config", "config.py", "--seed", "3407"]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertEqual(args.seed, 3407)


if __name__ == "__main__":
    unittest.main()
