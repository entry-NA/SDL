"""Concurrency regression tests for the AEEM v2 offline pipeline."""

import threading
import time
import unittest

from aeem_v2.pipeline import ordered_staged_map


class OrderedStagedPipelineTests(unittest.TestCase):
    def test_results_preserve_input_order_when_finish_completes_out_of_order(self):
        delays = {0: 0.04, 1: 0.01, 2: 0.02, 3: 0.0}

        def finish(prepared, inference):
            time.sleep(delays[prepared])
            return inference

        outputs = list(ordered_staged_map(
            range(4),
            prepare=lambda item: item,
            infer=lambda prepared: prepared,
            finish=finish,
            finish_workers=2,
            max_pending=4,
        ))

        self.assertEqual(outputs, [0, 1, 2, 3])

    def test_finish_stage_overlaps_next_inference(self):
        first_finish_started = threading.Event()
        second_inference_started = threading.Event()

        def infer(prepared):
            if prepared == 1:
                self.assertTrue(first_finish_started.wait(timeout=1.0))
                second_inference_started.set()
            return prepared

        def finish(prepared, inference):
            if prepared == 0:
                first_finish_started.set()
                self.assertTrue(second_inference_started.wait(timeout=1.0))
            return inference

        outputs = list(ordered_staged_map(
            range(3),
            prepare=lambda item: item,
            infer=infer,
            finish=finish,
            finish_workers=2,
            max_pending=4,
        ))

        self.assertEqual(outputs, [0, 1, 2])

    def test_finish_failure_propagates_to_caller(self):
        def finish(prepared, inference):
            if prepared == 1:
                raise RuntimeError("finish failed")
            return inference

        with self.assertRaisesRegex(RuntimeError, "finish failed"):
            list(ordered_staged_map(
                range(3),
                prepare=lambda item: item,
                infer=lambda prepared: prepared,
                finish=finish,
                finish_workers=2,
                max_pending=4,
            ))

    def test_buffer_cannot_be_smaller_than_worker_count(self):
        with self.assertRaisesRegex(ValueError, "max_pending"):
            list(ordered_staged_map(
                [1],
                prepare=lambda item: item,
                infer=lambda prepared: prepared,
                finish=lambda prepared, inference: inference,
                finish_workers=2,
                max_pending=1,
            ))


if __name__ == "__main__":
    unittest.main()
