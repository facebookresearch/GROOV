# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import importlib.resources

import sys 
sys.path.append("..") 
import eval
from tests import test_data


class TestEval(unittest.TestCase):

    # Test the computation of rank-based metrics
    def test_rank_metrics(self, ks=[1, 4]):

        with importlib.resources.open_text(test_data, "gold.jsonl") as gold_file:
            
            gold_records = eval.load_data(gold_file.name)

            # 1. no matching
            with importlib.resources.open_text(
                test_data, "pred_miss.jsonl"
                ) as guess_file:

                guess_records = eval.load_data(guess_file.name)

                # compute evaluation metrics
                result = eval.compute(gold_records, guess_records, ks)
                
                self.assertEqual(result["precision"], 0.0)
                self.assertEqual(result["recall"], 0.0)

                self.assertEqual(result["precision@1"], 0.0)
                self.assertEqual(result["precision@4"], 0.0)
                self.assertEqual(result["recall@4"], 0.0)
            
            # 2. partial matching
            with importlib.resources.open_text(
                test_data, "pred_part.jsonl"
            ) as guess_file:

                guess_records = eval.load_data(guess_file.name)

                # compute evaluation metrics
                result = eval.compute(gold_records, guess_records, ks)
                
                self.assertAlmostEqual(result["precision"], 1 / 2)
                self.assertAlmostEqual(result["recall"], 1 / 2)
                self.assertAlmostEqual(result["f1"], 1 / 2)

                self.assertEqual(result["precision@1"], 0.0)
                self.assertAlmostEqual(result["precision@4"], 1 / 2)
                self.assertAlmostEqual(result["recall@4"], 1 / 2)
                self.assertAlmostEqual(result["f1@4"], 1 / 2)

            # 3. all correct prediction
            with importlib.resources.open_text(
                test_data, "pred_all.jsonl"
            ) as guess_file:

                guess_records = eval.load_data(guess_file.name)

                # compute evaluation metrics
                result = eval.compute(gold_records, guess_records, ks)
                
                self.assertEqual(result["precision"], 1.0)
                self.assertAlmostEqual(result["recall"], 3 / 4)
                self.assertAlmostEqual(result["f1"], 6 / 7)

                self.assertEqual(result["precision@1"], 1.0)
                self.assertAlmostEqual(result["precision@4"], 3 / 4)
                self.assertAlmostEqual(result["recall@4"], 3 / 4)
                self.assertAlmostEqual(result["f1@4"], 3 / 4)


    # Test the computation of rank-independent metrics
    def test_stat_metrics(self):

        with importlib.resources.open_text(test_data, "gold.jsonl") as gold_file:
            
            gold_records = eval.load_data(gold_file.name)

            with importlib.resources.open_text(
                test_data, "pred_part.jsonl"
            ) as guess_file:

                guess_records = eval.load_data(guess_file.name)
                result = eval.compute(gold_records, guess_records)

                self.assertAlmostEqual(result["precision"], 1 / 2)
                self.assertAlmostEqual(result["recall"], 1 / 2)
                self.assertAlmostEqual(result["f1"], 1 / 2)


    # Test the computation of aggregated metrics among multiple samples
    def test_average_metrics(self):

        with importlib.resources.open_text(test_data, "gold_multi.jsonl") as gold_file:

            gold_records = eval.load_data(gold_file.name)

            with importlib.resources.open_text(
                test_data, "pred_multi.jsonl"
            ) as guess_file:

                guess_records = eval.load_data(guess_file.name)
                result = eval.compute(gold_records, guess_records)

                self.assertAlmostEqual(result["precision"], 1 / 3)
                self.assertAlmostEqual(result["recall"], 1 / 4)
                self.assertAlmostEqual(result["f1"], 2 / 7)


if __name__ == '__main__':
    unittest.main()