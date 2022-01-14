import random
import unittest

import numpy as np

import CCAugmentation as cca
from CCAugmentation.tests import pipeline_helper


class DuplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_len = 5
        self.dups_num = 4
        self.data = list(pipeline_helper.generate_data(self.orig_len))
        self.op = cca.Duplicate(self.dups_num)
        self.result = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [self.op])

    def test_invalid_dup_number_fails(self):
        self.assertRaises(ValueError, lambda: cca.Duplicate(0))
        self.assertRaises(ValueError, lambda: cca.Duplicate(-1))

    def test_check_expected_samples_num_on_output(self):
        self.assertEqual(1, cca.Duplicate(1).get_output_samples_number_multiplier())
        self.assertEqual(5, cca.Duplicate(5).get_output_samples_number_multiplier())

    def test_produces_correct_number_of_tuples(self):
        self.assertEqual(self.orig_len * self.dups_num, len(self.result[0]))
        self.assertEqual(self.orig_len * self.dups_num, len(self.result[1]))

    def test_duplicates_are_equal(self):
        self.assertTrue(np.all(self.data[0][0] == self.result[0][0]))
        self.assertTrue(np.all(self.data[1][0] == self.result[1][0]))
        self.assertTrue(np.all(self.result[0][0] == self.result[0][:self.dups_num]))
        self.assertTrue(np.all(self.result[1][1] == self.result[1][:self.dups_num]))

    def test_values_are_preserved(self):
        self.assertFalse(np.all(self.result[0][0] == self.result[0][self.dups_num:]))
        self.assertFalse(np.all(self.result[1][0] == self.result[1][self.dups_num:]))


class DropoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_len = 10

    def _do_dropout(self, rate):
        data = pipeline_helper.generate_data(self.orig_len)
        op = cca.Dropout(rate)
        return pipeline_helper.run_ops_in_trivial_pipeline(data, [op])

    def test_invalid_dropout_rate_fails(self):
        self.assertRaises(ValueError, lambda: cca.Dropout(1))
        self.assertRaises(ValueError, lambda: cca.Dropout(-1))
        self.assertRaises(ValueError, lambda: cca.Dropout(5))

    def test_check_expected_samples_num_on_output(self):
        self.assertEqual(1.0, cca.Dropout(0).get_output_samples_number_multiplier())
        self.assertEqual(0.5, cca.Dropout(0.5).get_output_samples_number_multiplier())
        self.assertEqual(0.75, cca.Dropout(0.25).get_output_samples_number_multiplier())

    def test_no_dropout_full_data(self):
        result = self._do_dropout(0.0)
        self.assertEqual(self.orig_len, len(result[0]))
        self.assertEqual(self.orig_len, len(result[1]))

    def test_partial_dropout_partial_data(self):
        random.seed(0)
        result = self._do_dropout(0.5)
        self.assertLess(0, len(result[0]))
        self.assertGreater(self.orig_len, len(result[0]))
        self.assertLess(0, len(result[1]))
        self.assertGreater(self.orig_len, len(result[1]))


class OptimizeBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # 2x (10, 10), 2x (15, 10), 1x (5, 10)
        self.sizes = [(10, 10), (15, 10), (15, 10), (5, 10), (10, 10)]
        self.data = self._generate_data(self.sizes)

    @staticmethod
    def _generate_sample(width, height):
        imgs, dms = pipeline_helper.generate_data(1, width, height)
        return imgs[0], dms[0]

    @staticmethod
    def _generate_data(sizes):
        return list(zip(*[OptimizeBatchTests._generate_sample(w, h) for (w, h) in sizes]))

    def _optimize_batches(self, batch_size, buffer_size_limit):
        op = cca.OptimizeBatch(batch_size, buffer_size_limit)
        return pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])

    def test_invalid_batch_size_fails(self):
        self.assertRaises(ValueError, lambda: cca.OptimizeBatch(0))
        self.assertRaises(ValueError, lambda: cca.OptimizeBatch(-3))

    def test_invalid_buffer_size_limit_fails(self):
        self.assertRaises(ValueError, lambda: cca.OptimizeBatch(10, 0))
        self.assertRaises(ValueError, lambda: cca.OptimizeBatch(10, -1))

    def test_batch_1_works(self):
        result = self._optimize_batches(1, None)
        self.assertEqual(len(self.data[0]), len(result[0]))
        self.assertEqual(len(self.data[1]), len(result[1]))


if __name__ == '__main__':
    unittest.main()
