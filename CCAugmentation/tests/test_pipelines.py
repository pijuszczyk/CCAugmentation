import unittest

import numpy as np

import CCAugmentation as cca
from CCAugmentation.tests import pipeline_helper


class PipelineResultsIteratorTests(unittest.TestCase):
    def test_invalid_total_samples(self):
        self.assertRaises(ValueError, lambda: cca.PipelineResultsIterator(zip(pipeline_helper.generate_data(1)), 0))
        self.assertRaises(ValueError, lambda: cca.PipelineResultsIterator(zip(pipeline_helper.generate_data(1)), -1))

    def test_iterates_correctly(self):
        imgs, dms = pipeline_helper.generate_data(5)
        data = zip(imgs, dms)
        iterator = cca.PipelineResultsIterator(zip(imgs, dms), 10, False)
        for i, (orig, from_iter) in enumerate(zip(data, iterator)):
            with self.subTest(i=i):
                self.assertTrue(np.all(orig[0] == from_iter[0]))
                self.assertTrue(np.all(orig[1] == from_iter[1]))


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples_num = 5
        self.data = pipeline_helper.generate_data(self.samples_num)

    def test_empty_pipeline(self):
        result = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [])
        self.assertEqual(self.samples_num, len(result[0]))
        self.assertEqual(self.samples_num, len(result[1]))

    def test_one_op_pipeline(self):
        result = pipeline_helper.run_ops_in_trivial_pipeline(
            self.data,
            [cca.LambdaTransformation(1.0, lambda img, dm: (img, dm))])
        self.assertEqual(self.samples_num, len(result[0]))
        self.assertEqual(self.samples_num, len(result[1]))

    def test_long_pipeline(self):
        result = pipeline_helper.run_ops_in_trivial_pipeline(
            self.data,
            [
                cca.LambdaTransformation(1.0, lambda img, dm: (img, dm)),
                cca.Crop(1, 1, probability=0.0),
                cca.Duplicate(3),
                cca.FlipLR(probability=0.0),
                cca.Rotate(0, probability=0.0)
            ])
        self.assertEqual(self.samples_num * 3, len(result[0]))
        self.assertEqual(self.samples_num * 3, len(result[1]))

    def test_pipeline_effects(self):
        result = pipeline_helper.run_ops_in_trivial_pipeline(
            self.data,
            [
                cca.Crop(None, None, 0.8, 0.8),  # to 8x8
                cca.Crop(None, None, 0.75, 0.75),  # to 6x6
                cca.Crop(None, None, 0.5, 0.5)  # to 3x3
            ])
        for i, (img, dm) in enumerate(zip(*result)):
            with self.subTest(i=i):
                self.assertEqual(img.shape, (3, 3, 3))
                self.assertEqual(dm.shape, (3, 3))
