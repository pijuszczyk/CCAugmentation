import unittest

import numpy as _np

import CCAugmentation as cca
from CCAugmentation.tests import pipeline_helper


class TransformationTests(unittest.TestCase):
    def test_invalid_probability(self):
        self.assertRaises(ValueError, lambda: cca.Transformation(-1))
        self.assertRaises(ValueError, lambda: cca.Transformation(2))


class CropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_len = 3
        self.data = list(pipeline_helper.generate_data(self.orig_len))

    def test_invalid_args_no_size(self):
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, None, None))

    def test_invalid_args_mixed_size(self):
        self.assertRaises(ValueError, lambda: cca.Crop(50, None, 0.2, None))
        self.assertRaises(ValueError, lambda: cca.Crop(None, 400, None, 0.5))

    def test_invalid_args_ambiguous_size(self):
        self.assertRaises(ValueError, lambda: cca.Crop(300, 200, 0.4, 0.3))
        self.assertRaises(ValueError, lambda: cca.Crop(300, 200, 0.1, None))
        self.assertRaises(ValueError, lambda: cca.Crop(None, 500, 0.7, 0.4))

    def test_invalid_args_out_of_bounds(self):
        self.assertRaises(ValueError, lambda: cca.Crop(-5, 42, None, None))
        self.assertRaises(ValueError, lambda: cca.Crop(35, 0, None, None))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, 0.4, -5))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, -0.3, 0.4))

    def test_smaller_output(self):
        op = cca.Crop(None, None, 0.5, 0.5)
        result = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        for i, (orig, transf) in enumerate(zip(self.data[0], result[0])):
            with self.subTest(i=i):
                self.assertGreater(orig.shape[0], transf.shape[0])
                self.assertGreater(orig.shape[1], transf.shape[1])
                self.assertEqual(orig.shape[2], transf.shape[2])

        for i, (orig, transf) in enumerate(zip(self.data[1], result[1])):
            with self.subTest(i=i):
                self.assertGreater(orig.shape[0], transf.shape[0])
                self.assertGreater(orig.shape[1], transf.shape[1])

    def test_borders_removed_after_centered_crop(self):
        op = cca.Crop(6, 6, centered=True)  # borders width = 2
        result = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        orig_img0, transf_img0 = self.data[0][0], result[0][0]
        orig_dm0, transf_dm0 = self.data[1][0], result[1][0]
        self.assertTrue(_np.all(orig_img0[2:-2, 2:-2, :] == transf_img0))
        self.assertTrue(_np.all(orig_dm0[2:-2, 2:-2] == transf_dm0))
