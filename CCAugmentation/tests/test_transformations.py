import unittest

import numpy as np

import CCAugmentation as cca
from CCAugmentation.tests import pipeline_helper


class TransformationTests(unittest.TestCase):
    def test_invalid_probability(self):
        self.assertRaises(ValueError, lambda: cca.Transformation(-1))
        self.assertRaises(ValueError, lambda: cca.Transformation(2))


def _assert_a_imgs_greater_than_b_imgs(case: unittest.TestCase, a_imgs, b_imgs):
    for i, (a, b) in enumerate(zip(a_imgs, b_imgs)):
        with case.subTest(i=i):
            case.assertGreater(a.shape[0], b.shape[0])
            case.assertGreater(a.shape[1], b.shape[1])
            case.assertEqual(a.shape[2], b.shape[2])


def _assert_a_dms_greater_than_b_dms(case: unittest.TestCase, a_dms, b_dms):
    for i, (a, b) in enumerate(zip(a_dms, b_dms)):
        with case.subTest(i=i):
            case.assertGreater(a.shape[0], b.shape[0])
            case.assertGreater(a.shape[1], b.shape[1])


class CropTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_len = 3
        self.data = pipeline_helper.generate_data(self.data_len)

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
        self.assertRaises(ValueError, lambda: cca.Crop(-10, -10, None, None))
        self.assertRaises(ValueError, lambda: cca.Crop(35, 0, None, None))
        self.assertRaises(ValueError, lambda: cca.Crop(0, 0, None, None))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, 1.2, 1.2))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, 0.4, -5))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, -0.3, 0.4))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, -0.1, -0.1))
        self.assertRaises(ValueError, lambda: cca.Crop(None, None, 0.0, 0.0))

    def test_ok_args(self):
        try:
            cca.Crop(5, 8, None, None)
            cca.Crop(1, 1, None, None)
            cca.Crop(None, None, 0.2, 0.2)
            cca.Crop(None, None, 0.4, 0.5)
        except ValueError as e:
            self.fail(str(e))

    def _test_smaller_output(self, op):
        transformed = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        _assert_a_imgs_greater_than_b_imgs(self, self.data[0], transformed[0])
        _assert_a_dms_greater_than_b_dms(self, self.data[1], transformed[1])

    def test_smaller_output_fixed(self):
        op = cca.Crop(4, 4)
        self._test_smaller_output(op)

    def test_smaller_output_ratio(self):
        op = cca.Crop(None, None, 0.5, 0.5)
        self._test_smaller_output(op)

    def test_borders_removed_after_centered_crop(self):
        op = cca.Crop(6, 6, centered=True)  # borders width = 2
        result = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        orig_img0, transf_img0 = self.data[0][0], result[0][0]
        orig_dm0, transf_dm0 = self.data[1][0], result[1][0]
        self.assertTrue(np.all(orig_img0[2:-2, 2:-2, :] == transf_img0))
        self.assertTrue(np.all(orig_dm0[2:-2, 2:-2] == transf_dm0))


class ScaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_len = 3
        self.data = pipeline_helper.generate_data(self.data_len)

    def test_invalid_args_no_size(self):
        self.assertRaises(ValueError, lambda: cca.Scale(None, None, None, None))

    def test_invalid_args_mixed_size(self):
        self.assertRaises(ValueError, lambda: cca.Scale(50, None, 0.2, None))
        self.assertRaises(ValueError, lambda: cca.Scale(None, 400, None, 0.5))

    def test_invalid_args_ambiguous_size(self):
        self.assertRaises(ValueError, lambda: cca.Scale(300, 200, 0.4, 0.3))
        self.assertRaises(ValueError, lambda: cca.Scale(300, 200, 0.1, None))
        self.assertRaises(ValueError, lambda: cca.Scale(None, 500, 0.7, 0.4))

    def test_invalid_args_out_of_bounds(self):
        self.assertRaises(ValueError, lambda: cca.Scale(-5, 42, None, None))
        self.assertRaises(ValueError, lambda: cca.Scale(-10, -10, None, None))
        self.assertRaises(ValueError, lambda: cca.Scale(35, 0, None, None))
        self.assertRaises(ValueError, lambda: cca.Scale(0, 0, None, None))
        self.assertRaises(ValueError, lambda: cca.Scale(None, None, 0.4, -5))
        self.assertRaises(ValueError, lambda: cca.Scale(None, None, -0.3, 0.4))
        self.assertRaises(ValueError, lambda: cca.Scale(None, None, -0.1, -0.1))
        self.assertRaises(ValueError, lambda: cca.Scale(None, None, 0.0, 0.0))

    def test_ok_args(self):
        try:
            cca.Scale(5, 8, None, None)
            cca.Scale(1, 1, None, None)
            cca.Scale(None, None, 0.2, 0.2)
            cca.Scale(None, None, 0.4, 0.5)
            cca.Scale(None, None, 1.2, 1.2)
        except ValueError as e:
            self.fail(str(e))

    def _test_smaller_output(self, op):
        transformed = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        _assert_a_imgs_greater_than_b_imgs(self, self.data[0], transformed[0])
        _assert_a_dms_greater_than_b_dms(self, self.data[1], transformed[1])

    def test_smaller_output_fixed(self):
        op = cca.Scale(5, 5)
        self._test_smaller_output(op)

    def test_smaller_output_ratio(self):
        op = cca.Scale(None, None, 0.5, 0.5)
        self._test_smaller_output(op)

    def _test_bigger_output(self, op):
        transformed = pipeline_helper.run_ops_in_trivial_pipeline(self.data, [op])
        _assert_a_imgs_greater_than_b_imgs(self, transformed[0], self.data[0])
        _assert_a_dms_greater_than_b_dms(self, transformed[1], self.data[1])

    def test_bigger_output_fixed(self):
        op = cca.Scale(20, 20)
        self._test_bigger_output(op)

    def test_bigger_output_ratio(self):
        op = cca.Scale(None, None, 1.5, 1.5)
        self._test_bigger_output(op)


class DownscaleTests(unittest.TestCase):
    def test_invalid_args_out_of_bounds(self):
        self.assertRaises(ValueError, lambda: cca.Downscale(1.3, 1.4))
        self.assertRaises(ValueError, lambda: cca.Downscale(1.5, 1.0))
        self.assertRaises(ValueError, lambda: cca.Downscale(0.8, 1.2))
        self.assertRaises(ValueError, lambda: cca.Downscale(0.0, 0.0))
        self.assertRaises(ValueError, lambda: cca.Downscale(-0.5, 0.3))

    def test_ok_args(self):
        try:
            cca.Downscale(0.3, 0.6)
            cca.Downscale(0.4, 0.4)
            cca.Downscale(0.001, 0.99)
        except ValueError as e:
            self.fail(str(e))

    def test_data_changed(self):
        orig = pipeline_helper.generate_data(1)
        orig_img, orig_dm = orig
        transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Downscale(0.5, 0.5)])
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(orig_img[0].shape, transf_img[0].shape)
        self.assertEqual(orig_dm[0].shape, transf_dm[0].shape)

        self.assertFalse(np.all(orig_img[0] == transf_img[0]))
        self.assertFalse(np.all(orig_dm[0] == transf_dm[0]))


class RotateTests(unittest.TestCase):
    def _assert_same_size(self, orig_img: np.ndarray, orig_dm: np.ndarray, transf_img: np.ndarray,
                          transf_dm: np.ndarray) -> None:
        self.assertEqual(orig_img.shape, transf_img.shape)
        self.assertEqual(orig_dm.shape, transf_dm.shape)

    def _assert_same_everything(self, orig_img: np.ndarray, orig_dm: np.ndarray, transf_img: np.ndarray,
                                transf_dm: np.ndarray) -> None:
        self._assert_same_size(orig_img, orig_dm, transf_img, transf_dm)
        self.assertTrue(np.all(orig_img == transf_img))
        self.assertTrue(np.all(orig_dm == transf_dm))

    def _assert_same_size_but_not_content(self, orig_img: np.ndarray, orig_dm: np.ndarray, transf_img: np.ndarray,
                                          transf_dm: np.ndarray) -> None:
        self._assert_same_size(orig_img, orig_dm, transf_img, transf_dm)
        self.assertFalse(np.all(orig_img == transf_img))
        self.assertFalse(np.all(orig_dm == transf_dm))

    def _assert_different_size(self, orig_img: np.ndarray, orig_dm: np.ndarray, transf_img: np.ndarray,
                               transf_dm: np.ndarray) -> None:
        self.assertNotEqual(orig_img.shape, transf_img.shape)
        self.assertNotEqual(orig_dm.shape, transf_dm.shape)
        # if size is different, further element-wise comparisons of content are ill-formed so there's no '_everything'

    def test_no_rotation(self):
        for angle in [0, 360, -360]:
            for expand in [True, False]:
                for width in [3, 4]:
                    for height in [5, 6]:
                        with self.subTest(angle=angle, expand=expand, width=width, height=height):
                            orig = pipeline_helper.generate_data(1, width, height)
                            orig_img, orig_dm = orig
                            transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Rotate(angle, expand)])
                            transf_img, transf_dm = list(transf[0]), list(transf[1])

                            self._assert_same_everything(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

    def test_90_rotation_same_shape(self):
        for angle in [90, -90, 270, 450]:
            for width in [3, 4]:
                for height in [5, 6]:
                    with self.subTest(angle=angle, width=width, height=height):
                        orig = pipeline_helper.generate_data(1, width, height)
                        orig_img, orig_dm = orig
                        transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Rotate(angle, False)])
                        transf_img, transf_dm = list(transf[0]), list(transf[1])

                        self._assert_same_size_but_not_content(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

    def test_90_rotation_new_shape(self):
        for angle in [90, -90, 270, 450]:
            for width in [3, 4]:
                for height in [5, 6]:
                    with self.subTest(angle=angle, width=width, height=height):
                        orig = pipeline_helper.generate_data(1, width, height)
                        orig_img, orig_dm = orig
                        transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Rotate(angle, True)])
                        transf_img, transf_dm = list(transf[0]), list(transf[1])

                        self._assert_different_size(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

    def test_180_rotation(self):
        for angle in [180, -180, 3*180]:
            for expand in [True, False]:
                for width in [3, 4]:
                    for height in [5, 6]:
                        with self.subTest(angle=angle, expand=expand, width=width, height=height):
                            orig = pipeline_helper.generate_data(1, width, height)
                            orig_img, orig_dm = orig
                            transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Rotate(angle, expand)])
                            transf_img, transf_dm = list(transf[0]), list(transf[1])

                            self._assert_same_size_but_not_content(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

                            y_offset = 1 if height % 2 == 0 else 0
                            x_offset = 1 if width % 2 == 0 else 0
                            for orig_y in range(y_offset, height):
                                for orig_x in range(x_offset, width):
                                    transf_y = height - orig_y - 1 + y_offset
                                    transf_x = width - orig_x - 1 + x_offset
                                    self.assertTrue(np.all(orig_img[0][orig_y][orig_x] ==
                                                           transf_img[0][transf_y][transf_x]))
                                    self.assertEqual(orig_dm[0][orig_y][orig_x], transf_dm[0][transf_y][transf_x])

    def test_other_rotations(self):
        for angle in [24, 89, -56, 444]:
            for expand in [True, False]:
                for width in [3, 4]:
                    for height in [5, 6]:
                        with self.subTest(angle=angle, expand=expand, width=width, height=height):
                            orig = pipeline_helper.generate_data(1, width, height)
                            orig_img, orig_dm = orig
                            transf = pipeline_helper.run_ops_in_trivial_pipeline(orig, [cca.Rotate(angle, expand)])
                            transf_img, transf_dm = list(transf[0]), list(transf[1])

                            if expand:
                                self._assert_different_size(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])
                            else:
                                self._assert_same_size_but_not_content(orig_img[0], orig_dm[0], transf_img[0],
                                                                       transf_dm[0])


class StandardizeSizeTests(unittest.TestCase):
    def test_invalid_args(self):
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([], 5))
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([1, -3/2], 10))
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([1], -5))
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([3/2], 0))
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([3/2, 3/2], 20))
        self.assertRaises(ValueError, lambda: cca.StandardizeSize([1], 10, ''))

    def test_ok_args(self):
        try:
            cca.StandardizeSize([1], 90)
            cca.StandardizeSize([5/4, 4/3], 25)
            cca.StandardizeSize([20], 80, 'scale')
            cca.StandardizeSize([3/2, 7/6, 1/1, 2/1, 3/1, 2/3], 70, 'crop')
        except ValueError as e:
            self.fail(str(e))
