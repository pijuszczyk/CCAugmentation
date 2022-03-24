import itertools
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
        transformed = pipeline_helper.run_op_in_trivial_pipeline(self.data, op)
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
        result = pipeline_helper.run_op_in_trivial_pipeline(self.data, op)
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
        transformed = pipeline_helper.run_op_in_trivial_pipeline(self.data, op)
        _assert_a_imgs_greater_than_b_imgs(self, self.data[0], transformed[0])
        _assert_a_dms_greater_than_b_dms(self, self.data[1], transformed[1])

    def test_smaller_output_fixed(self):
        op = cca.Scale(5, 5)
        self._test_smaller_output(op)

    def test_smaller_output_ratio(self):
        op = cca.Scale(None, None, 0.5, 0.5)
        self._test_smaller_output(op)

    def _test_bigger_output(self, op):
        transformed = pipeline_helper.run_op_in_trivial_pipeline(self.data, op)
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
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Downscale(0.5, 0.5))
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
                            transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Rotate(angle, expand))
                            transf_img, transf_dm = list(transf[0]), list(transf[1])

                            self._assert_same_everything(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

    def test_90_rotation_same_shape(self):
        for angle in [90, -90, 270, 450]:
            for width in [3, 4]:
                for height in [5, 6]:
                    with self.subTest(angle=angle, width=width, height=height):
                        orig = pipeline_helper.generate_data(1, width, height)
                        orig_img, orig_dm = orig
                        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Rotate(angle, False))
                        transf_img, transf_dm = list(transf[0]), list(transf[1])

                        self._assert_same_size_but_not_content(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])

    def test_90_rotation_new_shape(self):
        for angle in [90, -90, 270, 450]:
            for width in [3, 4]:
                for height in [5, 6]:
                    with self.subTest(angle=angle, width=width, height=height):
                        orig = pipeline_helper.generate_data(1, width, height)
                        orig_img, orig_dm = orig
                        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Rotate(angle, True))
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
                            transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Rotate(angle, expand))
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
                            transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.Rotate(angle, expand))
                            transf_img, transf_dm = list(transf[0]), list(transf[1])

                            if expand:
                                self._assert_different_size(orig_img[0], orig_dm[0], transf_img[0], transf_dm[0])
                            else:
                                self._assert_same_size_but_not_content(orig_img[0], orig_dm[0], transf_img[0],
                                                                       transf_dm[0])


def _generate_variable_size_data(shapes):
    imgs, dms = [], []
    for w, h in shapes:
        img_list, dm_list = pipeline_helper.generate_data(1, w, h)
        imgs.append(img_list)
        dms.append(dm_list)
    return list(itertools.chain(*imgs)), list(itertools.chain(*dms))


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

    def test_adjust_ratio(self):
        orig = pipeline_helper.generate_data(1, 15, 9)
        orig_img, orig_dm = orig[0][0], orig[1][0]
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2, 5/4], 15))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        assert orig_img.shape == (9, 15, 3)
        assert orig_dm.shape == (9, 15)
        self.assertEqual(transf_img[0].shape, (10, 15, 3))
        self.assertEqual(transf_dm[0].shape, (10, 15))

    def test_ratio_already_ok(self):
        orig = pipeline_helper.generate_data(1, 15, 10)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2, 5/4], 15))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (10, 15, 3))
        self.assertEqual(transf_dm[0].shape, (10, 15))

    def test_beyond_last_ratio(self):
        orig = pipeline_helper.generate_data(1, 30, 10)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2], 150))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (100, 150, 3))
        self.assertEqual(transf_dm[0].shape, (100, 150))

    def test_beyond_first_ratio(self):
        orig = pipeline_helper.generate_data(1, 5, 10)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2], 10))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (10, 10, 3))
        self.assertEqual(transf_dm[0].shape, (10, 10))

    def test_change_base_size(self):
        orig = pipeline_helper.generate_data(1, 15, 10)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2, 5/4], 24))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (16, 24, 3))
        self.assertEqual(transf_dm[0].shape, (16, 24))

    def test_invert_ratio(self):
        orig = pipeline_helper.generate_data(1, 15, 10)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([2/3], 15))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (15, 10, 3))
        self.assertEqual(transf_dm[0].shape, (15, 10))

    def test_use_cropping(self):
        orig = pipeline_helper.generate_data(1, 15, 11)
        orig_img, orig_dm = orig[0][0], orig[1][0]
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.StandardizeSize([1, 3/2], 15, "crop"))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (10, 15, 3))
        self.assertEqual(transf_dm[0].shape, (10, 15))
        self.assertTrue(np.all(orig_img[1:, :, :] == transf_img) or np.all(orig_img[:-1, :, :] == transf_img))
        self.assertTrue(np.all(orig_dm[1:, :] == transf_dm) or np.all(orig_dm[:-1, :] == transf_dm))

    def test_many_samples(self):
        orig_img, orig_dm = _generate_variable_size_data([(15, 10), (20, 20), (21, 14), (10, 15)])
        transf = pipeline_helper.run_op_in_trivial_pipeline((orig_img, orig_dm), cca.StandardizeSize([1, 3/2], 30))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (20, 30, 3))
        self.assertEqual(transf_dm[0].shape, (20, 30))
        self.assertEqual(transf_img[1].shape, (30, 30, 3))
        self.assertEqual(transf_dm[1].shape, (30, 30))
        self.assertEqual(transf_img[2].shape, (20, 30, 3))
        self.assertEqual(transf_dm[2].shape, (20, 30))
        self.assertEqual(transf_img[3].shape, (30, 30, 3))
        self.assertEqual(transf_dm[3].shape, (30, 30))


class AutoStandardizeSizeTests(unittest.TestCase):
    def test_invalid_args(self):
        self.assertRaises(ValueError, lambda: cca.AutoStandardizeSize(-5))
        self.assertRaises(ValueError, lambda: cca.AutoStandardizeSize(0))
        self.assertRaises(ValueError, lambda: cca.AutoStandardizeSize(20, ""))
        self.assertRaises(ValueError, lambda: cca.AutoStandardizeSize(100, "sth"))

    def test_ok_args(self):
        try:
            cca.AutoStandardizeSize(90)
            cca.AutoStandardizeSize(250000)
            cca.AutoStandardizeSize(80, 'scale')
            cca.AutoStandardizeSize(70, 'crop')
        except ValueError as e:
            self.fail(str(e))

    def test_dataset_too_small_for_threshold(self):
        orig = pipeline_helper.generate_data(1, 15, 5)
        self.assertRaises(ValueError,
                          lambda: pipeline_helper.run_op_in_trivial_pipeline(orig, cca.AutoStandardizeSize(40)))

    def test_one_sample(self):
        orig = pipeline_helper.generate_data(1, 150, 105)
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.AutoStandardizeSize(1))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        self.assertEqual(transf_img[0].shape, (100, 150, 3))
        self.assertEqual(transf_dm[0].shape, (100, 150))

    def test_no_good_ratios(self):
        orig = _generate_variable_size_data([(150, 100), (100, 150)])
        self.assertRaises(ValueError,
                          lambda: pipeline_helper.run_op_in_trivial_pipeline(orig, cca.AutoStandardizeSize(2)))

    def test_one_dominant_ratio(self):
        orig = _generate_variable_size_data([(150, 100), (145, 99), (102, 95)])
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.AutoStandardizeSize(2))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        target_img_shape = (68, 102, 3)
        target_dm_shape = (68, 102)
        for i in range(len(transf_img)):
            with self.subTest(i=i):
                self.assertEqual(transf_img[i].shape, target_img_shape)
                self.assertEqual(transf_dm[i].shape, target_dm_shape)


class OmitDownscalingPixelsTests(unittest.TestCase):
    def test_invalid_args(self):
        self.assertRaises(ValueError, lambda: cca.OmitDownscalingPixels(-2))
        self.assertRaises(ValueError, lambda: cca.OmitDownscalingPixels(y_factor=-0.5))
        self.assertRaises(ValueError, lambda: cca.OmitDownscalingPixels(0, 0))

    def test_cut_pixels(self):
        for orig_size in [7, 8, 9, 10, 12, 15, 16, 18, 20]:
            for maxpool_downscale_ratio in [1, 2, 3, 4, 8]:
                with self.subTest(orig_size=orig_size, maxpool_downscale_ratio=maxpool_downscale_ratio):
                    orig = pipeline_helper.generate_data(1, orig_size, orig_size)
                    op = cca.OmitDownscalingPixels(maxpool_downscale_ratio, maxpool_downscale_ratio)
                    transf = pipeline_helper.run_op_in_trivial_pipeline(orig, op)
                    transf_img, transf_dm = list(transf[0])[0], list(transf[1])[0]

                    adjusted_size = orig_size - orig_size % maxpool_downscale_ratio
                    self.assertEqual(transf_img.shape, (adjusted_size, adjusted_size, 3))
                    self.assertEqual(transf_dm.shape, (adjusted_size, adjusted_size))


class NormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        # variably shaped data useful for additional checks if CCA can handle it in samplewise and fixed range norm.
        self.var_shape_orig = _generate_variable_size_data([(4, 3), (5, 9), (7, 7)])
        self.var_shape_orig_img, self.var_shape_orig_dm = self.var_shape_orig
        # consistently shaped data required for featurewise normalization
        self.consist_shape_orig = pipeline_helper.generate_data(3, 5, 4)
        self.consist_shape_orig_img, self.consist_shape_orig_dm = self.consist_shape_orig
        # precision when comparing means or stds
        self.prec = 1e-2

    def _check_sw_means_stds(self, transf, mean_0: bool, std_1: bool, by_channel: bool):
        transf_img, transf_dm = list(transf[0]), list(transf[1])
        means_stds_axes = (0, 1) if by_channel else None

        for i in range(len(transf_img)):
            with self.subTest(i=i):
                self.assertTrue(np.all(self.var_shape_orig_dm[i] == transf_dm[i]))
                if mean_0:
                    self.assertTrue(np.allclose(np.mean(transf_img[i], means_stds_axes), 0.0, atol=self.prec))
                else:
                    pass  # mean changes after std norm
                if std_1:
                    self.assertTrue(np.allclose(np.std(transf_img[i], means_stds_axes), 1.0, atol=self.prec))
                else:
                    self.assertTrue(np.allclose(np.std(transf_img[i], means_stds_axes),
                                                np.std(self.var_shape_orig_img[i], means_stds_axes), atol=self.prec))

    def _check_fw_means_stds(self, transf, mean_0: bool, std_1: bool, by_channel: bool):
        transf_img, transf_dm = list(transf[0]), list(transf[1])
        means_stds_axes = (0, 1, 2) if by_channel else None

        self.assertTrue(np.all(np.array(self.consist_shape_orig_dm) == np.array(transf_dm)))

        if mean_0:
            self.assertTrue(np.allclose(np.mean(transf_img, means_stds_axes), 0.0, atol=self.prec))
        else:
            pass  # mean changes after std norm
        if std_1:
            self.assertTrue(np.allclose(np.std(transf_img, means_stds_axes), 1.0, atol=self.prec))
        else:
            self.assertTrue(np.allclose(np.std(transf_img, means_stds_axes),
                                        np.std(self.consist_shape_orig_img, means_stds_axes), atol=self.prec))

    def _check_fixed_range(self, transf, low: float, high: float):
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        for i in range(len(transf_img)):
            with self.subTest(i=i):
                self.assertTrue(np.all(self.var_shape_orig_dm[i] == transf_dm[i]))
                self.assertGreaterEqual(np.min(transf_img[i]), low)
                self.assertLessEqual(np.max(transf_img[i]), high)

    def test_invalid_args(self):
        self.assertRaises(ValueError, lambda: cca.Normalize("range_0_to_1", False, np.array([1])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_0_to_1", False, None, np.array([1])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_0_to_1", True, np.array([1, 2, 3])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_0_to_1", True, None, np.array([1, 2, 3])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_-1_to_1", False, np.array([1])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_-1_to_1", False, None, np.array([1])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_-1_to_1", True, np.array([1, 2, 3])))
        self.assertRaises(ValueError, lambda: cca.Normalize("range_-1_to_1", True, None, np.array([1, 2, 3])))
        self.assertRaises(ValueError, lambda: cca.Normalize(""))
        self.assertRaises(ValueError, lambda: cca.Normalize("samplewise_range"))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_samplewise_centering"))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", False, np.array([])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", False, np.array([1, 2])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", False, np.array([1, 2, 3])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", False, np.array([1, 2, 3, 4])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", True, np.array([])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", True, np.array([1])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", True, np.array([1, 2])))
        self.assertRaises(ValueError, lambda: cca.Normalize("featurewise_centering", True, np.array([1, 2, 3, 4])))

    def test_normalize_sw_bc_z_score(self):
        op = cca.Normalize("samplewise_z-score", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, True, True, True)

    def test_normalize_sw_nbc_z_score(self):
        op = cca.Normalize("samplewise_z-score", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, True, True, False)

    def test_normalize_sw_bc_centering(self):
        op = cca.Normalize("samplewise_centering", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, True, False, True)

    def test_normalize_sw_nbc_centering(self):
        op = cca.Normalize("samplewise_centering", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, True, False, False)

    def test_normalize_sw_bc_std_norm(self):
        op = cca.Normalize("samplewise_std_normalization", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, False, True, True)

    def test_normalize_sw_nbc_std_norm(self):
        op = cca.Normalize("samplewise_std_normalization", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_sw_means_stds(transf, False, True, False)

    def test_normalize_fw_bc_z_score(self):
        op = cca.Normalize("featurewise_z-score", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, True, True, True)

    def test_normalize_fw_nbc_z_score(self):
        op = cca.Normalize("featurewise_z-score", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, True, True, False)

    def test_normalize_fw_bc_centering(self):
        op = cca.Normalize("featurewise_centering", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, True, False, True)

    def test_normalize_fw_nbc_centering(self):
        op = cca.Normalize("featurewise_centering", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, True, False, False)

    def test_normalize_fw_bc_std_norm(self):
        op = cca.Normalize("featurewise_std_normalization", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, False, True, True)

    def test_normalize_fw_nbc_std_norm(self):
        op = cca.Normalize("featurewise_std_normalization", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.consist_shape_orig, op)
        self._check_fw_means_stds(transf, False, True, False)

    def test_normalize_0_1_bc(self):
        op = cca.Normalize("range_0_to_1", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_fixed_range(transf, 0.0, 1.0)

    def test_normalize_0_1_nbc(self):
        op = cca.Normalize("range_0_to_1", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_fixed_range(transf, 0.0, 1.0)

    def test_normalize__1_1_bc(self):
        op = cca.Normalize("range_-1_to_1", True)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_fixed_range(transf, -1.0, 1.0)

    def test_normalize__1_1_nbc(self):
        op = cca.Normalize("range_-1_to_1", False)
        transf = pipeline_helper.run_op_in_trivial_pipeline(self.var_shape_orig, op)
        self._check_fixed_range(transf, -1.0, 1.0)


class NormalizeDensityMapTests(unittest.TestCase):
    def test_invalid_args(self):
        self.assertRaises(ValueError, lambda: cca.NormalizeDensityMap(-1.0))

    def test_dm_is_multiplied(self):
        orig = _generate_variable_size_data([(5, 6), (3, 3)])
        orig_img, orig_dm = orig
        multiplier = 6.0
        prec = 1e-3
        transf = pipeline_helper.run_op_in_trivial_pipeline(orig, cca.NormalizeDensityMap(multiplier))
        transf_img, transf_dm = list(transf[0]), list(transf[1])

        for i in range(len(orig_img)):
            with self.subTest(i=i):
                self.assertTrue(np.all(transf_img[i] == orig_img[i]))
                self.assertTrue(np.allclose(transf_dm[i] / multiplier, orig_dm[i], atol=prec))
