import unittest
import unittest.mock

import numpy as np

import CCAugmentation as cca


class BasicImageFileLoaderTests(unittest.TestCase):
    def test_no_input(self):
        self.assertRaises(ValueError, lambda: cca.BasicImageFileLoader([]))

    def test_loads_file(self):
        data = np.random.random((10, 10, 3))
        loader = cca.BasicImageFileLoader(['file.jpg'])
        with unittest.mock.patch('cv2.imread', return_value=data) as p:
            loaded = list(loader.load())
            p.assert_called()
            self.assertTrue(np.all(loaded[0] == data))


class VariableLoaderTests(unittest.TestCase):
    def test_load_list(self):
        data = [np.random.random((np.random.randint(5, 15), np.random.randint(5, 15), 3)) for _ in range(5)]
        loader = cca.VariableLoader(data)
        for i, (original, loaded) in enumerate(zip(data, loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))


class CombinedLoaderTests(unittest.TestCase):
    def test_invalid_label_loaders(self):
        with unittest.mock.patch('glob._iglob', return_value=['img1.jpg']):
            img_loader = cca.ImageFileLoader('img/')
        with unittest.mock.patch('glob._iglob', return_value=['gt1.mat']):
            gt_loader = cca.GTPointsMatFileLoader('ground_truth/', lambda v: v["annPoints"])
        with unittest.mock.patch('glob._iglob', return_value=['dm1.csv']):
            dm_loader = cca.DensityMapCSVFileLoader('dm/')
        self.assertRaises(ValueError, lambda: cca.CombinedLoader(img_loader, gt_loader, dm_loader))

    def test_invalid_converter(self):
        with unittest.mock.patch('glob._iglob', return_value=['img1.jpg']):
            img_loader = cca.ImageFileLoader('img/')
        with unittest.mock.patch('glob._iglob', return_value=['gt1.mat']):
            gt_loader = cca.GTPointsMatFileLoader('ground_truth/', lambda v: v["annPoints"])
        self.assertRaises(ValueError, lambda: cca.CombinedLoader(img_loader, gt_loader, gt_to_dm_converter='wroong'))


if __name__ == '__main__':
    unittest.main()
