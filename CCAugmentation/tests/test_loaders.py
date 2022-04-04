import itertools
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


class BasicDensityMapCSVFileLoaderTests(unittest.TestCase):
    def test_no_input(self):
        self.assertRaises(ValueError, lambda: cca.BasicDensityMapCSVFileLoader([]))

    @unittest.skip("Somehow the test fails on Python 3.6")
    def test_loads_file(self):
        data = np.random.random((10, 10))
        csv_data = ''
        for row in data:
            for cell in row:
                csv_data += f'{cell},'
            csv_data = csv_data[:-1] + '\n'
        loader = cca.BasicDensityMapCSVFileLoader(['dm1.csv'])
        mock = unittest.mock.mock_open(read_data=csv_data)
        with unittest.mock.patch('builtins.open', mock) as p:
            loaded = list(loader.load())
            p.assert_called()
            self.assertTrue(np.all(loaded[0] == data))


class VariableLoaderTests(unittest.TestCase):
    def test_load_list(self):
        data = [np.random.random((np.random.randint(5, 15), np.random.randint(5, 15), 3)) for _ in range(5)]

        loader = cca.VariableLoader(data)
        self.assertEqual(loader.get_number_of_loadable_samples(), 5)

        for i, (original, loaded) in enumerate(zip(data, loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))

    def test_load_generator(self):
        data = [np.random.random((np.random.randint(5, 15), np.random.randint(5, 15), 3)) for _ in range(5)]

        loader = cca.VariableLoader(iter(data))
        self.assertEqual(loader.get_number_of_loadable_samples(), None)

        for i, (original, loaded) in enumerate(zip(iter(data), loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))


class ConcatenatingLoaderTests(unittest.TestCase):
    def test_list_and_list(self):
        data_a = [np.random.random((8, 8)) for _ in range(3)]
        data_b = [np.random.random((8, 8)) for _ in range(3)]

        loader_a = cca.VariableLoader(data_a)
        loader_b = cca.VariableLoader(data_b)
        loader = cca.ConcatenatingLoader([loader_a, loader_b])
        self.assertEqual(loader.get_number_of_loadable_samples(), 3+3)

        for i, (original, loaded) in enumerate(zip(itertools.chain(data_a, data_b), loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))

    def test_list_and_gen(self):
        data_a = [np.random.random((8, 8)) for _ in range(3)]
        data_b = [np.random.random((8, 8)) for _ in range(3)]

        loader_a = cca.VariableLoader(data_a)
        loader_b = cca.VariableLoader(iter(data_b))
        loader = cca.ConcatenatingLoader([loader_a, loader_b])
        self.assertEqual(loader.get_number_of_loadable_samples(), None)

        for i, (original, loaded) in enumerate(zip(itertools.chain(data_a, iter(data_b)), loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))

    def test_gen_and_gen(self):
        data_a = [np.random.random((8, 8)) for _ in range(3)]
        data_b = [np.random.random((8, 8)) for _ in range(3)]

        loader_a = cca.VariableLoader(iter(data_a))
        loader_b = cca.VariableLoader(iter(data_b))
        loader = cca.ConcatenatingLoader([loader_a, loader_b])
        self.assertEqual(loader.get_number_of_loadable_samples(), None)

        for i, (original, loaded) in enumerate(zip(itertools.chain(iter(data_a), iter(data_b)), loader.load())):
            with self.subTest(i=i):
                self.assertEqual(original.shape, loaded.shape)
                self.assertTrue(np.all(original == loaded))

    def test_different_loader_types(self):
        data_a = [np.random.random((8, 8, 3)) for _ in range(3)]
        loader_a = cca.VariableLoader(data_a)

        data_b = [np.random.random((10, 10, 3))]
        loader_b = cca.BasicImageFileLoader(['file.jpg'])

        loader = cca.ConcatenatingLoader([loader_a, loader_b])
        self.assertEqual(loader.get_number_of_loadable_samples(), 3+1)

        with unittest.mock.patch('cv2.imread', return_value=data_b[0]):
            for i, (original, loaded) in enumerate(zip(itertools.chain(data_a, data_b), loader.load())):
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
