import csv
import math
import os
from glob import glob

import cv2
import numpy as np
from scipy.io import loadmat


def get_density_map_gaussian(im, points):
    """
    Create a Gaussian density map from the points.
    Credits: https://github.com/ZhengPeng7/Multi_column_CNN_in_Keras/blob/master/data_preparation/get_density_map_gaussian.py

    :param im: Original image, used only for getting needed shape of the density map.
    :param points: List of (X, Y) tuples that point at where human heads are located in a picture.
    :return: Density map constructed from the points.
    """
    im_density = np.zeros_like(im[:, :, 0], dtype=np.float64)
    h, w = im_density.shape
    if points is None:
        return im_density
    if points.shape[0] == 1:
        x1 = max(0, min(w-1, round(points[0, 0])))
        y1 = max(0, min(h-1, round(points[0, 1])))
        im_density[y1, x1] = 255
        return im_density
    for j in range(points.shape[0]):
        f_sz = 15
        sigma = 4.0
        H = np.multiply(cv2.getGaussianKernel(f_sz, sigma), (cv2.getGaussianKernel(f_sz, sigma)).T)
        x = min(w-1, max(0, abs(int(math.floor(points[j, 0])))))
        y = min(h-1, max(0, abs(int(math.floor(points[j, 1])))))
        if x >= w or y >= h:
            continue
        x1 = x - f_sz//2 + 0
        y1 = y - f_sz//2 + 0
        x2 = x + f_sz//2 + 1
        y2 = y + f_sz//2 + 1
        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change_H = False
        if x1 < 0:
            dfx1 = abs(x1) + 0
            x1 = 0
            change_H = True
        if y1 < 0:
            dfy1 = abs(y1) + 0
            y1 = 0
            change_H = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True
        x1h, y1h, x2h, y2h = 1 + dfx1, 1 + dfy1, f_sz - dfx2, f_sz - dfy2
        if change_H is True:
            H = np.multiply(cv2.getGaussianKernel(y2h-y1h+1, sigma), (cv2.getGaussianKernel(x2h-x1h+1, sigma)).T)
        im_density[y1:y2, x1:x2] += H

    return im_density


class Loader:
    """
    Abstract base loader that should return an iterable of samples, either images, lists of points or density maps.
    """
    def load(self):
        """ Method that must be implemented in the subclasses, returning an iterable of samples """
        raise NotImplementedError("load not implemented in the child class")

    @staticmethod
    def _prepare_args(local_vars):
        """ Simple method that removes unwanted 'self' variable from the set that will be stored for loading and saving pipelines"""
        return {k: v for k, v in local_vars.items() if k != 'self'}

    def get_number_of_loadable_samples(self):
        """
        Return number of samples from the dataset that can and will be loaded by the loader, or None if it's unknown.

        :return: Number of samples that can be loaded, including the already loaded ones.
        """
        return None


class BasicImageFileLoader(Loader):
    """
    Loader for images stored in image files. Allows reading any files that opencv-python can handle - e.g. JPG, PNG.
    """
    def __init__(self, img_paths):
        """
        Create a new image loader that reads all image files from paths.

        :param img_paths: Paths to all images that are to be loaded.
        """
        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.img_paths = img_paths

    def get_number_of_loadable_samples(self):
        """
        Get number of images to load, according to number of specified paths.

        :return: Number of images.
        """
        return len(self.img_paths)

    def load(self):
        """
        Load all images based on provided paths to files.

        :return: Generator of images in BGR format.
        """
        for path in self.img_paths:
            yield cv2.imread(path, cv2.IMREAD_COLOR)


class ImageFileLoader(BasicImageFileLoader):
    """
    Loader for all images of some type in a given directory.
    """
    def __init__(self, img_dir, file_extension="jpg"):
        """
        Create a new image loader that reads all the images with specified file extension in a given directory.

        :param img_dir: Directory to be searched.
        :param file_extension: Desired extension of files to be loaded.
        """
        local = locals().copy()
        paths = sorted(glob(os.path.join(img_dir, f"*.{file_extension}")))
        BasicImageFileLoader.__init__(self, paths)
        self.args = self._prepare_args(local)


class BasicGTPointsMatFileLoader(Loader):
    """
    Loader for ground truth data stored as lists of head positions in Matlab files.
    """
    def __init__(self, gt_paths, getter):
        """
        Create a loader that loads all data from the provided file paths using a given getter.

        :param gt_paths: Paths of files that are to be read.
        :param getter: Lambda that takes Matlab file content and returns list of head positions in form of (X, Y) tuples.
        """
        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.gt_paths = gt_paths
        self.getter = getter

    def get_number_of_loadable_samples(self):
        """
        Get number of GTs to load, according to number of specified paths.

        :return: Number of GTs.
        """
        return len(self.gt_paths)

    def load(self):
        """
        Load all Matlab files from paths.

        :return: Generator of lists of head positions - (X, Y) tuples.
        """
        for path in self.gt_paths:
            yield self.getter(loadmat(path))


class GTPointsMatFileLoader(BasicGTPointsMatFileLoader):
    """
    Loader for head positions in all Matlab files in a given directory.
    """
    def __init__(self, gt_dir, getter, file_extension="mat"):
        """
        Create a loader that searches for files with specified extension in a given directory and loads them.

        :param gt_dir: Directory to be searched.
        :param file_extension: Desired file extension of Matlab files.
        """
        local = locals().copy()
        paths = sorted(glob(os.path.join(gt_dir, f"*.{file_extension}")))
        BasicGTPointsMatFileLoader.__init__(self, paths, getter)
        self.args = self._prepare_args(local)


class BasicDensityMapCSVFileLoader(Loader):
    """
    Loader for density maps stored in separate CSV files.
    """
    def __init__(self, dm_paths):
        """
        Create a loader that loads density maps at specified paths.

        :param dm_paths: Paths to CSV files with density maps.
        """
        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.dm_paths = dm_paths

    def get_number_of_loadable_samples(self):
        """
        Get number of density maps to load, according to number of specified paths.

        :return: Number of density maps.
        """
        return len(self.dm_paths)

    def load(self):
        """
        Load all density maps from all specified paths.

        :return: Generator of density maps.
        """
        for path in self.dm_paths:
            den_map = []
            with open(path, 'r', newline='') as f:
                for row in csv.reader(f):
                    den_row = []
                    for cell in row:
                        den_row.append(float(cell))
                    den_map.append(den_row)

            yield np.array(den_map)


class DensityMapCSVFileLoader(BasicDensityMapCSVFileLoader):
    """
    Loader for density maps stored in all CSV files in a given directory.
    """
    def __init__(self, den_map_dir, file_extension="csv"):
        """
        Create a loader that searches for files with the given extension in the given directory and loads them.

        :param den_map_dir: Directory to be searched.
        :param file_extension: Desired extension of files to be loaded.
        """
        local = locals().copy()
        paths = sorted(glob(os.path.join(den_map_dir, f"*.{file_extension}")))
        BasicDensityMapCSVFileLoader.__init__(self, paths)
        self.args = self._prepare_args(local)


class VariableLoader(Loader):
    """
    Loader that loads from a variable (list or array) instead of file. May be useful when connecting pipelines.
    """
    def __init__(self, data):
        """
        Create a loader that reads from a variable (list or array most probably) and yields the results.

        :param data: Iterable that has len() with either images or density maps.
        """
        self.args = None  # saving dataset variables, possibly consisting of thousands of samples, to a json file would be dangerous
        self.data = data

    def get_number_of_loadable_samples(self):
        """
        Return length of the dataset in the variable.

        :return: Number of samples.
        """
        return len(self.data)

    def load(self):
        """
        Read the variable and yield samples one by one.

        :return: Generator of either images or density maps.
        """
        for sample in self.data:
            yield sample


class ConcatenatingLoader(Loader):
    """
    Loader that doesn't perform any loading on its own but rather concatenates samples from a few sources.
    """
    def __init__(self, loaders):
        """
        Create a loader that concatenates loading results from a few loaders.

        :param loaders: Loaders whose results will be concatenated.
        """
        Loader.__init__(self)
        self.args = [{'name': loader.__class__.__name__, 'args': loader.args} for loader in loaders]
        self.loaders = loaders

    def get_number_of_loadable_samples(self):
        """
        Get number of samples to load throughout loaders.

        :return: Cumulative number of samples.
        """
        return sum([loader.get_number_of_loadable_samples() for loader in self.loaders])

    def load(self):
        """
        Load all samples from all connected loaders.

        :return: Generator of samples, be it images, GT point lists or density maps.
        """
        for loader in self.loaders:
            for sample in loader:
                yield sample


class CombinedLoader(Loader):
    """
    Loader that should be primarily used with a pipeline - zips or combines an iterable of images with an iterable of
    density maps (be it straight from a loader or from transformed on-the-fly GT points).
    """
    def __init__(self, img_loader, gt_loader, den_map_loader=None):
        """
        Create a combined loader. Either `gt_loader` or `den_map_loader` must be specified (but not both) in order to
        provide density maps related to the images loaded using `img_loader`.

        :param img_loader: Loader that provides an iterable of images.
        :param gt_loader: Loader that provides an iterable of lists of points.
        :param den_map_loader: Loader that provides an iterable of density maps.
        """
        if (gt_loader is None) == (den_map_loader is None):
            raise ValueError("One and only one loader for target must be selected")

        Loader.__init__(self)
        self.args = {
            'img_loader': {'name': img_loader.__class__.__name__, 'args': img_loader.args},
            'gt_loader': None if gt_loader is None else {'name': img_loader.__class__.__name__, 'args': img_loader.args},
            'den_map_loader': None if den_map_loader is None else {'name': den_map_loader.__class__.__name__, 'args': den_map_loader.args}
        }
        self.img_loader = img_loader
        self.gt_loader = gt_loader
        self.den_map_loader = den_map_loader

    def get_number_of_loadable_samples(self):
        """
        Get number of full samples (img+DM pairs).

        :return: Number of samples.
        """
        if self.den_map_loader is None:
            return min(self.img_loader.get_number_of_loadable_samples(), self.gt_loader.get_number_of_loadable_samples())
        else:
            return min(self.img_loader.get_number_of_loadable_samples(), self.den_map_loader.get_number_of_loadable_samples())

    def load(self):
        """
        Load and return all img+DM pairs, one by one. If a GT loader is used instead of a DM loader, first transform
        GT points to a density map.

        :return: Generator of img+DM pairs.
        """
        cnt = 0
        img_gen = self.img_loader.load()
        if self.den_map_loader is None:
            gt_gen = self.gt_loader.load()
            try:
                while True:
                    img = next(img_gen)
                    try:
                        gt = next(gt_gen)
                    except StopIteration:
                        raise ValueError(f"Missing ground truth for image {str(cnt)}")
                    den_map = get_density_map_gaussian(img, gt)
                    yield img, den_map
                    cnt += 1
            except StopIteration:
                pass
        else:
            dm_gen = self.den_map_loader.load()
            try:
                while True:
                    img = next(img_gen)
                    try:
                        den_map = next(dm_gen)
                    except StopIteration:
                        raise ValueError(f"Missing density map for image {str(cnt)}")
                    yield img, den_map
                    cnt += 1
            except StopIteration:
                pass
