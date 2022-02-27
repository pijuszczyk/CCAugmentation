import csv as _csv
import math as _math
import os as _os
import typing as _typing
from glob import glob as _glob

import cv2 as _cv2
import numpy as _np
from scipy.io import loadmat as _loadmat
from scipy.ndimage.filters import gaussian_filter as _gaussian_filter
from scipy.spatial import KDTree as _KDTree

from .common import _IMG_TYPE, _DM_TYPE, _IMG_DM_PAIR_TYPE

_HEADS_POS_ITER_TYPE = _np.ndarray


def get_density_map_gaussian(im: _IMG_TYPE, points: _HEADS_POS_ITER_TYPE) -> _DM_TYPE:
    """
    Create a Gaussian density map from the points.
    Credits:
    https://github.com/ZhengPeng7/Multi_column_CNN_in_Keras/blob/master/data_preparation/get_density_map_gaussian.py

    Args:
        im: Original image, used only for getting needed shape of the density map.
        points: List of (X, Y) tuples that point at where human heads are located in a picture.

    Returns:
        Density map constructed from the points.
    """
    im_density = _np.zeros_like(im[:, :, 0], dtype=_np.float64)
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
        H = _np.multiply(_cv2.getGaussianKernel(f_sz, sigma), (_cv2.getGaussianKernel(f_sz, sigma)).T)
        x = min(w-1, max(0, abs(int(_math.floor(points[j, 0])))))
        y = min(h-1, max(0, abs(int(_math.floor(points[j, 1])))))
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
            H = _np.multiply(
                _cv2.getGaussianKernel(y2h - y1h + 1, sigma),
                (_cv2.getGaussianKernel(x2h - x1h + 1, sigma)).T
            )
        im_density[y1:y2, x1:x2] += H

    return im_density


def get_density_map_geometry_adaptive_gaussian_kernel(im: _IMG_TYPE, points: _HEADS_POS_ITER_TYPE) -> _DM_TYPE:
    """
    Create a density map using Geometry-Adaptive Gaussian kernel proposed in:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf

    Args:
        im: Original image, used only for getting needed shape of the density map.
        points: List of (X, Y) tuples that point at where human heads are located in a picture.

    Returns:
        Density map constructed from the points.
    """
    h, w = im.shape[:2]
    im_density = _np.zeros((h, w), dtype=_np.float32)
    if len(points) == 0:
        return im_density

    for point in points:
        x = _np.clip(round(point[0]), 0, w-1)
        y = _np.clip(round(point[1]), 0, h-1)
        im_density[y, x] = 1

    if len(points) == 1:
        sigma = 0
        im_density += _gaussian_filter(points, sigma, mode='constant')
        return im_density

    tree = _KDTree(points, leafsize=2048)
    distances, _ = tree.query(points, k=3)

    beta = 0.3

    for i, point in enumerate(points):
        point_2d = _np.zeros(im_density.shape, dtype=_np.floa32)
        point_2d[point[1], point[0]] = 1.
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * beta
        im_density += _gaussian_filter(points, sigma, mode='constant')
    return im_density


class Loader:
    """
    Abstract base loader that should return an iterable of samples, either images, lists of points or density maps.
    """
    def load(self):
        """ Method that must be implemented in the subclasses, returning an iterable of samples """
        raise NotImplementedError("load not implemented in the child class")

    @staticmethod
    def _prepare_args(local_vars: _typing.Dict[str, _typing.Any]):
        """ Simple method that removes unwanted 'self' variable from the set that will be stored for loading and
        saving pipelines"""
        return {k: v for k, v in local_vars.items() if k != 'self'}

    def get_number_of_loadable_samples(self) -> _typing.Optional[int]:
        """
        Return number of samples from the dataset that can and will be loaded by the loader, or None if it's unknown.

        Returns:
            Number of samples that can be loaded, including the already loaded ones.
        """
        return None


class BasicImageFileLoader(Loader):
    """
    Loader for images stored in image files. Allows reading any files that opencv-python can handle - e.g. JPG, PNG.
    """
    def __init__(self, img_paths: _typing.Collection[str]):
        """
        Create a new image loader that reads all image files from paths.

        Args:
            img_paths: Paths to all images that are to be loaded.
        """
        if len(img_paths) == 0:
            raise ValueError("At least one path must be specified")

        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.img_paths = img_paths

    def get_number_of_loadable_samples(self) -> int:
        """
        Get number of images to load, according to number of specified paths.

        Returns:
            Number of images.
        """
        return len(self.img_paths)

    def load(self) -> _typing.Generator[_IMG_TYPE, None, None]:
        """
        Load all images based on provided paths to files.

        Returns:
            Generator of images in BGR format.
        """
        for path in self.img_paths:
            yield _cv2.imread(path, _cv2.IMREAD_COLOR)


class ImageFileLoader(BasicImageFileLoader):
    """
    Loader for all images of some type in a given directory.
    """
    def __init__(self, img_dir: str, file_extension: str = "jpg"):
        """
        Create a new image loader that reads all the images with specified file extension in a given directory.

        Args:
            img_dir: Directory to be searched.
            file_extension: Desired extension of files to be loaded.
        """
        local = locals().copy()
        paths = sorted(_glob(_os.path.join(img_dir, f"*.{file_extension}")))
        BasicImageFileLoader.__init__(self, paths)
        self.args = self._prepare_args(local)


_MATLAB_GT_TYPE = _typing.Dict[str, _typing.Any]
_MATLAB_GT_GETTER_TYPE = _typing.Callable[[_MATLAB_GT_TYPE], _HEADS_POS_ITER_TYPE]


class BasicGTPointsMatFileLoader(Loader):
    """
    Loader for ground truth data stored as lists of head positions in Matlab files.
    """
    def __init__(self, gt_paths: _typing.Collection[str], getter: _MATLAB_GT_GETTER_TYPE):
        """
        Create a loader that loads all data from the provided file paths using a given getter.

        Args:
            gt_paths: Paths of files that are to be read.
            getter: Lambda that takes Matlab file content and returns list of head positions in form of (X, Y) tuples.
        """
        if len(gt_paths) == 0:
            raise ValueError("At least one path must be specified")

        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.gt_paths = gt_paths
        self.getter = getter

    def get_number_of_loadable_samples(self) -> int:
        """
        Get number of GTs to load, according to number of specified paths.

        Returns:
            Number of GTs.
        """
        return len(self.gt_paths)

    def load(self) -> _typing.Generator[_HEADS_POS_ITER_TYPE, None, None]:
        """
        Load all Matlab files from paths.

        Returns:
            Generator of lists of head positions - (X, Y) tuples.
        """
        for path in self.gt_paths:
            yield self.getter(_loadmat(path))


class GTPointsMatFileLoader(BasicGTPointsMatFileLoader):
    """
    Loader for head positions in all Matlab files in a given directory.
    """
    def __init__(self, gt_dir: str, getter: _MATLAB_GT_GETTER_TYPE, file_extension: str = "mat"):
        """
        Create a loader that searches for files with specified extension in a given directory and loads them.

        Args:
            gt_dir: Directory to be searched.
            file_extension: Desired file extension of Matlab files.
        """
        local = locals().copy()
        paths = sorted(_glob(_os.path.join(gt_dir, f"*.{file_extension}")))
        BasicGTPointsMatFileLoader.__init__(self, paths, getter)
        self.args = self._prepare_args(local)


class BasicDensityMapCSVFileLoader(Loader):
    """
    Loader for density maps stored in separate CSV files.
    """
    def __init__(self, dm_paths: _typing.Collection[str]):
        """
        Create a loader that loads density maps at specified paths.

        Args:
            dm_paths: Paths to CSV files with density maps.
        """
        if len(dm_paths) == 0:
            raise ValueError("At least one path must be specified")

        Loader.__init__(self)
        self.args = self._prepare_args(locals())
        self.dm_paths = dm_paths

    def get_number_of_loadable_samples(self) -> int:
        """
        Get number of density maps to load, according to number of specified paths.

        Returns:
            Number of density maps.
        """
        return len(self.dm_paths)

    def load(self) -> _typing.Generator[_DM_TYPE, None, None]:
        """
        Load all density maps from all specified paths.

        Returns:
            Generator of density maps.
        """
        for path in self.dm_paths:
            den_map = []
            with open(path, 'r', newline='') as f:
                for row in _csv.reader(f):
                    den_row = []
                    for cell in row:
                        den_row.append(float(cell))
                    den_map.append(den_row)

            yield _np.array(den_map)


class DensityMapCSVFileLoader(BasicDensityMapCSVFileLoader):
    """
    Loader for density maps stored in all CSV files in a given directory.
    """
    def __init__(self, den_map_dir: str, file_extension: str = "csv"):
        """
        Create a loader that searches for files with the given extension in the given directory and loads them.

        Args:
            den_map_dir: Directory to be searched.
            file_extension: Desired extension of files to be loaded.
        """
        local = locals().copy()
        paths = sorted(_glob(_os.path.join(den_map_dir, f"*.{file_extension}")))
        BasicDensityMapCSVFileLoader.__init__(self, paths)
        self.args = self._prepare_args(local)


class VariableLoader(Loader):
    """
    Loader that loads from a variable (list or array) instead of file. May be useful when connecting pipelines.
    """
    def __init__(self, data: _typing.Collection[_typing.Any]):
        """
        Create a loader that reads from a variable (list or array most probably) and yields the results.

        Args:
            data: Iterable that has len() with either images or density maps.
        """
        Loader.__init__(self)
        # saving dataset variables (possibly consisting of thousands of samples) to a json file would be dangerous
        self.args = None
        self.data = data

    def get_number_of_loadable_samples(self) -> int:
        """
        Return length of the dataset in the variable.

        Returns:
            Number of samples.
        """
        return len(self.data)

    def load(self) -> _typing.Generator[_typing.Any, None, None]:
        """
        Read the variable and yield samples one by one.

        Returns:
            Generator of either images or density maps.
        """
        for sample in self.data:
            yield sample


class ConcatenatingLoader(Loader):
    """
    Loader that doesn't perform any loading on its own but rather concatenates samples from a few sources.
    """
    def __init__(self, loaders: _typing.Iterable[Loader]):
        """
        Create a loader that concatenates loading results from a few loaders.

        Args:
            loaders: Loaders whose results will be concatenated.
        """
        Loader.__init__(self)
        self.args = [{'name': loader.__class__.__name__, 'args': loader.args} for loader in loaders]
        self.loaders = loaders

    def get_number_of_loadable_samples(self) -> int:
        """
        Get number of samples to load throughout loaders.

        Returns:
            Cumulative number of samples.
        """
        return sum([loader.get_number_of_loadable_samples() for loader in self.loaders])

    def load(self) -> _typing.Generator[_typing.Any, None, None]:
        """
        Load all samples from all connected loaders.

        Returns:
            Generator of samples, be it images, GT point lists or density maps.
        """
        for loader in self.loaders:
            for sample in loader:
                yield sample


class CombinedLoader(Loader):
    """
    Loader that should be primarily used with a pipeline - zips or combines an iterable of images with an iterable of
    density maps (be it straight from a loader or from transformed on-the-fly GT points).
    """
    def __init__(self, img_loader: Loader, gt_loader: _typing.Optional[Loader],
                 den_map_loader: _typing.Optional[Loader] = None, gt_to_dm_converter: str = 'gaussian'):
        """
        Create a combined loader. Either `gt_loader` or `den_map_loader` must be specified (but not both) in order to
        provide density maps related to the images loaded using `img_loader`. If using ground truth loader, you may
        also customize how the points will be converted to a density map by specifying `gt_to_dm_converter`.

        Args:
            img_loader: Loader that provides an iterable of images.
            gt_loader: Loader that provides an iterable of lists of points.
            den_map_loader: Loader that provides an iterable of density maps.
            gt_to_dm_converter: How the points from the ground truth will be converted to a density map, should such
                conversion happen - when using gt_loader.
        """
        if (gt_loader is None) == (den_map_loader is None):
            raise ValueError("One and only one loader for target must be selected")
        if gt_to_dm_converter not in ['gaussian', 'geometry-adaptive gaussian']:
            raise ValueError("Only two converters are available: 'gaussian' and 'geometry-adaptive gaussian'. Please"
                             "use one of them.")

        Loader.__init__(self)
        self.args = {
            'img_loader': {'name': img_loader.__class__.__name__, 'args': img_loader.args},
            'gt_loader': None if gt_loader is None else {
                'name': img_loader.__class__.__name__, 'args': img_loader.args
            },
            'den_map_loader': None if den_map_loader is None else {
                'name': den_map_loader.__class__.__name__, 'args': den_map_loader.args
            }
        }
        self.img_loader = img_loader
        self.gt_loader = gt_loader
        self.den_map_loader = den_map_loader
        self.gt_to_dm_converter = get_density_map_gaussian if 'gaussian' \
            else get_density_map_geometry_adaptive_gaussian_kernel

    def get_number_of_loadable_samples(self) -> int:
        """
        Get number of full samples (img+DM pairs).

        Returns:
            Number of samples.
        """
        if self.den_map_loader is None:
            return min(self.img_loader.get_number_of_loadable_samples(),
                       self.gt_loader.get_number_of_loadable_samples())
        else:
            return min(self.img_loader.get_number_of_loadable_samples(),
                       self.den_map_loader.get_number_of_loadable_samples())

    def load(self) -> _typing.Generator[_IMG_DM_PAIR_TYPE, None, None]:
        """
        Load and return all img+DM pairs, one by one. If a GT loader is used instead of a DM loader, first transform
        GT points to a density map.

        Returns:
            Generator of img+DM pairs.
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
                    den_map = self.gt_to_dm_converter(img, gt)
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
