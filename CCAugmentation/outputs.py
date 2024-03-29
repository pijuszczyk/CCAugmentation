import csv as _csv
import os as _os
import pickle as _pickle

import cv2 as _cv2
import matplotlib.pyplot as _plt

from .operations import Operation


class Output(Operation):
    """
    Abstract base class for operations that don't modify the processed samples but generate some output.
    Note that when using augmentation on-the-fly, using Output classes is not needed and one may just collect the data
    from the pipeline repeatedly.
    """
    def __init__(self):
        """ Trivial constructor """
        Operation.__init__(self)
        self.args = self._prepare_args(locals())

    def execute(self, images_and_density_maps):
        """ See `output` """
        return self.output(images_and_density_maps)

    def output(self, images_and_density_maps):
        """ Abstract method that must be implemented in the subclasses. Takes and returns an iterable of unchanged
        img+DM pairs, while generating some side-effect output """
        raise NotImplementedError("output method not implemented in the child class")


class Demonstrate(Output):
    """
    Output that shows a few examples of preprocessed img+DM pairs using GUI.
    """
    def __init__(self, max_examples=None, show_density_map=True, density_map_cmap=None):
        """
        Specify demonstration looks.

        Args:
            max_examples: Number of examples that will be shown. The examples are taken from the beginning of iteration.
                If `max_examples` exceeds the actual number of samples, the function ends earlier.
            show_density_map: Whether to show the density map right next to the preprocessed image.
            density_map_cmap: CMAP to use if and when plotting the density map. If None, the argument isn't passed to
                imshow().
        """
        if max_examples is not None and max_examples < 1:
            raise ValueError("Max examples number must be an integer greater than 0. If you wish to set no limit, "
                             "please use None")

        Output.__init__(self)
        self.args = self._prepare_args(locals())
        self.max_examples = max_examples
        self.show_density_map = show_density_map
        self.density_map_cmap = density_map_cmap

    def output(self, images_and_density_maps):
        """
        Show examples of preprocessed data.

        Args:
            images_and_density_maps: Iterator of img+DM pairs.

        Returns:
            Iterator of unchanged img+DM pairs.
        """
        max_examples = self.max_examples if self.max_examples is not None else float("inf")
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map
            if cnt < max_examples:
                cols = 2 if self.show_density_map else 1
                _, axes = _plt.subplots(1, cols, figsize=(20, 4))
                axes[0].set_title(f"Image {str(cnt)}")
                axes[0].imshow(_cv2.cvtColor(image, _cv2.COLOR_BGR2RGB))
                if self.show_density_map:
                    axes[1].set_title(f"Density map {str(cnt)}")
                    if self.density_map_cmap is None:
                        axes[1].imshow(density_map)
                    else:
                        axes[1].imshow(density_map, cmap=self.density_map_cmap)
                _plt.show(block=False)
                _cv2.waitKey(0)
            yield image, density_map
            cnt += 1


class SaveImagesToFiles(Output):
    """
    Output that saves the images to image files, be it PNG, JPG or BMP.
    Each image is saved separately, and the name of the file is deduced from the image's index.
    """
    def __init__(self, dir_path, file_extension="jpg"):
        """
        Define an output that saves the images with the chosen file extension to a given directory. For the list of
        supported extensions, please check opencv-python documentation.

        Args:
            dir_path: Directory where all the images will be saved.
            file_extension: File extension and at the same time format of the saved image.
        """
        Output.__init__(self)
        self.args = self._prepare_args(locals())
        self.dir_path = dir_path
        self.file_extension = file_extension

    def output(self, images_and_density_maps):
        """
        Save the images in a given directory, making the directory if necessary. File names are constructed using
        image's index, starting from 0.

        Args:
            images_and_density_maps: Iterator of img+DM pairs.

        Returns:
            Iterator of unchanged img+DM pairs.
        """
        if not _os.path.exists(self.dir_path):
            _os.makedirs(self.dir_path)
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map
            path = _os.path.join(self.dir_path, f"IMG_{str(cnt)}.{self.file_extension}")
            _cv2.imwrite(path, image)
            yield image, density_map
            cnt += 1


class SaveImagesToBinaryFile(Output):
    """
    Output that saves all the images in form of a serialized list to a single binary file, allowing faster loading.
    """
    def __init__(self, file_path, keep_3_dimensions=True):
        """
        Defines the output operation.

        Args:
            file_path: Path where the file with all the images will be saved.
        :param keep_3_dimensions: When working on grayscale images, their numerical representations may be numpy arrays
            with shape[2] (indicating channels number) left out instead of being 1. This fixes their shape.
        """
        Output.__init__(self)
        self.args = self._prepare_args(locals())
        self.requires_full_dataset_in_memory = True
        self.file_path = file_path
        self.keep_3_dimensions = keep_3_dimensions

    def output(self, images_and_density_maps):
        """
        Save the images to a binary file.

        Args:
            images_and_density_maps: Iterator of img+DM pairs.

        Returns:
            Iterator of unchanged img+DM pairs.
        """
        dir_path = _os.path.dirname(self.file_path)
        if not _os.path.exists(dir_path):
            _os.makedirs(dir_path)
        images = []
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map
            if self.keep_3_dimensions and len(image.shape) != 3:
                images.append(image.copy().reshape(*image.shape, 1))
            else:
                images.append(image)
            yield image, density_map
        with open(self.file_path, 'wb') as f:
            _pickle.dump(images, f)


class SaveDensityMapsToCSVFiles(Output):
    """
    Output that saves density maps to CSV files, one density map per file.
    """
    def __init__(self, dir_path, downscaling=None):
        """
        Define output that saves to a given directory. Optionally, downscaling may be used when the density maps are
        expected to be smaller than their corresponding images (e.g. when the model has unbalanced pooling).

        Args:
            dir_path: Directory where the CSV files will be saved.
            downscaling: If not None, downscales the density maps by a given factor - for example, when using
                `downscaling` equal to 0.25, maps' widths and heights will be reduced to 1/4ths the original.
        """
        if downscaling is not None and not 0.0 < downscaling <= 1.0:
            raise ValueError("Downscaling factor must be between 0 (exclusive) and 1 (inclusive)")

        Output.__init__(self)
        self.args = self._prepare_args(locals())
        self.dir_path = dir_path
        self.downscaling = downscaling

    def output(self, images_and_density_maps):
        """
        Save the density maps to files.

        Args:
            images_and_density_maps: Iterator of img+DM pairs.

        Returns:
            Iterator of unchanged img+DM pairs.
        """
        if not _os.path.exists(self.dir_path):
            _os.makedirs(self.dir_path)
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map
            den_map_to_save = density_map
            if self.downscaling is not None:
                den_map_to_save = _cv2.resize(
                    den_map_to_save, None, fx=self.downscaling, fy=self.downscaling, interpolation=_cv2.INTER_LINEAR
                ) / self.downscaling ** 2
            path = _os.path.join(self.dir_path, f"GT_{str(cnt)}.csv")
            with open(path, 'w', newline='') as f:
                _csv.writer(f).writerows(den_map_to_save)
            yield image, density_map
            cnt += 1


class SaveDensityMapsToBinaryFile(Output):
    """
    Output that saves density maps in form of serialized lists to a single large binary file, allowing faster loading.
    """
    def __init__(self, file_path, downscaling=None, keep_3_dimensions=True):
        """
        Define output that saves to a given path. Optionally, downscaling may be used when the density maps are
        expected to be smaller than their corresponding images (e.g. when the model has unbalanced pooling).

        Args:
            file_path: Path where the file with all the density maps will be saved.
            downscaling: If not None, downscales the density maps by a given factor - for example, when using
                `downscaling` equal to 0.25, maps' widths and heights will be reduced to 1/4ths the original.
            keep_3_dimensions: The numerical representations of density maps may be numpy arrays with shape[2]
                (indicating channels number) left out instead of being 1. This fixes their shape.
        """
        if downscaling is not None and not 0.0 < downscaling <= 1.0:
            raise ValueError("Downscaling factor must be between 0 (exclusive) and 1 (inclusive)")

        Output.__init__(self)
        self.args = self._prepare_args(locals())
        self.requires_full_dataset_in_memory = True
        self.file_path = file_path
        self.downscaling = downscaling
        self.keep_3_dimensions = keep_3_dimensions

    def output(self, images_and_density_maps):
        """
        Save the density maps to a file.

        Args:
            images_and_density_maps: Iterator of img+DM pairs.

        Returns:
            Iterator of unchanged img+DM pairs.
        """
        dir_path = _os.path.dirname(self.file_path)
        if not _os.path.exists(dir_path):
            _os.makedirs(dir_path)
        den_maps = []
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map
            den_map_to_save = density_map
            if self.downscaling is not None:
                den_map_to_save = _cv2.resize(
                    den_map_to_save, None, self.downscaling, self.downscaling, interpolation=_cv2.INTER_LINEAR
                ) / self.downscaling ** 2
            if self.keep_3_dimensions and len(den_map_to_save.shape) != 3:
                den_map_to_save = den_map_to_save.copy().reshape(*den_map_to_save.shape, 1)
            den_maps.append(den_map_to_save)
            yield image, density_map
        with open(self.file_path, 'wb') as f:
            _pickle.dump(den_maps, f)
