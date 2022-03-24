import random as _random
import typing as _typing

import cv2 as _cv2
import numpy as _np

from .common import _IMG_TYPE, _DM_TYPE, _IMG_DM_PAIR_TYPE, _IMG_DM_ITER_TYPE
from .operations import Operation


class Transformation(Operation):
    """
    Abstract base class for any transformation that may occur on a set of image and density map pairs.
    Each transformation accepts an iterable (most commonly a generator) of image and density map pairs and is expected
    to return an iterable of the same cardinality. For custom transformations, please subclass this class or use
    LambdaTransformation.
    """
    def __init__(self, probability: float):
        """
        Create a new abstract transformation that is applied with specified probability.

        Args:
            probability: Float value between 0 and 1 (inclusive).
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1")

        Operation.__init__(self)
        self.args = self._prepare_args(locals())
        self.probability = probability

    def execute(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """ See `transform_all` """
        return self.transform_all(images_and_density_maps)

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Abstract method to be implemented in child classes. Most often, this and __init__ methods are the only ones
        that must be defined.

        Args:
            image: Image that will be transformed.
            density_map: Density map that will be transformed according to the image transformation.

        Returns:
            Transformed pair of image and density map.
        """
        raise NotImplementedError("transform method not implemented in the child class")

    def transform_all(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """
        Execute transformation with earlier specified probability on an iterable of img+DM pairs.

        Args:
            images_and_density_maps: Iterable of img+DM pairs to maybe be transformed.

        Returns:
            Iterable of maybe transformed img+DM pairs.
        """
        for img_and_den_map in images_and_density_maps:
            yield self.transform(*img_and_den_map) if _random.random() < self.probability else img_and_den_map


def _crop(image: _IMG_TYPE, density_map: _DM_TYPE, new_w: int, new_h: int, centered: bool) -> _IMG_DM_PAIR_TYPE:
    h, w = image.shape[:2]

    if centered:
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
    else:
        x0 = _random.randint(0, w - new_w)
        y0 = _random.randint(0, h - new_h)
    x1 = x0 + new_w
    y1 = y0 + new_h

    new_img = image[y0:y1, x0:x1]
    new_den_map = density_map[y0:y1, x0:x1]

    return new_img, new_den_map


class Crop(Transformation):
    """
    Cropping transformation that cuts out and returns a part of an image with specified size (either fixed one
    or a fraction of the original one). Density map is also reduced to keep it relevant to the image.
    """
    def __init__(self, width: _typing.Optional[int] = None, height: _typing.Optional[int] = None,
                 x_factor: _typing.Optional[float] = None, y_factor: _typing.Optional[float] = None,
                 centered: bool = False, probability: float = 1.0):
        """
        Define cropping with specified output size, applied with some probability. One may use a combination of
        fixed and relative size for separate image dimensions but fixed and relative size cannot be mixed for one
        dimension: one and only of them can be specified.

        Args:
            width: Fixed output width.
            height: Fixed output height.
            x_factor: Output width relative to the input width.
            y_factor: Output height relative to the input height.
            centered: Whether crops are taken from the center or at random positions.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if (width is not None and x_factor is not None) or (height is not None and y_factor is not None):
            raise ValueError("Cannot provide factor and fixed size at the same time")
        if (width is None and x_factor is None) or (height is None and y_factor is None):
            raise ValueError("Must provide factor or fixed size for both dimensions")
        if width is not None and width <= 0:
            raise ValueError("Width must be greater than 0 (and less than/equal original width)")
        if height is not None and height <= 0:
            raise ValueError("Height must be greater than 0 (and less than/equal original height)")
        if x_factor is not None and not 0.0 < x_factor <= 1.0:
            raise ValueError("Width factor must be between 0 (exclusive) and 1 (inclusive)")
        if y_factor is not None and not 0.0 < y_factor <= 1.0:
            raise ValueError("Height factor must be between 0 (exclusive) and 1 (inclusive)")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.width = width
        self.height = height
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.centered = centered

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Crop an image at a random position (or center) to specified size.

        Args:
            image: Image to be cropped.
            density_map: Density map to be cropped accordingly, with the same size as the image.

        Returns:
            Cropped pair of image and density map.
        """
        h, w = image.shape[:2]
        new_w = round(w * self.x_factor) if self.width is None else self.width
        new_h = round(h * self.y_factor) if self.height is None else self.height
        return _crop(image, density_map, new_w, new_h, self.centered)


class Scale(Transformation):
    """
    Scaling transformation that increases or decreases input size.
    """
    def __init__(self, width: _typing.Optional[int] = None, height: _typing.Optional[int] = None,
                 x_factor: _typing.Optional[float] = None, y_factor: _typing.Optional[float] = None,
                 img_interpolation: int = _cv2.INTER_CUBIC, dm_interpolation: int = _cv2.INTER_LINEAR,
                 probability: float = 1.0):
        """
        Create a scaling transformation that scales the image and the corresponding density map to a specified fixed or
        relative size with a given probability. One may use a combination of fixed and relative size for separate image
        dimensions but fixed and relative size cannot be mixed for one dimension - one and only of them can be
        specified. You may also specify custom interpolation for images and density maps.

        Args:
            width: Fixed output width.
            height: Fixed output height.
            x_factor: Output width relative to the input width.
            y_factor: Output height relative to the input height.
            img_interpolation: Interpolation that will be applied to images when scaling.
            dm_interpolation: Interpolation that will be applied to density maps when scaling.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if (width is not None and x_factor is not None) or (height is not None and y_factor is not None):
            raise ValueError("Cannot provide factor and fixed size at the same time")
        if (width is None and x_factor is None) or (height is None and y_factor is None):
            raise ValueError("Must provide factor or fixed size for both dimensions")
        if width is not None and width <= 0:
            raise ValueError("Width must be greater than 0")
        if height is not None and height <= 0:
            raise ValueError("Height must be greater than 0")
        if x_factor is not None and x_factor <= 0.0:
            raise ValueError("Width factor must be greater than 0")
        if y_factor is not None and y_factor <= 0.0:
            raise ValueError("Height factor must be greater than 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.width = width
        self.height = height
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.img_interpolation = img_interpolation
        self.dm_interpolation = dm_interpolation

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Scale an image and the corresponding density map to a specified size.

        Args:
            image: Image to be scaled.
            density_map: Density map to be scaled accordingly.

        Returns:
            Scaled pair of image and density map.
        """
        if self.width and self.height:
            h, w = image.shape[:2]
            scale_x = self.width / w
            scale_y = self.height / h

            new_img = _cv2.resize(
                image, (self.width, self.height), interpolation=self.img_interpolation
            )
            new_den_map = _cv2.resize(
                density_map, (self.width, self.height), interpolation=self.dm_interpolation
            ) / scale_x / scale_y
        else:
            new_img = _cv2.resize(
                image, None, fx=self.x_factor, fy=self.y_factor, interpolation=self.img_interpolation
            )
            new_den_map = _cv2.resize(
                density_map, None, fx=self.x_factor, fy=self.y_factor, interpolation=self.dm_interpolation
            ) / self.x_factor / self.y_factor
        return new_img, new_den_map


class Downscale(Transformation):
    """
    Downscales and then upscales an image, decreasing its quality.
    """
    def __init__(self, x_factor: float, y_factor: float, probability: float = 1.0):
        """
        Define downscaling in terms of how much the image will be downscaled before getting upscaled back to the
        original size. Note that some pixels may be lost due to integer rounding, leading to a slightly different size.

        Args:
            x_factor: Downscaling factor for width.
            y_factor: Downscaling factor for height.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if not 0.0 < x_factor <= 1.0:
            raise ValueError("Width factor must be between 0 (exclusive) and 1 (inclusive)")
        if not 0.0 < y_factor <= 1.0:
            raise ValueError("Height factor must be between 0 (exclusive) and 1 (inclusive)")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.downscaler = Scale(None, None, x_factor, y_factor, probability=1.0)
        self.upscaler = Scale(None, None, 1 / x_factor, 1 / y_factor, probability=1.0)

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Decrease an image quality. Density map stays the same.

        Args:
            image: Image to be downscaled and upscaled back to normal.
            density_map: Corresponding density map that won't be affected.

        Returns:
            Pair of transformed img+DM.
        """
        return self.upscaler.transform(*self.downscaler.transform(image, density_map))


class Rotate(Transformation):
    """
    Rotates the given image and density map. The rotation can be executed in two ways:

    - The size of the input doesn't change, discarding pixels outside the frame and filling missing pixels in the frame
        with black
    - The size of the input changes, adjusting the frame so that it can hold the whole image/density map and filling
        missing pixels in the frame with black
    """
    def __init__(self, angle: float, expand: bool = False, probability: float = 1.0):
        """
        Create a rotation at the center of image and density map by certain angle measured in degrees. Positive angle
        means counterclockwise rotation.

        Args:
            angle: Rotation angle in degrees, positive rotates counterclockwise, negative - clockwise.
            expand: Whether to adjust frame size for it to contain all the original pixels.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.angle = angle
        self.expand = expand

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Rotate the image according to specification.

        Args:
            image: Image to be rotated.
            density_map: Related density map that will be rotated accordingly.

        Returns:
            Pair of rotated image and rotated accordingly density map.
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        rot_mat = _cv2.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)

        if self.expand:
            cos, sin = _np.abs(rot_mat[0][:2])

            # calculate width and height that will allow the rotated image to be fully preserved
            new_w = int(w * cos + h * sin)
            new_h = int(w * sin + h * cos)

            # add translation to the matrix
            rot_mat[0, 2] += (new_w // 2) - center_x
            rot_mat[1, 2] += (new_h // 2) - center_y
        else:
            new_w, new_h = w, h

        return _cv2.warpAffine(image, rot_mat, (new_w, new_h)), _cv2.warpAffine(density_map, rot_mat, (new_w, new_h))


def _prepare_standard_aspect_ratios(std_ratios: _typing.Collection[float]) \
        -> _typing.Tuple[_np.ndarray, _np.ndarray]:
    """
    Find boundaries between consequent allowed aspect ratios to make finding the most appropriate ratios easier.
    Boundaries are such aspect ratio values that are uniformly placed between two nearest allowed aspect ratios.

    Args:
        std_ratios: Aspect ratios that are allowed for the output images and density maps.

    Returns:
        Sorted allowed aspect ratios and calculated boundaries between them.
    """
    ratios = _np.sort(_np.array(std_ratios))
    boundaries = _np.array([(prev + curr) / 2 for (prev, curr) in zip(ratios[:-1], ratios[1:])])
    return ratios, boundaries


def _find_the_most_similar_ratio(ratio_to_improve: float, std_ratios: _np.ndarray, std_boundaries: _np.ndarray) \
        -> float:
    """
    Find an allowed aspect ratio that is the most similar to the one provided.

    Args:
        ratio_to_improve: Aspect ratio that maybe can be improved.
        std_ratios: Allowed aspect ratios.
        std_boundaries: Uniformly distributed boundaries between allowed aspect ratios.

    Returns:
        Allowed aspect ratio that is the most similar to the provided one.
    """
    last_matching_boundary_idx = _np.nonzero(ratio_to_improve < std_boundaries)[0]
    if last_matching_boundary_idx.shape[0] > 0:
        chosen_std_ratio_idx = last_matching_boundary_idx[0]
    else:
        chosen_std_ratio_idx = std_ratios.shape[0] - 1
    return std_ratios[chosen_std_ratio_idx]


class StandardizeSize(Transformation):
    """
    Standardizes image and density map sizes in order to reduce variance in size and allow bigger batches.
    This transformation takes a list of allowed aspect ratios for the images and the base size corresponding to the
    length of the longer side of an image and scales each image (and the relevant density map) to the size best fitting
    the original size.
    """
    def __init__(self, std_aspect_ratios: _typing.Collection[float], std_base_size: int, method: str = "scale"):
        """
        Create a size standardization transformation.

        Args:
            std_aspect_ratios: List of aspect ratios defined as float values that are allowed to exist in the output
                images. In case of ratios suited for portrait/vertical mode, inversion of the same aspect ratio from
                horizontal mode is used, e.g. ratio of 4:3 vertical image is to be seen as 3:4 (or 0.75, to be exact).
            std_base_size: Desired length of the longer side of the output images and density maps.
            method: How to standardize sizes. You can either scale the images, making them a bit distorted but keeping
                all the information, or randomly crop them, losing some information in exchange for quality.
        """
        if len(std_aspect_ratios) == 0:
            raise ValueError("At least 1 allowed aspect ratio must be entered")
        for ratio in std_aspect_ratios:
            if ratio <= 0:
                raise ValueError("Each allowed aspect ratio must be a fraction of width to height, greater than 0")
        if len(set(std_aspect_ratios)) < len(std_aspect_ratios):
            raise ValueError("There must not be any duplicate allowed aspect ratios")
        if std_base_size <= 0:
            raise ValueError("Base size must be greater than 0")
        if method not in ["scale", "crop"]:
            raise ValueError("You can only use scaling (scale) or cropping (crop)")

        Transformation.__init__(self, 1.0)
        self.args = self._prepare_args(locals())
        self.std_ratios, self.std_bounds = _prepare_standard_aspect_ratios(std_aspect_ratios)
        self.std_base_size = std_base_size
        self.method = method

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Scale the image and its density map to such a size that its aspect ratio is in the allowed ones.

        Args:
            image: Image with any aspect ratio.
            density_map: Relevant density map of the same size as image.

        Returns:
            Scaled pair of image and its density map.
        """
        h, w = image.shape[:2]

        chosen_std_ratio = _find_the_most_similar_ratio(w / h, self.std_ratios, self.std_bounds)

        if chosen_std_ratio >= 1.0:  # landscape or 1:1
            new_w = self.std_base_size
            new_h = int(new_w / chosen_std_ratio)
        else:  # portrait
            new_h = self.std_base_size
            new_w = int(new_h * chosen_std_ratio)

        # if the size cannot be standardized further, return
        if (w, h) == (new_w, new_h):
            return image, density_map

        if self.method == 'scale':
            scale_x = new_w / w
            scale_y = new_h / h

            new_img = _cv2.resize(image, (new_w, new_h), interpolation=_cv2.INTER_CUBIC)
            new_den_map = _cv2.resize(density_map, (new_w, new_h), interpolation=_cv2.INTER_LINEAR) / scale_x / scale_y
        else:
            new_img, new_den_map = _crop(image, density_map, new_w, new_h, False)

        return new_img, new_den_map


class AutoStandardizeSize(Transformation):
    """
    A transformation that applies StandardizeSize in an automated way. Instead of relying on manually specified
    accepted aspect ratios, most common ratios are checked if there are enough instances that have similar
    characteristic to them, and the present ones are used for the transformation. As for the base size to be used, the
    minimum found greater dimension (width or height) of an image is selected. It may not be the optimal choice but it
    should be rather safe, assuming that there aren't any drastic outliers in the dataset.
    """
    MOST_COMMON_RATIOS = _np.array([1/1, 3/2, 2/3, 4/3, 3/4, 5/4, 4/5, 16/9, 9/16])
    TRIED_STD_RATIOS, TRIED_STD_BOUNDS = _prepare_standard_aspect_ratios(MOST_COMMON_RATIOS)

    def __init__(self, min_instances_threshold: int, method: str = "scale"):
        """
        Set up the automated transformation.

        Args:
            min_instances_threshold: Minimum number of instances of an aspect ratio (not necessarily exactly that one
                but being closest to that one) from the tried ratios list for that ratio to be considered relevant to
                the dataset used.
            method: See the same argument in `StandardizeSize`.
        """
        if min_instances_threshold <= 0:
            raise ValueError("Threshold must be greater than 0")
        if method not in ["scale", "crop"]:
            raise ValueError("You can only use scaling (scale) or cropping (crop)")

        Transformation.__init__(self, 1.0)
        self.requires_full_dataset_in_memory = True
        self.args = self._prepare_args(locals())
        self.min_instances_threshold = min_instances_threshold
        self.method = method

    def _select_relevant_ratios_and_size(self, images_and_density_maps: _typing.Collection[_IMG_DM_PAIR_TYPE]) \
            -> _typing.Tuple[_np.ndarray, int]:
        """
        Find out which aspect ratios are relevant enough for the dataset and which size is the most appropriate (the
        minimum size in the bigger dimension (width or height) is used).

        Args:
            images_and_density_maps: Iterable of img+DM pairs that was saved so that it can be traversed more than once.

        Returns:
             Array of selected aspect ratios and the selected base size.
        """
        occur_counters = _np.zeros(len(self.MOST_COMMON_RATIOS))
        min_size = float('inf')
        for img, dm in images_and_density_maps:
            h, w = img.shape[:2]
            min_size = min(min_size, max(h, w))
            chosen_std_ratio = _find_the_most_similar_ratio(w / h, self.TRIED_STD_RATIOS, self.TRIED_STD_BOUNDS)
            occur_counters[_np.where(self.MOST_COMMON_RATIOS == chosen_std_ratio)] += 1

        rel_ratio_ind = _np.where(occur_counters >= self.min_instances_threshold)
        return self.MOST_COMMON_RATIOS[rel_ratio_ind], int(min_size)

    def transform_all(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """
        Apply the standardization to the dataset. First, the whole dataset needs to be saved so that it can be
        traversed more than once. Then, the follow-up setup is conducted. Finally, the most appropriate size
        standardization is performed.

        Params:
            images_and_density_maps: Iterable of img+DM pairs.

        Returns:
            Iterable of automatically standardized img+DM pairs.
        """
        imgs_and_dms_list = list(images_and_density_maps)
        if len(imgs_and_dms_list) < self.min_instances_threshold:
            raise ValueError("Minimum required ratio instances must not be greater than number of samples")
        std_ratios, min_base_size = self._select_relevant_ratios_and_size(imgs_and_dms_list)
        substandardizer = StandardizeSize(std_ratios, min_base_size, self.method)
        return substandardizer.transform_all(imgs_and_dms_list)


class OmitDownscalingPixels(Transformation):
    """
    Removes pixels that were lost due to downscaling (either by using a CCA transformation or by doing Pooling) by
    performing a centered crop.
    """
    def __init__(self, x_factor: _typing.Optional[float] = None, y_factor: _typing.Optional[float] = None):
        """
        Create a centered cropping transformation that omits pixels lost due to downscaling. For example, given an input
        image of 401x404 size, when it is downscaled by a factor of 4 in both dimensions (e.g. by doing
        2 2x2 MaxPoolings and then interpolating using a factor of 4), the final size of the image is 400x400. This
        transformation automatically resizes images and density maps to match the expected cut shape.

        Args:
            x_factor: By what factor was the input downscaled horizontally. Leave it None to omit horizontal crop.
            y_factor: By what factor was the input downscaled vertically. Leave it None to omit vertical crop.
        """
        if x_factor is not None and x_factor < 1:
            raise ValueError("Width factor must be an integer greater than/equal 1")
        if y_factor is not None and y_factor < 1:
            raise ValueError("Height factor must be an integer greater than/equal 1")

        Transformation.__init__(self, 1.0)
        self.args = self._prepare_args(locals())
        self.x_factor = x_factor
        self.y_factor = y_factor

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Transform the given image and density map.

        Args:
            image: Image that was downscaled.
            density_map: Density map that was downscaled.

        Returns:
            Pair of transformed image and density map.
        """
        h, w = image.shape[:2]
        new_w = w - (w % self.x_factor) if self.x_factor is not None else w
        new_h = h - (h % self.y_factor) if self.y_factor is not None else h
        return _crop(image, density_map, new_w, new_h, True)


class Normalize(Transformation):
    """
    Normalizes image pixel values using one of normalization methods.
    """
    def __init__(self, method: str, by_channel: bool = False, means: _typing.Optional[_np.ndarray] = None,
                 stds: _typing.Optional[_np.ndarray] = None):
        """
        Create a transformation that normalizes image pixel values using one of the following methods:

        - `range_0_to_1` - values are scaled to be in fixed <0; 1> range mapped to the original <0; 255> range
        - `range_-1_to_1` - values are scaled to be in fixed <-1; 1> range mapped to the original <0; 255> range
        - 'samplewise_z-score' - combination of samplewise_centering and samplewise_std_normalization
        - `samplewise_centering` - values are translated to make their mean (in respect to a single image) equal to 0
        - `samplewise_std_normalization` - values are scaled to make their sample standard deviation (in respect to
            a single image) equal to 1
        - 'featurewise_z-score' - combination of featurewise_centering and featurewise_std_normalization
        - `featurewise_centering` - values are translated to make their mean (in respect to all images) equal to 0
        - `featurewise_std_normalization` - values are scaled to make their sample standard deviation (in respect to all
            images) equal to 1

        Args:
            method: Method that will be used for normalization, one of the above.
            by_channel: If true, methods using mean or standard deviation will calculate that metric over each channel
                separately and the normalization will also occur by each channel. Else, the values will be normalized
                by all channels at once.
            means: If not None, the operation uses those mean values instead of computing them on its own. Shape varies
                depending on the normalization. Currently, only featurewise normalization uses it.
            stds: If not None, the operation uses those standard deviation values instead of computing them on its own.
                Shape varies depending on the normalization. Currently, only featurewise normalization uses it.
        """
        if method not in ["range_0_to_1", "range_-1_to_1", "samplewise_z-score", "samplewise_centering",
                          "samplewise_std_normalization", "featurewise_z-score", "featurewise_centering",
                          "featurewise_std_normalization"]:
            raise ValueError(f"Wrong method of normalization selected: {method}")
        if method.startswith("range") and (means is not None or stds is not None):
            raise ValueError("Fixed range normalization doesn't require computed mean/std values.")
        if by_channel and ((means is not None and len(means) != 3) or (stds is not None and len(stds) != 3)):
            raise ValueError("When applying by-channel normalization with specified means or stds, they must consist of"
                             " exactly 3 values, one for each channel.")
        if not by_channel and ((means is not None and len(means) != 1) or (stds is not None and len(stds) != 1)):
            raise ValueError("When applying global normalization with specified means or stds, they must consist of"
                             " exactly 1 value.")

        Transformation.__init__(self, 1.0)
        self.args = self._prepare_args(locals())
        self.method = method
        self.by_channel = by_channel
        self.means = means
        self.stds = stds
        if (method == "featurewise_z-score" and (means is None or stds is None)) or \
                (method == "featurewise_centering" and means is None) or \
                (method == "featurewise_std_normalization" and stds is None):
            self.requires_full_dataset_in_memory = True
            # discard potential means or stds
            self.means = None
            self.stds = None

    @staticmethod
    def _apply_nonfw_normalization(method, image, mean_std_axes):
        """ Apply a specific fixed range or samplewise normalization on an image"""
        if method == "range_-1_to_1":
            return (image - 127.5) / 255.0
        elif method == "range_0_to_1":
            return image / 255.0
        # TODO: provided means and stds could be put to use
        elif method == "samplewise_centering":
            return image - _np.resize(_np.mean(image, mean_std_axes), [*image.shape])
        else:  # method == "samplewise_std_normalization"
            return image / _np.resize(_np.std(image, mean_std_axes), [*image.shape])

    @staticmethod
    def _apply_fw_normalization_on_all(method, all_images, mean_std_axes):
        """ Apply a specific featurewise normalization on all images at once """
        if method == "featurewise_centering":
            return all_images - _np.resize(_np.mean(all_images, mean_std_axes), [*all_images.shape])
        else:  # method == "featurewise_std_normalization"
            return all_images / _np.resize(_np.std(all_images, mean_std_axes), [*all_images.shape])

    @staticmethod
    def _apply_fw_normalization(method, image, means, stds):
        """ Apply a specific featurewise normalization on an image """
        if method == "featurewise_centering":
            return image - _np.resize(means, [*image.shape])
        else:  # method == "featurewise_std_normalization"
            return image / _np.resize(stds, [*image.shape])

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Transform the given image, return with the untouched density map.

        Args:
            image: Image whose pixel values will be normalized.
            density_map: Density map that won't be affected.

        Returns:
            Pair of transformed image and corresponding density map.
        """
        mean_std_axes = (0, 1) if self.by_channel else None
        if self.by_channel and len(image.shape) != 3:
            image.shape = (*image.shape, 1)

        if self.method == "samplewise_z-score":
            image = self._apply_nonfw_normalization("samplewise_centering", image, mean_std_axes)
            image = self._apply_nonfw_normalization("samplewise_std_normalization", image, mean_std_axes)
            return image, density_map
        else:
            image = self._apply_nonfw_normalization(self.method, image, mean_std_axes)
            return image, density_map

    def transform_all(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """
        Overriding method looping over all img+DM pairs from an iterable.
        Behaves the same like the base class implementation when using a method that doesn't require full dataset to be
        in memory, or first collects all samples and then transforms them otherwise.

        Args:
            images_and_density_maps: Iterable of pairs of images and corresponding density maps.

        Returns:
            Iterable of transformed img+DM pairs.
        """
        if self.method.startswith("range") or self.method.startswith("samplewise"):
            for image, density_map in Transformation.transform_all(self, images_and_density_maps):
                yield image, density_map
            return

        if self.requires_full_dataset_in_memory:
            all_images, all_density_maps = zip(*list(images_and_density_maps))
            all_images, all_density_maps = _np.array(all_images), _np.array(all_density_maps)
            mean_std_axes = (0, 1, 2) if self.by_channel else None
            if self.by_channel and len(all_images.shape) != 4:
                all_images.shape = (*all_images.shape, 1)

            if self.method == "featurewise_z-score":
                all_images = self._apply_fw_normalization_on_all("featurewise_centering", all_images, mean_std_axes)
                all_images = self._apply_fw_normalization_on_all(
                    "featurewise_std_normalization", all_images, mean_std_axes)
                for t in zip(all_images, all_density_maps):
                    yield t
            else:
                all_images = self._apply_fw_normalization_on_all(self.method, all_images, mean_std_axes)
                for t in zip(all_images, all_density_maps):
                    yield t
        else:
            if self.method == "featurewise_z-score":
                for image, density_map in images_and_density_maps:
                    image = self._apply_fw_normalization("featurewise_centering", image, self.means, self.stds)
                    image = self._apply_fw_normalization("featurewise_std_normalization", image, self.means, self.stds)
                    yield image, density_map
            else:
                for image, density_map in images_and_density_maps:
                    image = self._apply_fw_normalization(self.method, image, self.means, self.stds)
                    yield image, density_map


class NormalizeDensityMap(Transformation):
    """
    Normalizes a density map by multiplying its values by a specified parameter. May help in training speed. Based on
    https://arxiv.org/pdf/1907.02724.pdf
    """
    def __init__(self, multiplier: float):
        """
        Create a label/density map normalization operation.

        Args:
            multiplier: The values in the density map will be multiplied by that number. Make sure to divide
                the predicted density maps by the same number when calculating the count.
        """
        if multiplier <= 0.0:
            raise ValueError("Multiplier must be greater than 0")

        Transformation.__init__(self, 1.0)
        self.args = self._prepare_args(locals())
        self.multiplier = multiplier

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Multiply the values in the density map.

        Args:
            image: Image that stays the same.
            density_map: Density map that is multiplied.

        Returns:
            Image and transformed density map.
        """
        return image, density_map * self.multiplier


class FlipLR(Transformation):
    """
    Horizontal / left-right random flipping transformation.
    """
    def __init__(self, probability: float = 0.5):
        """
        Create LR flipping transformation that flips with the given probability. In most cases, should stay at 0.5
        (equal chances).

        Args:
            probability: Probability of flipping the image and its density map. Between 0 and 1 (inclusive). In most
                cases, should stay at 0.5.
        """
        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Horizontally flip the image and its density map.

        Args:
            image: Input image.
            density_map: Corresponding density map.

        Returns:
            Flipped image and density map.
        """
        return _np.fliplr(image), _np.fliplr(density_map)


class ToGrayscale(Transformation):
    """
    Transformation that converts images to grayscale (reduces channels to 1).
    """
    def __init__(self, probability: float = 1.0):
        """
        Create the transformation applied with the given probability.

        Args:
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Transform the image to grayscale.

        Args:
            image: BGR image. Note that a loader can load an originally grayscale image to a BGR image. In such cases,
                the transformation doesn't provide any real effect.
            density_map: Density map that won't be affected.

        Returns:
            Pair of grayscale image and its density map.
        """
        return _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY), density_map


class LambdaTransformation(Transformation):
    """
    One of the two ways to create a custom transformation, the other one being subclassing `Transformation`.
    This class works by applying a transformation specified as a lambda, over all samples. Alternatively, one use a
    custom loop, also specified as a lambda.
    """
    TRANSF_TYPE = _typing.Callable[[_IMG_TYPE, _DM_TYPE], _IMG_DM_PAIR_TYPE]
    TRANSF_LOOP_TYPE = _typing.Callable[[_IMG_DM_ITER_TYPE, TRANSF_TYPE], _IMG_DM_ITER_TYPE]

    def __init__(self, probability: float, transformation: TRANSF_TYPE,
                 loop: _typing.Optional[TRANSF_LOOP_TYPE] = None):
        """
        Create a custom transformation that applies a given transformation lambda over all samples. Alternatively, a
        custom loop lambda can be given, e.g. to collect all samples before executing a transformation on them.

        Args:
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
            transformation: Lambda that takes img+DM from one pair and returns the transformed pair.
            loop: Lambda that takes an iterable of img+DM pairs and transformation lambda and returns an iterable of
                transformed img+DM pairs. If None, standard loop is used.
        """
        Transformation.__init__(self, probability)
        self.args = None  # can't save lambda definitions to a human-readable format
        self.transformation = transformation
        self.loop = loop

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """ Run the given lambda on a img+DM pair """
        return self.transformation(image, density_map)

    def transform_all(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """ If a custom loop is defined, use it to loop over all samples from the iterable. Otherwise, use the standard
        loop. """
        if self.loop is None:
            return Transformation.transform_all(self, images_and_density_maps)
        return self.loop(images_and_density_maps, self.transformation)


def _get_random_area(img_w: int, img_h: int, area_w: int, area_h: int, allow_out_of_bounds: bool) \
        -> _typing.Tuple[int, int, int, int]:
    """
    Get coordinates of a randomly placed rectangular area. Note that actual area size may differ from (area_h, area_w)
    if out of bounds selection is enabled and an area on the border is chosen.

    Args:
        img_w: Width of the image to be sampled from.
        img_h: Height of the image to be sampled from.
        area_w: Width of area to be obtained. In case of out of bounds selection, it's only the max width.
        area_h: Height of area to be obtained. In case of out of bounds selection, it's only the max height.
        allow_out_of_bounds: If true, area is free to be placed partially outside the image.

    Returns:
        Tuple with start X, start Y, end X, end Y.
    """
    if allow_out_of_bounds:
        min_area_x, min_area_y, max_area_x, max_area_y = -area_w, -area_h, img_w - 1, img_h - 1
        area_x, area_y = _random.randint(min_area_x, max_area_x), _random.randint(min_area_y, max_area_y)
        area_x1, area_y1 = max(0, area_x), max(0, area_y)
        area_x2, area_y2 = min(img_w, area_x + area_w), min(img_h, area_y + area_h)
    else:
        min_area_x, min_area_y, max_area_x, max_area_y = 0, 0, img_w - area_w, img_h - area_h
        area_x, area_y = _random.randint(min_area_x, max_area_x), _random.randint(min_area_y, max_area_y)
        area_x1, area_y1 = area_x, area_y
        area_x2, area_y2 = area_x + area_w, area_y + area_h
    return area_x1, area_y1, area_x2, area_y2


class Cutout(Transformation):
    """
    Experimental method based on this paper: https://arxiv.org/abs/1708.04552
    Selects random rectangular regions in an image and zeroes them out. According to the paper, the box should be
    allowed to encompass some area out of bounds for good performance, although one may also specify a lower probability
    so that a mix of images with more and less area removed can be found in the dataset. Shape of the cut out box
    seemed not to be that important so its definition is simplified.
    """
    def __init__(self, size: _typing.Optional[int], factor: _typing.Optional[float] = None, cuts_num: int = 1,
                 allow_out_of_bounds: bool = True, probability: float = 1.0):
        """
        Create a transformation that zeroes out random rectangular regions of given size, specified number of times.
        One may specify one and only one size, absolute or relative.

        Args:
            size: Fixed, absolute width and height of cutout regions. Always results in generated squares.
            factor: Relative width and height of cutout regions. Depending on the image's aspect ratio, it may produce
                a square or a rectangle.
            cuts_num: Number of times this operation is performed. Choice of position is random, so the cutouts may be
                overlays over each other.
            allow_out_of_bounds: Whether to allow randomizing such positions that produce regions sticking out of
                the frame, zeroing out less pixels than expected.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if size is not None and factor is not None:
            raise ValueError("Cannot provide factor and fixed size at the same time")
        if size is None and factor is None:
            raise ValueError("Must provide factor or fixed size")
        if size is not None and size <= 0:
            raise ValueError("Size must be greater than 0 (and less than/equal smaller original image dimension")
        if factor is not None and not 0.0 < factor <= 1.0:
            raise ValueError("Factor must be between 0 (exclusive) and 1 (inclusive)")
        if cuts_num < 1:
            raise ValueError("Number of cuts must be an integer greater than 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.size = size
        self.factor = factor
        self.cuts_num = cuts_num
        self.allow_out_of_bounds = allow_out_of_bounds

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> _IMG_DM_PAIR_TYPE:
        """
        Cut out random regions based on the settings.

        Args:
            image: Image to be cut.
            density_map: Related density map that will be cut accordingly.

        Returns:
            Cut image and density map.
        """
        new_img, new_den_map = image.copy(), density_map.copy()
        h, w = image.shape[:2]
        area_h, area_w = (self.size, self.size) if self.factor is None else (int(h * self.factor), int(w * self.factor))
        for _ in range(self.cuts_num):
            area_x1, area_y1, area_x2, area_y2 = _get_random_area(w, h, area_w, area_h, self.allow_out_of_bounds)
            new_img[area_y1:area_y2, area_x1:area_x2] = 0
            new_den_map[area_y1:area_y2, area_x1:area_x2] = 0
        return new_img, new_den_map


class Copyout(Transformation):
    """
    Experimental method based on this paper: https://arxiv.org/abs/1909.00390
    Copies square areas of specified size from random position from random image to a random position in currently
    transformed image. When out of bounds copying is allowed, the area may become a rectangle when placed on an edge.
    However, out of bounds selection doesn't apply to the destination image.
    """
    def __init__(self, size: int, allow_out_of_bounds: bool = True, probability: float = 1.0):
        """
        Create copyout transformation.

        Args:
            size: Size of the extent that will be copied from one image to another (along with the density data).
            allow_out_of_bounds: Whether to allow selection of area partially outside the source image frame.
                Destination area selection is unaffected.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if size <= 0:
            raise ValueError("Size must be an integer greater than 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.requires_full_dataset_in_memory = True
        self.size = size
        self.allow_out_of_bounds = allow_out_of_bounds

    def transform(self, image: _IMG_TYPE, density_map: _DM_TYPE) -> None:
        """ Transformation of a single image+density map pair without loading other pairs is not supported. """
        raise NotImplementedError("Only transforming the whole dataset at once is currently supported for Copyout")

    def transform_all(self, images_and_density_maps: _IMG_DM_ITER_TYPE) -> _IMG_DM_ITER_TYPE:
        """
        Create a generator that for each image and density map pair generates a transformed pair.

        Args:
            images_and_density_maps: Iterable of tuples of image and density map. Both are affected by the operation.

        Returns:
            Generator of tuples with transformed image and density map.
        """
        imgs_dms = list(images_and_density_maps)
        images_num = len(imgs_dms)
        for img, dm in imgs_dms:
            src_index = _np.random.randint(0, images_num)
            src_img, src_dm = imgs_dms[src_index]

            src_h, src_w = src_img.shape[:2]
            dest_h, dest_w = img.shape[:2]
            area_h, area_w = self.size, self.size

            src_area_x1, src_area_y1, src_area_x2, src_area_y2 = \
                _get_random_area(src_w, src_h, area_w, area_h, self.allow_out_of_bounds)

            if self.allow_out_of_bounds:
                # shape of the area may have changed
                area_w, area_h = src_area_x2 - src_area_x1, src_area_y2 - src_area_y1

            dest_area_x1, dest_area_y1, dest_area_x2, dest_area_y2 = \
                _get_random_area(dest_w, dest_h, area_w, area_h, False)

            new_img, new_dm = img.copy(), dm.copy()
            new_img[dest_area_y1:dest_area_y2, dest_area_x1:dest_area_x2] = \
                src_img[src_area_y1:src_area_y2, src_area_x1:src_area_x2]
            new_dm[dest_area_y1:dest_area_y2, dest_area_x1:dest_area_x2] = \
                src_dm[src_area_y1:src_area_y2, src_area_x1:src_area_x2]

            yield new_img, new_dm


class Shearing(Transformation):
    """
    Shears the given image and density map along X and Y axis.

    """
    def __init__(self, shearing_x, shearing_y, probability=1.0):
        """
        Define shearing parameters.

        Args:
            shearing_x: Shearing factor for X axis. Value of 0 corresponds to no shearing, value 0.5 to
                moderate shearing.
            shearing_y: Shearing factor for Y axis. Value of 0 corresponds to no shearing, value 0.5 to
                moderate shearing.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """

        if shearing_x < 0:
            raise ValueError("Shearing_x must be >= 0")
        if shearing_y < 0:
            raise ValueError("Shearing_y must be >= 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.shearing_x = shearing_x
        self.shearing_y = shearing_y

    def transform(self, image, density_map):
        """
        Shear the image according to the specification.

        Args:
            image: Image to be sheared.
            density_map: Related density map that will be sheared accordingly.

        Returns:
            A pair of sheared image and density map.
        """
        h, w = image.shape[:2]

        m = _np.float32([[1, self.shearing_x, 0],
                        [self.shearing_y, 1, 0],
                        [0, 0, 1]])

        # calculate width and height that will allow the sheared image to be fully preserved
        new_w = int(w + _np.abs(self.shearing_x) * h)
        new_h = int(h + _np.abs(self.shearing_y) * w)

        return _cv2.warpPerspective(image, m, (new_w, new_h)), _cv2.warpPerspective(density_map, m, (new_w, new_h))


class Blur(Transformation):
    """
    Blurs the given image with normalized box filter (averaging). Corresponding density map remains the same.

    """
    def __init__(self, ksize, probability=1.0):
        """
        Define blur transformation parameters.

        Args:
            ksize: Size of kernel used in averaging.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if ksize <= 0:
            raise ValueError("Kernel size must be an integer greater than 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.ksize = ksize

    def transform(self, image, density_map):
        """
        Blur the image according to the specification.

        Args:
            image: Image to be blurred.
            density_map: Related density map.

        Returns:
            A pair of blurred image and corresponding density map.
        """

        return _cv2.blur(image, (self.ksize, self.ksize)), density_map


class BlurCutout(Transformation):
    """
    Experimental method inspired by paper: https://arxiv.org/abs/1708.04552
    Selects random circular regions and blurs them (normalized box filtering). Corresponding density map
    remains unchanged.

    """
    def __init__(self, ksize, blurs_num, size, probability=1.0):
        """
        Create a transformation that blurs random circular regions (normalized box filtering) of given size,
        specified number of times. One may specify one and only one size, relative to the image size.

        Args:
            ksize: Size of kernel used in averaging.
            blurs_num: Number of times this operation is performed. Choice of position is random, so the blurs may
                overlay over each other.
            size: Size of single circular region to be blurred. Circle radius is calculated as average of
                height and width of the image multiplied by size parameter (fraction).
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """

        if ksize <= 0:
            raise ValueError("Kernel size must be an integer greater than 0")
        if blurs_num <= 0:
            raise ValueError("Number of blurs must be an integer greater than 0")
        if not 0.0 < size <= 1.0:
            raise ValueError("Size must be between 0 (exclusive) and 1 (inclusive)")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.ksize = ksize
        self.blurs_num = blurs_num
        self.size = size

    def transform(self, image, density_map):
        """
        Blurs out random regions according to the specification.

        Args:
            image: Image to be partially blurred out.
            density_map: Related density map.

        Returns:
            A pair of partially blurred image and corresponding density map.
        """
        h, w = image.shape[:2]
        image_blurred = _cv2.blur(image, (self.ksize, self.ksize))

        mask = _np.zeros((h, w, 3), dtype=_np.uint8)
        for _ in range(self.blurs_num):
            x_cord = _np.random.uniform(0, 1)
            y_cord = _np.random.uniform(0, 1)
            size = int(((h+w)/2)*self.size)
            mask = _cv2.circle(mask, (int(w*x_cord), int(h*y_cord)), size, (255, 255, 255), -1)

        return _np.where(mask == (255, 255, 255), image_blurred, image), density_map


class DistortPerspective(Transformation):
    """
    Experimental method. Distorts perspective of the given image and applies corresponding transformation to the density
    map. Perspective is distorted randomly based on 4 presets - extension of the top, bottom, left or right part of the
    picture. Strength od perspective distortion in specified by the user.
    """
    def __init__(self, strength, probability=1.0):
        """
        Define strength of perspective distortion.

        Args:
            strength: Strength of perspective distortion. Specified as percentage of height or width which will
                be omitted during the perspective extension. between 0 (inclusive) and 1 (exclusive).
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """

        if not 0.0 <= strength < 1:
            raise ValueError("Distortion strength must be between 0 (inclusive) and 1 (exclusive)")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.strength = strength

    def transform(self, image, density_map):
        """
        Distort perspective of the given image according to the specification.

        Args:
            image: Image to be distorted.
            density_map: Related density map that will be transformed accordingly.

        Returns:
            A pair of distorted image and density map.
        """
        h, w = image.shape[:2]

        modes = ['top', 'bottom', 'left', 'right']
        mode = _np.random.choice(modes)
        if mode == 'top':
            pts1 = _np.float32([[0+w*self.strength/2, 0], [w-w*self.strength/2, 0],
                                [0, h], [w, h]])
        elif mode == 'bottom':
            pts1 = _np.float32([[0, 0], [w, 0],
                                [0+w*self.strength/2, h], [w-w*self.strength/2, h]])
        elif mode == 'left':
            pts1 = _np.float32([[0, 0+h*self.strength/2], [w, 0],
                                [0, h-h*self.strength/2], [w, h]])
        else:
            pts1 = _np.float32([[0, 0], [w, 0+h*self.strength/2],
                                [0, h], [w, h-h*self.strength/2]])

        pts2 = _np.float32([[0, 0], [w, 0],
                            [0, h], [w, h]])

        matrix = _cv2.getPerspectiveTransform(pts1, pts2)

        return _cv2.warpPerspective(image, matrix, (w, h)), _cv2.warpPerspective(density_map, matrix, (w, h))


class ChangeSaturation(Transformation):
    """
    Transformation that changes saturation of given image.

    """
    def __init__(self, factor, probability=1.0):
        """
        Define saturation change transformation parameters.

        Args:
            factor: Strength of saturation change. Values smaller than 1 corresponds to decrease in saturation,
                values greater than 1 corresponds to increase in saturation.
            probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """

        if factor < 0:
            raise ValueError("Factor must be greater than/equal 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.factor = factor

    def transform(self, image, density_map):
        """
        Change saturation of the image according to specification.

        Args:
            image: Image to be transformed.
            density_map: Related density map that will be sheared accordingly.

        Returns:
            A pair of image with changed saturation and density map.
        """
        image = _cv2.cvtColor(image, _cv2.COLOR_BGR2HSV)
        image[..., 1] = image[..., 1] * self.factor
        image = _cv2.cvtColor(image, _cv2.COLOR_HSV2BGR)

        return image, density_map


class Mixup(Transformation):
    """
    Experimental method based on: https://arxiv.org/abs/1710.09412
    In this technique 2 randomly selected input images x_i, x_j and corresponding to them density maps y_i, y_j
    are mixed together to create new training sample lambda * x_i + (1 - lambda) * x_j with density map
    lambda * y_i + (1 - lambda) * y_j, where lambda is sampled from Beta distribution.

    """
    def __init__(self, alpha, probability=1):
        """
        Create mixup transformation.

        Args:
            alpha: Parameter of Beta distribution used in the mixup.
        """
        if alpha <= 0:
            raise ValueError("Alpha must be greater than 0")

        Transformation.__init__(self, probability)
        self.args = self._prepare_args(locals())
        self.requires_full_dataset_in_memory = True
        self.alpha = alpha

    def transform(self, image, density_map):
        """ Transformation of a single image+density map pair without loading other pairs is not supported. """
        raise NotImplementedError("Only transforming the whole dataset at once is currently supported for Copyout")

    def transform_all(self, images_and_density_maps):
        """
        Create a generator that for each image and density map pair performs mixup procedure with randomly selected
        pair from the dataset.

        Args:
            images_and_density_maps: Iterable of tuples of image and density map. Both are affected by the operation.

        Returns:
            Generator of tuples with transformed image and density map.
        """
        imgs_dms = list(images_and_density_maps)
        images_num = len(imgs_dms)
        for img, dm in imgs_dms:
            if _random.random() < self.probability:
                src_index = _np.random.randint(0, images_num)
                src_img, src_dm = imgs_dms[src_index]

                dest_h, dest_w = img.shape[:2]
                src_img, src_dm = _cv2.resize(src_img, (dest_w, dest_h)), _cv2.resize(src_dm, (dest_w, dest_h)),

                weight = _np.random.beta(self.alpha, self.alpha, size=1)[0]
                new_img = _cv2.addWeighted(img, weight, src_img, 1-weight, 0.0)
                new_dm = _cv2.addWeighted(dm, weight, src_dm, 1-weight, 0.0)

                yield new_img, new_dm
            else:
                yield img, dm
