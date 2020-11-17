import random

import cv2
import numpy as np

from .operations import Operation


class Transformation(Operation):
    """
    Abstract base class for any transformation that may occur on a set of image and density map pairs.
    Each transformation accepts an iterable (most commonly a generator) of image and density map pairs and is expected
    to return an iterable of the same cardinality. For custom transformations, please subclass this class or use
    LambdaTransformation.
    """
    def __init__(self, probability):
        """
        Create a new abstract transformation that is applied with specified probability.

        :param probability: Float value between 0 and 1 (inclusive).
        """
        Operation.__init__(self)
        self.probability = probability

    def execute(self, images_and_density_maps):
        """ See `transform_all` """
        return self.transform_all(images_and_density_maps)

    def transform(self, image, density_map):
        """
        Abstract method to be implemented in child classes. Most often, this and __init__ methods are the only ones
        that must be defined.

        :param image: Image that will be transformed.
        :param density_map: Density map that will be transformed according to the image transformation.
        :return: Transformed pair of image and density map.
        """
        raise NotImplementedError("transform method not implemented in the child class")

    def transform_all(self, images_and_density_maps):
        """
        Execute transformation with earlier specified probability on an iterable of img+DM pairs.

        :param images_and_density_maps: Iterable of img+DM pairs to maybe be transformed.
        :return: Iterable of maybe transformed img+DM pairs.
        """
        for image_and_density_map in images_and_density_maps:
            yield self.transform(*image_and_density_map) if random.random() < self.probability else image_and_density_map


class Crop(Transformation):
    """
    Cropping transformation that randomly cuts out and returns a part of an image with specified size (either fixed one
    or a fraction of the original one). Density map is also reduced to keep it relevant to the image.
    """
    def __init__(self, width, height, x_factor=None, y_factor=None, probability=1.0):
        """
        Define cropping with specified output size, applied with some probability. One may use a combination of
        fixed and relative size for separate image dimensions but fixed and relative size cannot be mixed for one
        dimension - one and only of them can be specified.

        :param width: Fixed output width.
        :param height: Fixed output height.
        :param x_factor: Output width relative to the input width.
        :param y_factor: Output height relative to the input height.
        :param probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if (width is not None and x_factor is not None) or (height is not None and y_factor is not None):
            raise ValueError("Cannot provide factor and fixed size at the same time")
        if (width is None and x_factor is None) or (height is None and y_factor is None):
            raise ValueError("Must provide factor or fixed size for both dimensions")

        Transformation.__init__(self, probability)
        self.width = width
        self.height = height
        self.x_factor = x_factor
        self.y_factor = y_factor

    def transform(self, image, density_map):
        """
        Crop an image at a random position to specified size.

        :param image: Image to be cropped.
        :param density_map: Density map to be cropped accordingly, with the same size as the image.
        :return: Cropped pair of image and density map.
        """
        h, w = image.shape
        new_w = round(w * self.x_factor) if self.width is None else self.width
        new_h = round(h * self.y_factor) if self.height is None else self.height

        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)
        x1 = x0 + new_w
        y1 = y0 + new_h

        new_img = image[y0:y1, x0:x1]
        new_den_map = density_map[y0:y1, x0:x1]

        return new_img, new_den_map


class Scale(Transformation):
    """
    Scaling transformation that increases or decreases input size.
    """
    def __init__(self, width, height, x_factor=None, y_factor=None, probability=1.0):
        """
        Create a scaling transformation that scales the image and the corresponding density map to a specified fixed or
        relative size with a given probability. One may use a combination of fixed and relative size for separate image
        dimensions but fixed and relative size cannot be mixed for one dimension - one and only of them can be specified.

        :param width: Fixed output width.
        :param height: Fixed output height.
        :param x_factor: Output width relative to the input width.
        :param y_factor: Output height relative to the input height.
        :param probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        if (width is not None and x_factor is not None) or (height is not None and y_factor is not None):
            raise ValueError("Cannot provide factor and fixed size at the same time")
        if (width is None and x_factor is None) or (height is None and y_factor is None):
            raise ValueError("Must provide factor or fixed size for both dimensions")

        Transformation.__init__(self, probability)
        self.width = width
        self.height = height
        self.x_factor = x_factor
        self.y_factor = y_factor

    def transform(self, image, density_map):
        """
        Scale an image and the corresponding density map to a specified size.

        :param image: Image to be scaled.
        :param density_map: Density map to be scaled accordingly.
        :return: Scaled pair of image and density map.
        """
        if self.width and self.height:
            h, w = image.shape
            scale_x = self.width / w
            scale_y = self.height / h

            new_img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            new_den_map = cv2.resize(density_map, (self.width, self.height), interpolation=cv2.INTER_LINEAR) / scale_x / scale_y
        else:
            new_img = cv2.resize(image, None, self.x_factor, self.y_factor, interpolation=cv2.INTER_CUBIC)
            new_den_map = cv2.resize(density_map, None, self.x_factor, self.y_factor, interpolation=cv2.INTER_LINEAR) / self.x_factor / self.y_factor
        return new_img, new_den_map


class StandardizeSize(Transformation):
    """
    Standardizes image and density map sizes in order to reduce variance in size and allow bigger batches.
    This transformation takes a list of allowed aspect ratios for the images and the base size corresponding to the
    length of the longer side of an image and scales each image (and relevant density map) to the size best fitting
    the original size.
    """
    def __init__(self, std_aspect_ratios, std_base_size):
        """
        Create a size standardization transformation.

        :param std_aspect_ratios: List of aspect ratios defined as float values that are allowed to exist in the output images. In case of ratios suited for portrait/vertical mode, inversion of the same aspect ratio from horizontal mode is used, e.g. ratio of 4:3 vertical image is to be seen as 3:4 (or 0.75, to be exact).
        :param std_base_size: Desired length of the longer side of the output images and density maps.
        """
        Transformation.__init__(self, 1.0)
        self.std_ratios, self.std_bounds = self._prepare_standard_aspect_ratios(std_aspect_ratios)
        self.std_base_size = std_base_size

    @staticmethod
    def _prepare_standard_aspect_ratios(std_ratios):
        """
        Find boundaries between consequent allowed aspect ratios to make finding the most appropriate ratios easier.
        Boundaries are such aspect ratio values that are uniformly placed between two nearest allowed aspect ratios.

        :param std_ratios: Aspect ratios that are allowed for the output images and density maps.
        :return: Sorted allowed aspect ratios and calculated boundaries between them.
        """
        ratios = sorted(std_ratios)
        boundaries = []
        for i in range(1, len(ratios)):
            boundaries.append((ratios[i - 1] + ratios[i]) / 2)
        return ratios, boundaries

    @staticmethod
    def _find_the_most_similar_ratio(ratio_to_improve, std_ratios, std_boundaries):
        """
        Find an allowed aspect ratio that is the most similar to the one provided.

        :param ratio_to_improve: Aspect ratio that maybe can be improved.
        :param std_ratios: Allowed aspect ratios.
        :param std_boundaries: Uniformly distributed boundaries between allowed aspect ratios.
        :return: Allowed aspect ratio that is the most similar to the provided one.
        """
        chosen_std_ratio = std_ratios[-1]
        for i in range(len(std_boundaries)):
            if ratio_to_improve < std_boundaries[i]:
                chosen_std_ratio = std_ratios[i]
                break

        return chosen_std_ratio

    def transform(self, image, density_map):
        """
        Scale the image and its density map to such a size that its aspect ratio is in the allowed ones.

        :param image: Image with any aspect ratio.
        :param density_map: Relevant density map of the same size as image.
        :return: Scaled pair of image and its density map.
        """
        h, w = image.shape

        chosen_std_ratio = self._find_the_most_similar_ratio(w / h, self.std_ratios, self.std_bounds)

        if chosen_std_ratio >= 1.0:
            new_w = self.std_base_size
            new_h = int(new_w / chosen_std_ratio)
        else:
            new_h = self.std_base_size
            new_w = int(new_h * chosen_std_ratio)

        # if the size cannot be standardized further, return
        if (w, h) == (new_w, new_h):
            return image, density_map

        scale_x = new_w / w
        scale_y = new_h / h

        new_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        new_den_map = cv2.resize(density_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR) / scale_x / scale_y

        return new_img, new_den_map


class Normalize(Transformation):
    """
    Normalizes image pixel values using one of normalization methods.
    """
    def __init__(self, method):
        """
        Create a transformation that normalizes image pixel values using one of the following methods:

        - `range_0_to_1` - values are scaled to be in fixed <0; 1> range
        - `range_-1_to_1` - values are scaled to be in fixed <-1; 1> range
        - `samplewise_centering` - values are translated to make their mean (in respect to a single image) equal to 0
        - `samplewise_std_normalization` - values are scaled to make their sample standard deviation (in respect to a single image) equal to 1
        - `featurewise_centering` - values are translated to make their mean (in respect to all images) equal to 0
        - `featurewise_std_normalization` - values are scaled to make their sample standard deviation (in respect to all images) equal to 1

        :param method: Method that will be used for normalization, one of the above.
        """
        if method not in ["range_0_to_1", "range_-1_to_1", "samplewise_centering", "samplewise_std_normalization",
                          "featurewise_centering", "featurewise_std_normalization"]:
            raise ValueError(f"Wrong method of normalization selected: {method}")

        Transformation.__init__(self, 1.0)
        if method.startswith("featurewise"):
            self.requires_full_dataset_in_memory = True
        self.method = method

    def transform(self, image, density_map):
        """
        Transform the given image, return the untouched density map.

        :param image: Image whose pixel values will be normalized.
        :param density_map: Density map that won't be affected.
        :return: Pair of transformed image and corresponding density map.
        """
        if self.method == "range_-1_to_1":
            return (image - 127.5) / 255.0, density_map
        elif self.method == "range_0_to_1":
            return image / 255.0, density_map
        elif self.method == "samplewise_centering":
            return image - np.mean(image), density_map
        elif self.method == "samplewise_std_normalization":
            return image / np.std(image), density_map

    def transform_all(self, images_and_density_maps):
        """
        Overriding method looping over all img+DM pairs from an iterable.
        Behaves the same like the base class implementation when using a method that doesn't require full dataset to be
        in memory, or first collects all samples and then transforms them otherwise.

        :param images_and_density_maps: Iterable of pairs of images and corresponding density maps.
        :return: Iterable of transformed img+DM pairs.
        """
        if self.method.startswith("range") or self.method.startswith("samplewise"):
            return Transformation.transform_all(self, images_and_density_maps)
        all_images, all_density_maps = zip(*list(images_and_density_maps))
        if self.method == "featurewise_centering":
            return zip(all_images - np.mean(all_images), all_density_maps)
        elif self.method == "featurewise_std_normalization":
            return zip(all_images / np.std(all_images), all_density_maps)


class FlipLR(Transformation):
    """
    Horizontal / left-right random flipping transformation.
    """
    def __init__(self, probability=0.5):
        """
        Create LR flipping transformation that flips with the given probability. In most cases, should stay at 0.5
        (equal chances).

        :param probability: Probability of flipping the image and its density map. Between 0 and 1 (inclusive). In most cases, should stay at 0.5.
        """
        Transformation.__init__(self, probability)

    def transform(self, image, density_map):
        """
        Horizontally flip the image and its density map.

        :param image: Input image.
        :param density_map: Corresponding density map.
        :return: Flipped image and density map.
        """
        return np.fliplr(image), np.fliplr(density_map)


class ToGrayscale(Transformation):
    """
    Transformation that converts images to grayscale (reduces channels to 1).
    """
    def __init__(self, probability=1.0):
        """
        Create the transformation applied with the given probability.

        :param probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        """
        Transformation.__init__(self, probability)

    def transform(self, image, density_map):
        """
        Transform the image to grayscale.

        :param image: BGR image. Note that a loader can load an originally grayscale image to a BGR image. In such cases, the transformation doesn't provide any real effect.
        :param density_map: Density map that won't be affected.
        :return: Pair of grayscale image and its density map.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), density_map


class LambdaTransformation(Transformation):
    """
    One of the two ways to create a custom transformation, the other one being subclassing `Transformation`.
    This class works by applying a transformation specified as a lambda, over all samples. Alternatively, one use a
    custom loop, also specified as a lambda.
    """
    def __init__(self, probability, transformation, loop=None):
        """
        Create a custom transformation that applies a given transformation lambda over all samples. Alternatively, a
        custom loop lambda can be given, e.g. to collect all samples before executing a transformation on them.

        :param probability: Probability for the transformation to be applied, between 0 and 1 (inclusive).
        :param transformation: Lambda that takes a pair of img+DM and returns a transformed pair of img+DM.
        :param loop: Lambda that takes an iterable of img+DM pairs and transformation lambda and returns an iterable of transformed img+DM pairs. If None, standard loop is used.
        """
        Transformation.__init__(self, probability)
        self.transformation = transformation
        self.loop = loop

    def transform(self, image, density_map):
        """ Run the given lambda on a img+DM pair """
        return self.transformation(image, density_map)

    def transform_all(self, images_and_density_maps):
        """ If a custom loop is defined, use it to loop over all samples from the iterable. Otherwise, use the standard loop. """
        if self.loop is None:
            return Transformation.transform_all(self, images_and_density_maps)
        return self.loop(images_and_density_maps, self.transformation)
