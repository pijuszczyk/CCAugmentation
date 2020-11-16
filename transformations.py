import random

import cv2
import numpy as np

from operations import Operation


class Transformation(Operation):
    def __init__(self, probability):
        Operation.__init__(self)
        self.probability = probability

    def execute(self, images_and_density_maps):
        return self.transform_all(images_and_density_maps)

    def transform(self, image, density_map):
        raise NotImplementedError("transform method not implemented in the child class")

    def transform_all(self, images_and_density_maps):
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            yield self.transform(image, density_map) if random.random() < self.probability else image, density_map


class Crop(Transformation):
    def __init__(self, width, height, x_factor=None, y_factor=None, probability=1.0):
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
    def __init__(self, width, height, x_factor=None, y_factor=None, probability=1.0):
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
    def __init__(self, std_aspect_ratios, std_base_size, probability=1.0):
        Transformation.__init__(self, probability)
        self.std_ratios, self.std_bounds = self._prepare_standard_aspect_ratios(std_aspect_ratios)
        self.std_base_size = std_base_size

    def _prepare_standard_aspect_ratios(self, std_ratios):
        ratios = sorted(std_ratios)
        boundaries = []
        for i in range(1, len(ratios)):
            boundaries.append((ratios[i - 1] + ratios[i]) / 2)
        return ratios, boundaries

    def _find_the_most_suitable_ratio(self, ratio_to_improve, std_ratios, std_boundaries):
        chosen_std_ratio = std_ratios[-1]
        for i in range(len(std_boundaries)):
            if ratio_to_improve < std_boundaries[i]:
                chosen_std_ratio = std_ratios[i]
                break

        return chosen_std_ratio

    def transform(self, image, density_map):
        h, w = image.shape

        chosen_std_ratio = self._find_the_most_suitable_ratio(w / h, self.std_ratios, self.std_bounds)

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
    def __init__(self, type, probability=1.0):
        if type not in ["range_0_to_1", "range_-1_to_1", "samplewise_centering", "samplewise_std_normalization",
                        "featurewise_centering", "featurewise_std_normalization"]:
            raise ValueError(f"Wrong type of normalization selected: {type}")

        Transformation.__init__(self, probability)
        if type.startswith("featurewise"):
            self.requires_full_dataset_in_memory = True
        self.type = type

    def transform(self, image, density_map):
        if self.type == "range_-1_to_1":
            return (image - 127.5) / 255.0, density_map
        elif self.type == "range_0_to_1":
            return image / 255.0, density_map
        elif self.type == "samplewise_centering":
            return image - np.mean(image), density_map
        elif self.type == "samplewise_std_normalization":
            return image / np.std(image), density_map

    def transform_all(self, images_and_density_maps):
        if self.type.startswith("range") or self.type.startswith("samplewise"):
            return Transformation.transform_all(self, images_and_density_maps)
        all_images, all_density_maps = zip(*list(images_and_density_maps))
        if self.type == "featurewise_centering":
            return all_images - np.mean(all_images), all_density_maps
        elif self.type == "featurewise_std_normalization":
            return all_images / np.std(all_images), all_density_maps


class FlipLR(Transformation):
    def __init__(self, probability=0.5):
        Transformation.__init__(self, probability)

    def transform(self, image, density_map):
        return np.fliplr(image), np.fliplr(density_map)


class ToGrayscale(Transformation):
    def __init__(self, probability=1.0):
        Transformation.__init__(self, probability)

    def transform(self, image, density_map):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), density_map


class Lambda(Transformation):
    def __init__(self, probability, transformation, loop=None):
        Transformation.__init__(self, probability)
        self.transformation = transformation
        self.loop = loop

    def transform(self, image, density_map):
        return self.transformation(image, density_map)

    def transform_all(self, images_and_density_maps):
        if self.loop is None:
            return Transformation.transform_all(self, images_and_density_maps)
        return self.loop(images_and_density_maps, self.transformation)
