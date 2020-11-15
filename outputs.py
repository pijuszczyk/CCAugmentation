import csv
import os
import pickle

import cv2
import matplotlib.pyplot as plt

from operations import Operation


class Output(Operation):
    def __init__(self):
        Operation.__init__(self)

    def execute(self, images_and_density_maps):
        return self.output(images_and_density_maps)

    def output(self, images_and_density_maps):
        raise NotImplementedError("output method not implemented in the child class")


class Demonstrate(Output):
    def __init__(self, max_examples=None, show_density_map=True):
        Output.__init__(self)
        self.max_examples = max_examples
        self.show_density_map = show_density_map

    def output(self, images_and_density_maps):
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            if cnt < self.max_examples:
                cols = 2 if self.show_density_map else 1
                _, axes = plt.subplots(1, cols, figsize=(20, 4))
                axes[0].set_title(f"Image {str(cnt)}")
                axes[0].imshow(image)
                if self.show_density_map:
                    axes[1].set_title(f"Density map {str(cnt)}")
                    axes[1].imshow(density_map)
                plt.show()
            yield image, density_map
            cnt += 1


class SaveImagesToFiles(Output):
    def __init__(self, dir_path, file_extension="png"):
        Output.__init__(self)
        self.dir_path = dir_path
        self.file_extension = file_extension

    def output(self, images_and_density_maps):
        os.makedirs(self.dir_path)
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            path = os.path.join(self.dir_path, f"IMG_{str(cnt)}.{self.file_extension}")
            cv2.imwrite(path, image)
            yield image, density_map
            cnt += 1


class SaveImagesToBinaryFile(Output):
    def __init__(self, file_path, keep_3_dimensions=True):
        Output.__init__(self)
        self.requires_full_dataset_in_memory = True
        self.file_path = file_path
        self.keep_3_dimensions = keep_3_dimensions

    def output(self, images_and_density_maps):
        dir_path = os.path.dirname(self.file_path)
        os.makedirs(dir_path)
        images = []
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            if self.keep_3_dimensions and len(image.shape) != 3:
                images.append(image.copy().reshape(*image.shape, 1))
            else:
                images.append(image)
            yield image, density_map
        with open(self.file_path, 'wb') as f:
            pickle.dump(images, f)


class SaveDensityMapsToCSVFiles(Output):
    def __init__(self, dir_path):
        Output.__init__(self)
        self.dir_path = dir_path

    def output(self, images_and_density_maps):
        os.makedirs(self.dir_path)
        cnt = 0
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            path = os.path.join(self.dir_path, f"GT_{str(cnt)}.csv")
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerows(density_map)
            yield image, density_map
            cnt += 1


class SaveDensityMapsToBinaryFile(Output):
    def __init__(self, file_path, keep_3_dimensions=True):
        Output.__init__(self)
        self.requires_full_dataset_in_memory = True
        self.file_path = file_path
        self.keep_3_dimensions = keep_3_dimensions

    def output(self, images_and_density_maps):
        dir_path = os.path.dirname(self.file_path)
        os.makedirs(dir_path)
        den_maps = []
        for image_and_density_map in images_and_density_maps:
            image, density_map = image_and_density_map[0]
            if self.keep_3_dimensions and len(density_map.shape) != 3:
                den_maps.append(density_map.copy().reshape(*density_map.shape, 1))
            else:
                den_maps.append(density_map)
            yield image, density_map
        with open(self.file_path, 'wb') as f:
            pickle.dump(den_maps, f)
