import random


class Operation:
    def __init__(self):
        self.requires_full_dataset_in_memory = False

    def __str__(self):
        return self.__class__.__name__

    def execute(self, images_and_density_maps):
        raise NotImplementedError("execute method not implemented in the child class")


class Duplicate(Operation):
    def __init__(self, duplicates_num):
        Operation.__init__(self)
        self.duplicates_num = duplicates_num

    def execute(self, images_and_density_maps):
        for image_and_density_map in images_and_density_maps:
            for i in range(self.duplicates_num):
                yield image_and_density_map[0]


class Dropout(Operation):
    def __init__(self, probability):
        Operation.__init__(self)
        self.probability = probability

    def execute(self, images_and_density_maps):
        for image_and_density_map in images_and_density_maps:
            if random.random() >= self.probability:
                yield image_and_density_map[0]
