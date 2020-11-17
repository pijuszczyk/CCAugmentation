import random


class Operation:
    """
    Abstract base class for all types of operations that may be put into a pipeline.
    """
    def __init__(self):
        """
        Create the Operation.
        By default, `requires_full_dataset_in_memory` member variable is set to False to tell that
        the operation works with generators (may process just one sample at a time) and doesn't require the full dataset
        to be stored in the memory. It may be overriden in the subclasses.
        """
        self.requires_full_dataset_in_memory = False

    def __str__(self):
        """ Stringify the operation """
        return self.__class__.__name__

    def execute(self, images_and_density_maps):
        """ Abstract method that must be implemented in the subclasses, should take and return an iterable of img+DM pairs """
        raise NotImplementedError("execute method not implemented in the child class")


class Duplicate(Operation):
    """
    Duplicates each sample in a dataset a specified number of times. One of the most helpful operations when it comes
    to data augmentation.
    """
    def __init__(self, duplicates_num):
        """
        Define duplication.

        :param duplicates_num: Each sample will be repeated that number of times.
        """
        Operation.__init__(self)
        self.duplicates_num = duplicates_num

    def execute(self, images_and_density_maps):
        """ Duplicates samples """
        for image_and_density_map in images_and_density_maps:
            for i in range(self.duplicates_num):
                yield image_and_density_map


class Dropout(Operation):
    """
    Drops out samples with a given probability.
    """
    def __init__(self, probability):
        """
        Define dropout.

        :param probability: Each sample will be dropped out with this probability, meaning that the estimated number of output images for a dataset with `N` samples is `N*(1-probability)`.
        """
        Operation.__init__(self)
        self.probability = probability

    def execute(self, images_and_density_maps):
        """ Drops out samples """
        for image_and_density_map in images_and_density_maps:
            if random.random() >= self.probability:
                yield image_and_density_map
