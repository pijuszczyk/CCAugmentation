import random

import numpy as np
from tqdm import tqdm

from .operations import Duplicate, Dropout


class PipelineResultsIterator:
    """
    Iterator for img+DM pairs on pipeline's output.
    """
    def __init__(self, images_and_density_maps, total_samples, verbose=True):
        """
        Create an iterator for img+DM pairs coming from the pipeline.

        :param images_and_density_maps: Iterator of preprocessed img+DM pairs.
        :param total_samples: Total expected number of samples on output.
        :param verbose: Whether to display a progress bar.
        """
        self._images_and_density_maps = images_and_density_maps
        self._progress = tqdm(total=total_samples) if verbose and total_samples is not None else None

    def __iter__(self):
        """ Return itself """
        return self

    def __next__(self):
        """ Return next img+DM pair, updating the progress bar if it's enabled """
        next_result = next(self._images_and_density_maps)
        if self._progress is not None:
            self._progress.update(1)
        return next_result


class Pipeline:
    """
    Pipelines define the data preprocessing and augmentation tasks, starting from loading the original data, through
    various transformations, up to showing and saving the results. They provide an easy to define and adjust workflow,
    that can be stored and summarized with ease.

    Unless loading the whole dataset is needed at some point, they provide means of loading just the required minimum,
    optimizing memory usage.

    First, a data loader must be selected. Such a loader needs to return an iterable of pairs of images and
    corresponding density maps. Then, a list of operations is prepared. This includes duplicating, cropping, flipping,
    saving results to files, etc. Finally, such a pipeline is executed to gather the results.
    """
    def __init__(self, loader, operations):
        """
        Create a new pipeline that loads the data using the given `loader` and performs `operations` on them.

        :param loader: Loader that loads pairs of images and corresponding density maps, providing an iterable of such data.
        :param operations: List of operations that will be executed on the loaded data.
        """
        self.loader = loader
        self.operations = operations
        self.requires_full_dataset_in_memory = any([op.requires_full_dataset_in_memory for op in operations])

    def get_input_samples_number(self):
        """ Get number of samples the loader will load """
        return self.loader.get_number_of_loadable_samples()

    def get_expected_output_samples_number(self):
        """ Starting with the input samples number, check for operations modifying the number and calculate the final size """
        output_samples_num = self.get_input_samples_number()
        for operation in self.operations:
            if type(operation) is Duplicate:
                output_samples_num *= operation.duplicates_num
            elif type(operation) is Dropout:
                output_samples_num *= operation.probability
        return output_samples_num

    def _connect_operations(self):
        """
        Connect the loader with all the prepared operations sequentially to create a working pipeline.

        :return: Iterable of img+DM pairs.
        """
        images_and_density_maps = self.loader.load()
        for operation in self.operations:
            images_and_density_maps = operation.execute(images_and_density_maps)
        return images_and_density_maps

    def execute_generate(self, seed=None):
        """
        Execute the pipeline and return an iterable of the preprocessed, augmented data samples. Minimizes peak
        memory usage when there is no bottleneck in the pipeline. If you wish to preprocess everything in one go and
        have a list of the results, please consider using `execute_collect`.

        :param seed: Random seed. When it's not None, it allows reproducibility of the results.
        :return: Iterable of preprocessed data.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        images_and_density_maps = self._connect_operations()

        return PipelineResultsIterator(images_and_density_maps, self.get_expected_output_samples_number(), False)

    def execute_collect(self, seed=None, verbose=True):
        """
        Execute the pipeline and return a list of the preprocessed, augmented data samples. In opposition to
        `execute_generate`, this method performs all operations on the whole dataset in one go.

        :param seed: Random seed. When it's not None, it allows reproducibility of the results.
        :param verbose: If true, display a progress bar.
        :return: List of preprocessed data.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        images_and_density_maps = self._connect_operations()

        return list(PipelineResultsIterator(images_and_density_maps, self.get_expected_output_samples_number(), verbose))

    def summary(self):
        """
        Print a summary of the pipeline.
        """
        width = 80
        print("*" * width)
        print("Pipeline summary")
        print("*" * width)
        print("")
        print("Operations:")
        for i, op in enumerate(self.operations):
            print(f"{str(i)}. {str(op)}")
        print("")
        print(f"Requires full dataset in memory: {self.requires_full_dataset_in_memory}")
