import json
import random

import numpy as np
from tqdm import tqdm


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
        """ Starting with the input samples number, internally check for operations modifying the number and calculate the final size """
        output_samples_num = self.get_input_samples_number()
        for operation in self.operations:
            output_samples_num *= operation.get_output_samples_number_multiplier()
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

    def execute_collect(self, seed=None, return_np_arrays=False, verbose=True):
        """
        Execute the pipeline and return lists of the preprocessed, augmented data samples (one list for images,
        the other for density maps). In opposition to `execute_generate`, this method performs all operations on
        the whole dataset in one go.

        :param seed: Random seed. When it's not None, it allows reproducibility of the results.
        :param return_np_arrays: Whether to use np.arrays (ndarrays) to store images and density maps or Python lists. Generally, np.arrays are more useful when training a model but they don't support elements of varying size (search for 'ragged array'), so for safety, Python lists are the default output.
        :param verbose: If true, display a progress bar.
        :return: List of preprocessed data.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        images_and_density_maps = self._connect_operations()

        results = PipelineResultsIterator(images_and_density_maps, self.get_expected_output_samples_number(), verbose)
        images, density_maps = zip(*results)
        if return_np_arrays:
            return np.array(images), np.array(density_maps)
        else:
            return images, density_maps

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

    def to_json(self):
        loader_json = {'name': self.loader.__class__.__name__, 'args': self.loader.args}
        operations_json = [{'name': op.__class__.__name__, 'args': op.args} for op in self.operations]
        return {'loader': loader_json, 'operations': operations_json}


def read_pipeline_from_json(json_path):
    """
    Create a new Pipeline with the same configuration as the one deserialized from a JSON file.

    :param json_path: Path to the JSON with serialized Pipeline (configuration).
    :return: Pipeline object, or None if errors occurred.
    """
    def create_instance_invokation(package, name, args):
        args_strs = []
        for arg in args.items():
            if type(arg[1]) is dict and 'name' in arg[1] and 'args' in arg[1]:
                args_strs.append(f'{arg[0]}={create_instance_invokation(package, arg[1]["name"], args[1]["args"])}')
            elif type(arg[1]) is str:
                val = arg[1].replace("\\", "\\\\")
                args_strs.append(f'{arg[0]}="{val}"')
            else:
                args_strs.append(f'{arg[0]}={arg[1]}')
        return f'{package}.{name}({",".join(args_strs)})'

    with open(json_path, 'r') as f:
        pipeline_structure = json.load(f)

    import CCAugmentation.examples.loading as cca_ex_load
    import CCAugmentation.loaders as cca_load
    _, _ = cca_ex_load, cca_load  # these modules are actually used, don't remove

    loader = None
    loader_entry = pipeline_structure['loader']
    loader_name, loader_args = loader_entry['name'], loader_entry['args']
    for package in ['cca_ex_load', 'cca_load']:
        try:
            getattr(eval(package), loader_name)
            loader = eval(create_instance_invokation(package, loader_name, loader_args))
            break
        except AttributeError:
            pass
    if loader is None:
        return None

    import CCAugmentation.operations as cca_op
    import CCAugmentation.outputs as cca_out
    import CCAugmentation.transformations as cca_trans
    _, _, _ = cca_op, cca_out, cca_trans  # these modules are actually used, don't remove

    operations = []
    operations_entry = pipeline_structure['operations']
    for op in operations_entry:
        op_name, op_args = op['name'], op['args']
        created = False
        for package in ['cca_op', 'cca_out', 'cca_trans']:
            try:
                getattr(eval(package), op_name)
                inv = create_instance_invokation(package, op_name, op_args)
                operations.append(eval(inv))
                created = True
                break
            except AttributeError:
                pass
        if not created:
            return None

    return Pipeline(loader, operations)


def write_pipeline_to_json(pipeline, json_path, optimized=True):
    """
    Serialize Pipeline (configuration) to a JSON file.

    :param pipeline: Pipeline to serialize.
    :param json_path: Path where the serialized data will be stored.
    :param optimized: Whether to produce an optimized JSON, or a prettiefied one.
    :return: None.
    """
    with open(json_path, 'w') as f:
        json.dump(pipeline.to_json(), f, indent=(None if optimized else 2))
