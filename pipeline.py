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

    def execute_generate(self):
        """
        Execute the pipeline and return an iterable, most probably a generator, of the preprocessed, augmented data
        samples. Minimizes peak memory usage when there is no bottleneck in the pipeline. If you wish to preprocess
        everything in one go and have a list of the results, please consider using `execute_collect`.

        :return: Iterable of preprocessed data.
        """
        images_and_density_maps = self.loader.load()
        for operation in self.operations:
            images_and_density_maps = operation.execute(images_and_density_maps)
        return images_and_density_maps

    def execute_collect(self):
        """
        Execute the pipeline and return a list of the preprocessed, augmented data samples. In opposition to
        `execute_generate`, this method performs all operations on the whole dataset in one go.

        :return: List of preprocessed data.
        """
        images_and_density_maps = self.execute_generate()
        return list(images_and_density_maps)

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
