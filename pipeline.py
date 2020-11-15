class Pipeline:
    def __init__(self, loader, operations):
        self.loader = loader
        self.operations = operations
        self.requires_full_dataset_in_memory = any([op.requires_full_dataset_in_memory for op in operations])

    def execute_generate(self):
        images_and_density_maps = self.loader.load()
        for operation in self.operations:
            images_and_density_maps = operation.execute(images_and_density_maps)
        return images_and_density_maps

    def execute_collect(self):
        images_and_density_maps = self.execute_generate()
        return list(images_and_density_maps)

    def summary(self):
        width = 30
        print("*" * width)
        print("Pipeline summary")
        print("*" * width)
        print("")
        print("Operations:")
        for i, op in enumerate(self.operations):
            print(f"{str(i)}. {str(op)}")
        print("")
        print(f"Requires full dataset in memory: {self.requires_full_dataset_in_memory}")
