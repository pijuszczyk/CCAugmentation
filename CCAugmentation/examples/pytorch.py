def create_iterable_dataset(torch_transforms_module, pipeline_results):
    """
    Create a PyTorch iterable dataset that loads samples from pipeline results.

    :param torch_transforms_module: The imported torch.transforms module.
    :param pipeline_results: Pipeline results iterator.
    :return: Dataset that has valid PyTorch images saved as tensors and density maps.
    """
    class PipelineDataset:
        def __init__(self):
            self.images_and_density_maps = pipeline_results
            self.image_transform = torch_transforms_module.Compose([
                torch_transforms_module.ToTensor()
            ])

        def __iter__(self):
            for image, density_map in self.images_and_density_maps:
                yield self.image_transform(image.copy().astype("float32")), density_map.copy().astype("float32")

    return PipelineDataset()


def create_data_loader(torch_transforms_module, dataset, batch_size):
    """
    Create a loader similar to PyTorch DataLoader but only with a single thread and no shuffling. Allows batching
    results in a way that batches are created only from samples with the same shape. If not enough samples of the same
    shape were present in a row, incomplete batches are returned.

    :param torch_transforms_module: The imported torch.transforms module.
    :param dataset: Dataset that yields tuples of image and density map.
    :param batch_size: Preferred batch size.
    :return: Iterator of batches of data from the dataset.
    """
    class PipelineDataLoader:
        def __init__(self):
            self.dataset = dataset
            self.batch_size = batch_size
            self._loaded_data = sorted([(image, density_map) for (image, density_map) in self.dataset], key=lambda t: t[0].shape)
            self._iterator = iter(self._loaded_data)
            self._batch = []
            self._current_batch_size = 0

        def _unload_batch_into_tensors(self):
            tensor_images = torch_transforms_module.ToTensor()(list(zip(*self._batch))[0])
            tensor_density_maps = torch_transforms_module.ToTensor()(list(zip(*self._batch))[1])
            self._batch = []
            self._current_batch_size = 0
            return tensor_images, tensor_density_maps

        def _add_to_batch(self, image, density_map):
            self._batch.append((image, density_map))
            self._current_batch_size += 1

        def __len__(self):
            return len(self._loaded_data)

        def __next__(self):
            for _ in range(self._current_batch_size, self.batch_size):
                try:
                    image, density_map = next(self._iterator)
                except StopIteration:
                    self._iterator = iter(self._loaded_data)
                    raise
                if self._current_batch_size != 0 and self._batch[0][0].shape != image.shape:
                    tensor_images, tensor_density_maps = self._unload_batch_into_tensors()
                    self._add_to_batch(image, density_map)
                    return tensor_images, tensor_density_maps
                else:
                    self._add_to_batch(image, density_map)
            tensor_images, tensor_density_maps = self._unload_batch_into_tensors()
            return tensor_images, tensor_density_maps

    return PipelineDataLoader()
