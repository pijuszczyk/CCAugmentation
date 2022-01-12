# IMPORTANT
# Additional packages are required to effectively use the functionality below:
# - torch
# - torchvision


import numpy as _np


def create_iterable_dataset(pipeline_results):
    """
    Create a PyTorch iterable dataset that loads samples from pipeline results.

    Args:
        pipeline_results: Pipeline results iterator.

    Returns:
        Dataset that has valid PyTorch images saved as tensors and density maps.
    """
    from torch.utils.data import IterableDataset
    import torchvision.transforms as torch_transforms

    class PipelineDataset(IterableDataset):
        def __init__(self):
            IterableDataset.__init__(self)
            self.images_and_density_maps = pipeline_results
            self.image_transform = torch_transforms.Compose([
                torch_transforms.ToTensor()
            ])

        def __iter__(self):
            return self

        def __next__(self):
            image, density_map = next(self.images_and_density_maps)
            return self.image_transform(image.copy().astype("float32")), density_map.copy().astype("float32")

    return PipelineDataset()


def collate_variadic_size_samples_with_crop(samples):
    """
    Collate (image, density map) samples into PyTorch Tensor batches. Deals with inconsistent sample sizes by cropping
    samples into minimum common shapes. To be used in PyTorch DataLoader as collate_fn argument. Note: it is highly
    recommended to apply OptimizeBatch transformation when dealing with shape-inconsistent datasets.

    Args:
        samples: Image and density map pairs to be put into a batch.
    Returns:
        Two tensors - one with images and one with density maps.
    """
    import random
    import torch

    def crop(image, density_map, new_w, new_h):
        # very similar to transformations._crop but with color dimension being first instead of last
        h, w = image.shape[1:]

        x0 = random.randint(0, w - new_w)
        y0 = random.randint(0, h - new_h)
        x1 = x0 + new_w
        y1 = y0 + new_h

        new_img = image[:, y0:y1, x0:x1]
        new_den_map = density_map[y0:y1, x0:x1]

        return new_img, new_den_map

    imgs, dms = [it for it in zip(*samples)]
    shapes = _np.array([img.shape[-2:] for img in imgs])
    if _np.any(shapes != shapes[0]):
        min_h, min_w = _np.min(shapes, 0)
        imgs, dms = [it for it in zip(*[crop(img, dm, min_w, min_h) for img, dm in zip(imgs, dms)])]
    return torch.stack(imgs), torch.tensor(dms, dtype=torch.float32)
