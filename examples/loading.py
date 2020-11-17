import os

from loaders import ImageFileLoader, GTPointsMatFileLoader, CombinedLoader


def _get_SHH_directory(dataset_dir, train_test, part):
    if train_test not in ["train", "test"]:
        raise ValueError("train_test must be either equal to 'train' or 'test'")
    if part.upper() not in ['A', 'B']:
        raise ValueError("Only 'A' and 'B' parts are allowed")
    return os.path.join(dataset_dir, f"part_{part}", f"{train_test}_data")


class _SHHImageLoader(ImageFileLoader):
    def __init__(self, dataset_dir, train_test, part):
        path = os.path.join(_get_SHH_directory(dataset_dir, train_test, part), "images")
        ImageFileLoader.__init__(self, path, "jpg")


class _SHHGTLoader(GTPointsMatFileLoader):
    def __init__(self, dataset_dir, train_test, part):
        path = os.path.join(_get_SHH_directory(dataset_dir, train_test, part), "ground-truth")
        GTPointsMatFileLoader.__init__(self, path)


class SHHLoader(CombinedLoader):
    def __init__(self, dataset_dir, train_test, part):
        img_loader = _SHHImageLoader(dataset_dir, train_test, part)
        gt_loader = _SHHGTLoader(dataset_dir, train_test, part)
        CombinedLoader.__init__(self, img_loader, gt_loader)
