import os

from CCAugmentation import loaders


def _get_SHH_directory(dataset_dir, train_test, part):
    if train_test not in ["train", "test"]:
        raise ValueError("train_test must be either equal to 'train' or 'test'")
    if part.upper() not in ['A', 'B']:
        raise ValueError("Only 'A' and 'B' parts are allowed")
    return os.path.join(dataset_dir, f"part_{part}", f"{train_test}_data")


class _SHHImageLoader(loaders.ImageFileLoader):
    def __init__(self, dataset_dir, train_test, part):
        path = os.path.join(_get_SHH_directory(dataset_dir, train_test, part), "images")
        loaders.ImageFileLoader.__init__(self, path, "jpg")


class _SHHGTLoader(loaders.GTPointsMatFileLoader):
    def __init__(self, dataset_dir, train_test, part):
        path = os.path.join(_get_SHH_directory(dataset_dir, train_test, part), "ground-truth")
        loaders.GTPointsMatFileLoader.__init__(self, path, lambda v: v["image_info"][0, 0][0, 0][0] - 1)


class SHHLoader(loaders.CombinedLoader):
    def __init__(self, dataset_dir, train_test, part):
        img_loader = _SHHImageLoader(dataset_dir, train_test, part)
        gt_loader = _SHHGTLoader(dataset_dir, train_test, part)
        loaders.CombinedLoader.__init__(self, img_loader, gt_loader)


def _get_NWPU_indices_for_set(dataset_dir, train_val_test):
    if train_val_test not in ["train", "val", "test"]:
        raise ValueError("train_val_test must be either equal to 'train', 'val' or 'test'")
    list_file_path = os.path.join(dataset_dir, f"{train_val_test}.txt")
    indices = []
    with open(list_file_path, 'r') as f:
        for line in f:
            num = line.split(' ')[0]
            indices.append(num)
    return indices


class _NWPUImageLoader(loaders.BasicImageFileLoader):
    def __init__(self, dataset_dir, train_val_test):
        indices = _get_NWPU_indices_for_set(dataset_dir, train_val_test)
        paths = [os.path.join(dataset_dir, "images", f"{index}.jpg") for index in indices]
        loaders.BasicImageFileLoader.__init__(self, paths)


class _NWPUGTLoader(loaders.BasicGTPointsMatFileLoader):
    def __init__(self, dataset_dir, train_val_test):
        indices = _get_NWPU_indices_for_set(dataset_dir, train_val_test)
        paths = [os.path.join(dataset_dir, "mats", f"{index}.mat") for index in indices]
        loaders.BasicGTPointsMatFileLoader.__init__(self, paths, lambda v: v["annPoints"])


class NWPULoader(loaders.CombinedLoader):
    def __init__(self, dataset_dir, train_val_test):
        img_loader = _NWPUImageLoader(dataset_dir, train_val_test)
        gt_loader = _NWPUGTLoader(dataset_dir, train_val_test)
        loaders.CombinedLoader.__init__(self, img_loader, gt_loader)
