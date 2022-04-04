import typing

import numpy as np

import CCAugmentation as cca


def generate_data(tuples_number: int, width: int = 10, height: int = 10) \
        -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """ Generate some img+DM tuples that all have the specified shape """
    imgs, dms = [], []
    for i in range(tuples_number):
        np.random.seed(i)
        imgs.append(np.random.rand(height, width, 3) * 255)
        dms.append(np.random.rand(height, width))
    return imgs, dms


def run_ops_in_trivial_pipeline(data: typing.Tuple[typing.Iterable[np.ndarray], typing.Iterable[np.ndarray]],
                                ops: typing.Iterable[cca.Operation]) -> typing.Tuple[typing.Collection[np.ndarray],
                                                                                     typing.Collection[np.ndarray]]:
    imgs, dms = data
    pipeline = cca.Pipeline(
        cca.CombinedLoader(img_loader=cca.VariableLoader(imgs), den_map_loader=cca.VariableLoader(dms), gt_loader=None),
        ops
    )
    return pipeline.execute_collect(verbose=False)


def run_op_in_trivial_pipeline(data: typing.Tuple[typing.Iterable[np.ndarray], typing.Iterable[np.ndarray]],
                               op: cca.Operation) -> typing.Tuple[typing.Collection[np.ndarray],
                                                                  typing.Collection[np.ndarray]]:
    return run_ops_in_trivial_pipeline(data, [op])
