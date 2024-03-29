import numpy as np

import CCAugmentation as cca


def generate_data(tuples_number, width=10, height=10):
    imgs, dms = [], []
    for i in range(tuples_number):
        np.random.seed(i)
        imgs.append(np.random.randint(0, 256, (height, width, 3)))
        dms.append(np.random.randint(0, 100, (height, width)) / 100.0)
    return imgs, dms


def run_ops_in_trivial_pipeline(data, ops):
    imgs, dms = data
    pipeline = cca.Pipeline(
        cca.CombinedLoader(img_loader=cca.VariableLoader(imgs), den_map_loader=cca.VariableLoader(dms), gt_loader=None),
        ops
    )
    return pipeline.execute_collect(verbose=False)
