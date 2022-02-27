import typing as _typing

import numpy as _np


_IMG_TYPE = _np.ndarray
_DM_TYPE = _np.ndarray
_IMG_DM_PAIR_TYPE = _typing.Tuple[_IMG_TYPE, _DM_TYPE]
_IMG_DM_ITER_TYPE = _typing.Iterable[_IMG_DM_PAIR_TYPE]
