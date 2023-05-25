import numpy as np
from collections.abc import Sequence


def classify_dict_kargs(dict_kargs):
    arr_like = {}
    others = {}

    for k, v in dict_kargs.items():
        if isinstance(v, str):
            others[k] = v
        elif isinstance(v, (Sequence, np.ndarray)):
            arr_like[k] = v
        else:
            others[k] = v

    return arr_like, others


def record(
        log_array: np.ndarray,
        data: np.ndarray) -> np.ndarray:

    return np.append(log_array, np.expand_dims(data, axis=0), axis=0)