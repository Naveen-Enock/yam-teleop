import numpy as np


def per_joint(value, n: int = 6, what: str = "value"):
    """Config value (None, scalar, or n-list) -> per-joint float array, or None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{what} must be a scalar or {n} values, got {arr.tolist()}")
    return arr
