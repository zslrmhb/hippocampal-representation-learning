"""Boundary-safe context contract; this is not a CEBRA training adapter."""
import numpy as np


def eligible_centers(valid, left=0, right=0, offset=0):
    """Trial x sample mask requiring both centers' full contexts in one trial."""
    if any(not isinstance(v, (int, np.integer)) for v in (left, right, offset)):
        raise ValueError('Context and offset must be integer sample counts')
    if left < 0 or right < 0:
        raise ValueError('Context must be nonnegative')
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError('Expected trial x sample mask')
    result = np.zeros_like(valid)
    n = valid.shape[1]
    lo, hi = max(left, left-offset), min(n-right, n-right-offset)
    for center in range(lo, hi):
        a = valid[:, center-left:center+right+1].all(axis=1)
        b = valid[:, center+offset-left:center+offset+right+1].all(axis=1)
        # A temporal positive must not jump over invalid samples either.
        bridge = valid[:, min(center, center+offset)-left:max(center, center+offset)+right+1].all(axis=1)
        result[:, center] = a & b & bridge
    return result


def pair_indices(valid, left=0, right=0, offset=1):
    """Return (trial, center, positive_center) rows, never flattened neighbors."""
    if offset == 0:
        raise ValueError('Temporal-positive offset must be nonzero')
    trial, center = np.nonzero(eligible_centers(valid, left, right, offset))
    return np.column_stack((trial, center, center + offset))
