"""Bounded-memory reads of the provider's numeric NPZ arrays."""
from pathlib import Path
import zipfile
import numpy as np
from numpy.lib import format


def header(stream):
    version = format.read_magic(stream)
    readers = {(1, 0): format.read_array_header_1_0, (2, 0): format.read_array_header_2_0}
    if version not in readers:
        raise ValueError(f'Unsupported NPY version {version}')
    shape, fortran, dtype = readers[version](stream)
    if dtype.hasobject or dtype.kind not in 'biuf':
        raise ValueError('Expected real numeric data')
    return shape, fortran, dtype


def metadata(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        with archive.open('data.npy') as stream:
            shape, fortran, dtype = header(stream)
        with archive.open('keys.npy') as stream:
            keys = format.read_array(stream, allow_pickle=False).tolist()
    if len(shape) != 2 or fortran or len(keys) != shape[0] or len(set(keys)) != len(keys):
        raise ValueError(f'Expected C-order channels x samples with unique keys: {path}')
    return dict(path=str(path.resolve()), shape=list(shape), dtype=str(dtype),
                keys=keys, size_bytes=path.stat().st_size)


def rows(path):
    with zipfile.ZipFile(path) as archive, archive.open('data.npy') as stream:
        shape, fortran, dtype = header(stream)
        if len(shape) != 2 or fortran:
            raise ValueError('Expected C-order 2-D data')
        size = shape[1] * dtype.itemsize
        for _ in range(shape[0]):
            buf = stream.read(size)
            if len(buf) != size:
                raise ValueError(f'Truncated data: {path}')
            yield np.frombuffer(buf, dtype=dtype)
