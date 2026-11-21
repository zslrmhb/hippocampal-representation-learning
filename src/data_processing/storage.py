"""Atomic artifacts and advisory locking for private replay."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile

import numpy as np

@contextmanager
def temporary_for(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + "-")
    os.close(fd)
    try:
        yield Path(name)
        with open(name, "rb") as saved:
            os.fsync(saved.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, data):
    with temporary_for(path) as tmp:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_npz(path, **values):
    with temporary_for(path) as tmp, tmp.open("wb") as file:
        np.savez_compressed(file, **values)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is already using {path.parent}.")
        yield
