"""Fixed 50 ms pooling, L1 decoder, and CPU inference from frozen CEBRA checkpoints."""
import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _sync_directory(directory):
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_write(path, writer, *, previous=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        if previous is not None and path.exists():
            os.replace(path, previous)
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_joblib(obj, path):
    """Replace a fitted decoder atomically. The output path belongs to this run."""
    _atomic_write(path, lambda file: joblib.dump(obj, file, compress=1))


def _trials(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or min(X.shape) < 1:
        raise ValueError(f"Expected nonempty trials x time x features, got {X.shape}")
    return X

def direct_features(X):
    X = _trials(X)
    if X.shape[1] % 50:
        raise ValueError('50 ms pooling requires complete bins')
    bins = X.reshape(len(X), X.shape[1] // 50, 50, X.shape[2])
    return bins.mean(axis=2).reshape(len(X), -1)

def fit_l1(X_train, y_train, C, seed):
    """Fit a categorical decoder. Caller controls the split and validation selection."""
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train)
    if (
        y.ndim != 1
        or X.ndim != 2
        or len(X) != len(y)
        or not np.issubdtype(y.dtype, np.integer)
    ):
        raise ValueError(
            "Expected a trial-feature matrix and one integer class per trial"
        )
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError('Training requires at least two distinct classes')
    estimator = LogisticRegression(C=float(C), solver='saga', class_weight='balanced',
                                   random_state=seed, max_iter=2000, tol=1e-3, penalty='l1')
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(X, y)
    # Inspect the solver directly. Python warning filters are shared between threads.
    pipeline.converged_ = int(estimator.n_iter_.max()) < estimator.max_iter
    pipeline.training_counts_ = dict(zip(labels.tolist(), counts.tolist()))
    return pipeline

def _neural_libs():
    # Direct baselines can be run on machines without a CUDA environment.
    import torch
    import cebra.models

    return torch, cebra


def _build_model(cebra, settings, device):
    return cebra.models.init(
        settings["architecture"],
        num_neurons=settings["n_features"],
        num_units=settings["hidden_units"],
        num_output=settings["output_dim"],
    ).to(device)


def _signature(X):
    digest = hashlib.blake2b(digest_size=20)
    # Chunk across trials so slicing a memory map need not make another full copy.
    for trial in X:
        digest.update(np.ascontiguousarray(trial).view(np.uint8))
    return digest.hexdigest()


def _read_checkpoint(torch, path, settings=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "settings",
        "step",
        "model",
        "optimizer",
        "torch_rng",
        "sampler_rng",
        "cuda_rng",
        "losses",
    }
    if (
        not isinstance(state, dict)
        or not required.issubset(state)
        or not isinstance(state["settings"], dict)
    ):
        raise EOFError("Incomplete checkpoint payload")
    if settings is not None and state["settings"] != settings:
        raise ValueError(
            f"Incompatible encoder checkpoint at {path}; choose a fresh output directory"
        )
    if (
        not isinstance(state["step"], int)
        or state["step"] < 0
        or len(state["losses"]) != state["step"]
    ):
        raise EOFError("Invalid checkpoint update count")
    return state

def latent_features(X, checkpoint_path):
    """Mean-pool valid within-trial encoder centers on CPU."""
    torch, cebra = _neural_libs()
    X = _trials(X)
    saved = _read_checkpoint(torch, checkpoint_path)
    settings = saved["settings"]
    if X.shape[2] != settings["n_features"] or X.shape[1] != settings["n_times"]:
        raise ValueError("Decoder input shape and trained epoch differ")
    model = _build_model(cebra, settings, 'cpu')
    model.load_state_dict(saved["model"])
    model.eval()
    offset = model.get_offset()
    width = int(offset.left + offset.right)
    n_out = X.shape[1] - width + 1
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), 16):
            batch = (
                torch.from_numpy(np.ascontiguousarray(X[start : start + 16]))
                .transpose(1, 2)
            )
            latent = model(batch)
            if latent.ndim != 3 or latent.shape[-1] != n_out:
                raise RuntimeError(
                    f"Unexpected temporal encoder output {tuple(latent.shape)}"
                )
            outputs.append(latent.mean(-1).cpu().numpy())
    return np.concatenate(outputs)
