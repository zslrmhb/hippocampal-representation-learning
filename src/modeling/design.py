"""Frozen 500 ms amplitude design, feature order, and unrefitted masking."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing import preprocess, storage
from data_processing.artifacts import MODELS, read, require, sha
from . import models

N_FREQ = len(preprocess.FREQUENCIES)
FREQUENCY_MASKS = {'freq_0_4': (0, 4), 'freq_4_8': (4, 8), 'freq_8_12': (8, 12)}
TIME_MASKS = {'time_0_5': (0, 5), 'time_5_10': (5, 10)}

@dataclass(frozen=True)
class Run:
    output: Path

    def cache(self, session, target):
        return self.output / 'preprocessing' / session / target

def trial_frame(run, session, target):
    frame = pd.read_csv(run.cache(session, target) / 'matched_trials.csv')
    if not frame.trial_id.is_unique or not frame.split.isin(['train', 'val', 'test']).all():
        raise ValueError('Invalid matched-trial keys or split')
    return frame

def _rows(array, indices):
    """Use a view for the contiguous rows of an ordinary chronological split."""
    if len(indices) and indices[-1] - indices[0] + 1 == len(indices):
        return array[indices[0]:indices[-1] + 1]
    return array[indices]

@dataclass
class Design:
    """One rat/target/window, in matched-cache row order and on one allowed split."""

    frame: object
    window: tuple
    channels: tuple
    _cache: Path
    _source: object
    _indices: np.ndarray
    _split: str
    _transform_path: Path
    _record_transform: bool
    _scaled: np.ndarray | None = field(default=None, init=False, repr=False)
    _matrices: dict = field(default_factory=dict, init=False, repr=False)

    def _cached_array(self, name, width):
        array = np.load(self._cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != (len(self._source), 500, width) or array.dtype != np.float32:
            raise ValueError(f'Cached {name} shape/dtype disagrees with matched trials or QA')
        return array

    def _amplitude(self):
        if self._scaled is None:
            start, duration = self.window
            train = self._source.loc[self._source.split.eq('train'), 'trial_id'].to_numpy()
            parameters = None
            if self._transform_path.is_file():
                with np.load(self._transform_path, allow_pickle=False) as f:
                    if not np.array_equal(f['train_id'], train):
                        raise ValueError('Saved amplitude scaler train trial IDs differ')
                    parameters = {k: f[k] for k in ('mean', 'sd')}
            if self._split == 'test':
                if parameters is None:
                    raise ValueError('No frozen amplitude window scaler')
                cached = self._cached_array('amp', len(self.channels) * N_FREQ)
                x = _rows(cached[:, start:start + duration], self._indices)
                self._scaled = preprocess._window_scale(x, self.frame, start, duration,
                                                         parameters=parameters)
            else:
                fitted = {} if self._record_transform and parameters is None else None
                self._scaled, observed = preprocess.load_data(
                    self._cache.parents[2], self._cache.parent.name, self._cache.name,
                    start, duration, 'amp', splits=('train', 'val'),
                    amplitude_parameters=parameters, fitted_amplitude=fitted,
                )
                if not self.frame.equals(observed) or self._scaled.shape != (
                    len(self.frame), duration, len(self.channels) * N_FREQ
                ):
                    raise ValueError('Loaded amplitude disagrees with matched rows or channels')
                if fitted is not None and fitted:
                    if not np.isfinite(self._scaled).all():
                        raise ValueError('Nonfinite amplitude during train normalization')
                    storage.write_npz(self._transform_path, train_id=train, **fitted)
        return self._scaled

    def matrix(self, recipe, pooling):
        if (recipe, pooling) != ('amp_l1', 'bin50'):
            raise ValueError('Replay supports only the frozen amplitude recipe')
        key = ('amp', 'bin50')
        if key not in self._matrices:
            self._matrices[key] = models.direct_features(self._amplitude())
        return self._matrices[key]

    def _columns(self, recipe, pooling):
        if (recipe, pooling) != ('amp_l1', 'bin50'):
            raise ValueError('Replay supports only the frozen amplitude recipe')
        for time_bin in range(self.window[1] // 50):
            for channel in self.channels:
                for freq in range(N_FREQ):
                    yield 'amp', freq, time_bin, channel, None

def transform_path(run, session, target, window):
    return run.output / 'transforms' / f'{session}-{target}-{window}-amp.npz'


def prepare(run, session, target, window, *, split='dev', record_transform=False):
    """Create an aligned view; development never loads a test feature row."""
    window = (0, window)
    if window != (0, 500) or target not in ('odor', 'position'):
        raise ValueError(f'Unknown target or window: {target}, {window}')
    if split not in ('dev', 'test'):
        raise ValueError(f'Unknown poster split: {split}')
    source = trial_frame(run, session, target)
    indices = np.flatnonzero(source.split.isin(('train', 'val') if split == 'dev' else ('test',)))
    if not len(indices):
        raise ValueError(f'No matched {split} trials')
    cache = run.cache(session, target)
    qa = read(cache / 'qa.json')
    channels = tuple(qa['channel_keys'])
    if (not channels or len(set(channels)) != len(channels)
            or qa['n_matched'] != len(source) or qa['target'] != target
            or not np.array_equal(qa['frequencies_hz'], preprocess.FREQUENCIES)):
        raise ValueError('Cached channel, frequency or matched-trial QA does not agree')
    frame = source.iloc[indices].reset_index(drop=True)
    return Design(frame, window, channels, cache, source, indices, split,
                  transform_path(run, session, target, window[1]), record_transform)

def mask_predict(design, recipe, pooling, head, group):
    """Replace a frequency group with the fitted scaler's feature means."""
    lo, hi = FREQUENCY_MASKS[group]
    X = design.matrix(recipe, pooling)
    columns = np.fromiter((lo <= freq < hi for _, freq, *_ in design._columns(recipe, pooling)),
                          dtype=bool, count=X.shape[1])
    # Zeroing after scaling is exactly mean imputation, without changing the
    # floating-point transform of unmasked columns.
    scaler, estimator = (step for _, step in head.steps)
    standardized = scaler.transform(X)
    standardized[:, columns] = 0
    return estimator.predict(standardized)

def weight_rows(design, recipe, pooling, head):
    """Class-centered signed coefficients on standardized, electrode-specific features."""
    estimator = head.named_steps['logisticregression']
    columns = list(design._columns(recipe, pooling))
    coefficients = estimator.coef_
    if coefficients.shape[1] != len(columns):
        raise ValueError('Coefficient and feature column counts differ')
    if len(estimator.classes_) == 2:
        # In sklearn's binary model, only the class log-odds difference is stored.
        coefficients = np.concatenate((-.5 * coefficients, .5 * coefficients), axis=0)
    else:
        coefficients = coefficients - coefficients.mean(axis=0)
    return [dict(kind=kind, freq_index=freq, time_bin=time_bin, channel=channel,
                 channel_b=other, class_label=int(label), weight=float(weight))
            for label, weights in zip(estimator.classes_, coefficients)
            for weight, (kind, freq, time_bin, channel, other) in zip(weights, columns)]


def checkpoint_model(path, training, expected_sha):
    import torch
    import cebra
    require(sha(path) == expected_sha, 'Checkpoint SHA-256 changed')
    state = models._read_checkpoint(torch, path)
    expected = dict(architecture='offset10-model', n_features=training.shape[2], n_times=500,
                    n_trials=len(training), hidden_units=32, output_dim=8, batch_size=512,
                    lag_samples=50, learning_rate=.0003, temperature=1., seed=42, format_version=1,
                    data_hash=models._signature(training), torch_version=str(torch.__version__),
                    cebra_version=str(cebra.__version__))
    require(state['step'] == 1000 and state['settings'] == expected, 'Checkpoint settings or TRAIN data hash differs')


def matrices(design, checkpoint, expected_sha, train=None):
    direct = design.matrix('amp_l1', 'bin50')
    phase = _rows(design._cached_array('phase', 2 * len(design.channels) * N_FREQ), design._indices)
    combined = np.concatenate((design._amplitude(), phase), axis=-1)
    if train is not None:
        checkpoint_model(checkpoint, _rows(combined, train), expected_sha)
    else:
        require(sha(checkpoint) == expected_sha, 'Checkpoint SHA-256 changed')
    latent = models.latent_features(combined, checkpoint)
    require(latent.shape == (len(direct), 8) and np.isfinite(latent).all() and np.isfinite(direct).all(),
            'Invalid comparison feature matrix')
    return dict(zip(MODELS, (direct, latent), strict=True))
