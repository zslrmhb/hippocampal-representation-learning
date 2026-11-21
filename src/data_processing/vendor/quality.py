"""Selected preprocessing functions from the supplied hippocampal_preprocessing package."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    window_ms: float = 500.0
    sample_rate_hz: float = 1000.0
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    artifact_z: float = 20.0
    left_context: int = 0
    right_context: int = 0
    positive_offset: int = 1
    profile: str = 'baseline'
    notch_hz: float = 60.0
    notch_q: float = 30.0
    filter_tolerance: float = 1e-6
    flatline_ms: float = 100.0
    window_start_ms: float = 0.0
    include_outseq: bool = False

    def validate(self):
        values = (self.window_ms, self.sample_rate_hz, self.train_fraction,
                  self.validation_fraction, self.artifact_z, self.window_start_ms)
        if not np.isfinite(values).all():
            raise ValueError('Configuration must be finite')
        if self.window_ms <= 0 or self.sample_rate_hz <= 0 or self.artifact_z <= 0:
            raise ValueError('Window, sample rate and artifact threshold must be positive')
        if self.window_start_ms < 0:
            raise ValueError('Window start must be at or after odor onset')
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1-self.train_fraction:
            raise ValueError('Split fractions must leave nonempty train/validation/test')
        samples_per_ms = self.sample_rate_hz / 1000
        offset = self.window_start_ms * samples_per_ms
        n = self.window_ms * samples_per_ms
        if not np.isfinite(offset) or not np.isclose(offset, round(offset), rtol=0, atol=1e-8):
            raise ValueError('Window start must align with a sample boundary')
        if not np.isfinite(n) or not np.isclose(n, round(n), rtol=0, atol=1e-8) or round(n) < 2:
            raise ValueError('Window must correspond to at least two integer samples')
        for value in (self.left_context, self.right_context, self.positive_offset):
            if not isinstance(value, int):
                raise ValueError('Context and offset must be integers')
        if self.positive_offset == 0:
            raise ValueError('Temporal-positive offset must be nonzero')
        if min(self.left_context, self.right_context) < 0:
            raise ValueError('Context must be nonnegative')
        if self.left_context + self.right_context + abs(self.positive_offset) >= round(n):
            raise ValueError('Context and positive offset leave no eligible centers')
        if self.profile not in ('baseline', 'inseq-decoding'):
            raise ValueError('Profile must be baseline or inseq-decoding')
        if self.include_outseq and self.profile != 'inseq-decoding':
            raise ValueError('--include-outseq applies to inseq-decoding. Baseline already includes both trial types.')
        if self.profile == 'inseq-decoding':
            if not np.isfinite((self.notch_hz, self.notch_q, self.filter_tolerance, self.flatline_ms)).all():
                raise ValueError('Filter configuration must be finite')
            if not 0 < self.notch_hz < self.sample_rate_hz/2 or self.notch_q <= 0:
                raise ValueError('Notch must be below Nyquist with positive Q')
            if not 0 < self.filter_tolerance < 1 or self.flatline_ms <= 0:
                raise ValueError('Filter tolerance must be in (0,1), flatline duration positive')
        return int(round(n))


def partition_trials(trials, segments, n_samples, config):
    """Assign entire source intervals first; purge windows crossing a boundary."""
    window = config.validate()
    offset = round(config.window_start_ms * (config.sample_rate_hz / 1000))
    cuts = [0, int(n_samples*config.train_fraction),
            int(n_samples*(config.train_fraction+config.validation_fraction)), n_samples]
    split_names = ['train', 'validation', 'test']
    intervals = []
    for split, a, b in zip(split_names, cuts[:-1], cuts[1:]):
        for segment, (lo, hi) in enumerate(segments):
            if max(a, lo) < min(b, hi):
                intervals.append(dict(split=split, segment=segment, start=max(a, lo), end=min(b, hi)))
    for trial in trials:
        onset = trial['onset']
        start = onset + offset
        end = start + window
        # A delayed window still belongs to the onset's split and segment.
        owner = next(i for i in range(3) if cuts[i] <= onset < cuts[i+1])
        trial.update(start=start, end=end, split=split_names[owner])
        if end > cuts[owner+1]:
            trial['reasons'].append('split_boundary')
        if end > segments[trial['segment']][1]:
            trial['reasons'].append('recording_boundary')
        # Exit is exclusive. A window ending exactly at exit is valid.
        if trial['poke_out'] is not None and end > trial['poke_out']:
            trial['reasons'].append('early_withdrawal')
    return intervals


def robust_stats(values):
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 0.0
    center = float(np.median(finite))
    scale = float(1.4826*np.median(np.abs(finite-center)))
    # Sparse/discrete signals may have zero MAD; SD is an explicit fallback.
    if scale == 0:
        scale = float(np.std(finite))
    return center, scale
