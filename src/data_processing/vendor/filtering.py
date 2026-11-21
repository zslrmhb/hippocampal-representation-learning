"""Selected filtering functions from the supplied hippocampal_preprocessing package."""
import numpy as np
from scipy.signal import iirnotch
from .io import rows


def runs(mask):
    """Half-open runs of true samples, without joining separated regions."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def notch_design(config):
    b, a = iirnotch(config.notch_hz, config.notch_q, fs=config.sample_rate_hz)
    radius = float(np.max(np.abs(np.roots(a))))
    if not 0 < radius < 1:
        raise ValueError('Notch must have stable, decaying poles')
    # This is a relative pole-envelope tolerance, not an absolute voltage bound.
    guard = max(9, int(np.ceil(np.log(config.filter_tolerance)/np.log(radius))))
    return np.r_[b, a][None, :], dict(
        frequency_hz=config.notch_hz, q=config.notch_q, b=b.tolist(), a=a.tolist(),
        single_pass_bandwidth_hz=config.notch_hz/config.notch_q,
        max_pole_radius=radius, guard_samples=guard,
        guard_ms=guard*1000/config.sample_rate_hz,
        pole_envelope_tolerance=config.filter_tolerance,
        direction='forward-backward (zero phase); squared single-pass magnitude',
        padding='odd reflection, guard_samples per edge',
        support='entire independently filtered block, not just the trial window',
        interpretation='offline context-supported; not strictly odor-only or causal')


def artifact_mask(lfp, intervals, stats, config):
    """Freeze training thresholds, then screen ALL context used by the filter.

    A union across retained channels defines common block boundaries. Invalid
    values are excluded, never filled. Jumps invalidate both adjacent samples.
    """
    from .io import metadata
    invalid = np.zeros(metadata(lfp)['shape'][1], dtype=bool)
    totals = dict(nonfinite=0, amplitude=0, jump=0, flatline=0)
    flat_samples = max(2, int(np.ceil(config.flatline_ms*config.sample_rate_hz/1000)))
    for ch, row in enumerate(rows(lfp)):
        if not stats['channel_mask'][ch]:
            continue
        for interval in intervals:
            lo, hi = interval['start'], interval['end']
            x = row[lo:hi]
            nonfinite = ~np.isfinite(x)
            with np.errstate(invalid='ignore', over='ignore'):
                amplitude = np.abs(x-stats['robust_center'][ch]) > stats['amplitude_limit'][ch]
                jump = np.abs(np.diff(x)-stats['jump_center'][ch]) > stats['jump_limit'][ch]
            bad = nonfinite | amplitude
            bad[1:] |= jump
            bad[:-1] |= jump
            totals['nonfinite'] += int(nonfinite.sum())
            totals['amplitude'] += int(amplitude.sum())
            totals['jump'] += int(jump.sum())
            for start, end in runs(np.r_[False, np.diff(x) == 0]):
                if end-start+1 >= flat_samples:
                    bad[start-1:end] = True
                    totals['flatline'] += int(end-start+1)
            invalid[lo:hi] |= bad
    return invalid, totals
