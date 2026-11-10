"""Reconstruct trials exclusively from sparse raw behavioral events."""
import numpy as np
from .io import metadata, rows


def time_segments(time, expected_hz=1000.0):
    if len(time) < 2 or not np.isfinite(time).all():
        raise ValueError('TimeBin must contain finite timestamps')
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError('Nonpositive TimeBin step')
    step = 1.0 / expected_hz
    gaps = np.flatnonzero(dt > 1.5 * step)
    regular = dt <= 1.5 * step
    if not regular.any() or not np.allclose(dt[regular], step, rtol=1e-4, atol=1e-8):
        raise ValueError('Inconsistent sample rate within TimeBin segments')
    bounds = np.r_[0, gaps + 1, len(time)]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def reconstruct(time, events, expected_hz=1000.0):
    """events maps names to {raw_index: nonzero_value}; retain every odor onset."""
    segments = time_segments(time, expected_hz)
    onsets = sorted(set().union(*(events[f'Odor{i}'] for i in range(1, 6))))
    transitions = sorted(events['PokeEvents'].items())
    brackets, anomalies = [], []
    for lo, hi in segments:
        active, bad = None, False
        for index, value in transitions:
            if not lo <= index < hi:
                continue
            if value == 1:
                if active is not None:
                    anomalies.append(dict(index=index, reason='repeated_entry'))
                    bad = True
                else:
                    active, bad = index, False
            elif value == -1:
                if active is None:
                    anomalies.append(dict(index=index, reason='unmatched_exit'))
                else:
                    brackets.append((active, index, bad))
                    active = None
            else:
                anomalies.append(dict(index=index, reason='invalid_poke_value'))
                if active is not None:
                    bad = True
        if active is not None:
            anomalies.append(dict(index=active, reason='unmatched_entry'))
    trials = []
    for trial_id, onset in enumerate(onsets):
        reasons = []
        odors = [(i, events[f'Odor{i+1}'][onset]) for i in range(5) if onset in events[f'Odor{i+1}']]
        positions = [(i, events[f'Position{i+1}'][onset]) for i in range(5) if onset in events[f'Position{i+1}']]
        odor = odors[0][0] if len(odors) == 1 else -1
        position = positions[0][0] if len(positions) == 1 else -1
        if len(odors) != 1 or any(v != 1 for _, v in odors):
            reasons.append('ambiguous_odor')
        if len(positions) != 1 or any(v != 1 for _, v in positions):
            reasons.append('ambiguous_position')
        if any(onset-1 in events[f'Odor{i+1}'] or onset+1 in events[f'Odor{i+1}'] for i, _ in odors):
            reasons.append('odor_not_single_sample_pulse')
        status = [events[k].get(onset, 0) for k in ('InSeqLog', 'PerformanceLog')]
        if any(v not in (-1, 1) for v in status):
            reasons.append('invalid_status')
        matching = [(a, b, bad) for a, b, bad in brackets if a <= onset < b]
        entry = exit_ = None
        if len(matching) != 1:
            reasons.append('missing_poke_bracket')
        else:
            entry, exit_, bad = matching[0]
            if bad:
                reasons.append('ambiguous_poke_bracket')
            if sum(entry <= k < exit_ for k in onsets) != 1:
                reasons.append('multiple_onsets_in_poke')
        segment = next(i for i, (a, b) in enumerate(segments) if a <= onset < b)
        trials.append(dict(trial_id=trial_id, onset=onset, onset_s=float(time[onset]), odor=odor,
                           position=position, inseq_raw=status[0], correct_raw=status[1],
                           inseq=int(status[0] == 1), correct=int(status[1] == 1), poke_in=entry,
                           poke_out=exit_, poke_in_s=None if entry is None else float(time[entry]),
                           poke_out_s=None if exit_ is None else float(time[exit_]),
                           sampling_duration_ms=None if exit_ is None else (exit_-onset)*1000/expected_hz,
                           segment=segment, odor_offset_s=None, reasons=reasons))
    return trials, segments, anomalies


def load_behavior(path, expected_hz=1000.0):
    meta = metadata(path)
    required = ['TimeBin', 'PokeEvents', 'InSeqLog', 'PerformanceLog'] + [
        f'{kind}{i}' for kind in ('Odor', 'Position') for i in range(1, 6)]
    if not set(required) <= set(meta['keys']):
        raise ValueError('Missing required behavior rows')
    events, time, movement = {}, None, {}
    for key, row in zip(meta['keys'], rows(path)):
        if key == 'TimeBin':
            time = row.copy()
        elif key in required or 'Reward' in key:
            if not np.isfinite(row).all():
                raise ValueError(f'Nonfinite behavioral events: {key}')
            ids = np.flatnonzero(row)
            events[key] = dict(zip(ids.tolist(), row[ids].tolist()))
    trials, segments, anomalies = reconstruct(time, events, expected_hz)
    ids = [t['onset'] for t in trials]
    for key, row in zip(meta['keys'], rows(path)):
        if key in ('XvalRatMazePosition', 'YvalRatMazePosition'):
            movement[key] = row[ids]
    for i, trial in enumerate(trials):
        for key, values in movement.items():
            trial[key] = float(values[i])
        a, b = trial['poke_in'], trial['poke_out']
        trial['reward_indices_during_poke'] = {key: [] if a is None else [k for k in values if a <= k <= b]
                                             for key, values in events.items() if 'Reward' in key}
    return time, trials, segments, anomalies, meta, events
