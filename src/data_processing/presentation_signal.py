"""Recording-level diagnostics for the historical presentation EDA notebook."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from . import preprocess
from .vendor.events import load_behavior
from .vendor.io import metadata, rows
from .vendor.filtering import notch_design, runs
from .vendor.quality import partition_trials

FS = 1000
WINDOW = 10 * FS
WELCH_SAMPLES = 4 * FS
CHANNELS = [f'T{i}' for i in range(1, 24)]
SESSIONS = {
    'Mitt': '080718_mitt', 'Barat': '081106_barat', 'Stella': '090212_stella',
    'Superchris': '090212_superchris', 'Buchanan': '090420_buchanan',
}


def channel_names(path):
    return [k.split('_', 1)[0] for k in metadata(path)['keys']]


def selected_channels(path, channels):
    return {k: row for k, row in zip(channel_names(path), rows(path), strict=True) if k in channels}


def _window_psd(windows):
    hz, power = welch(windows, fs=FS, window='hann', nperseg=WELCH_SAMPLES,
                      noverlap=WELCH_SAMPLES // 2, axis=1)
    return hz, np.median(power, axis=0)


def recording_stats(pairs):
    """Complete, nonoverlapping 10 s windows; never bridge a clock gap."""
    summaries = {}
    for name, session in SESSIONS.items():
        lfp, behavior = pairs[session]
        time, _, segments, *_ = load_behavior(behavior)
        keys = channel_names(lfp)
        if len(time) != metadata(lfp)['shape'][1]:
            raise ValueError(f'{session}: unaligned recordings')
        counts = [(b - a) // WINDOW for a, b in segments]
        local_sd, overall_sd, psds, longest = [], [], [], (0, None, 0)
        for key, row in zip(keys, rows(lfp), strict=True):
            overall_sd.append(np.std(row))
            matrices = [row[a:a+n*WINDOW].reshape(n, WINDOW) for (a, _), n in zip(segments, counts) if n]
            windows = matrices[0] if len(matrices) == 1 else np.concatenate(matrices)
            local_sd.append(np.std(windows, axis=1))
            hz, psd = _window_psd(windows)
            psds.append(psd)
            if name == 'Buchanan':
                for a, b in segments:
                    for lo, hi in runs(np.diff(row[a:b]) == 0):
                        if hi - lo + 1 > longest[0]:
                            longest = (hi - lo + 1, key, a + lo)
        raw_sd = np.array(overall_sd)
        sd = np.array(local_sd)
        psd = np.array(psds)
        flank = (abs(hz - 60) >= 3) & (abs(hz - 60) <= 6)
        line = abs(hz - 60) <= 1
        def ratio(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan), where=(a > 0) & (b > 0))
        summaries[name] = dict(keys=keys, time=time, cuts=np.cumsum(counts)[:-1], hz=hz, psd=psd,
                               local_change=np.log2(ratio(sd, np.median(sd, axis=1, keepdims=True))),
                               relative_sd=np.log2(ratio(raw_sd, np.median(raw_sd))),
                               line_db=10 * np.log10(ratio(psd[:, line].max(axis=1), np.median(psd[:, flank], axis=1))),
                               plateau=longest)
    return summaries


def alignment_example(pair):
    """One real 200 ms odor-aligned sample window and its raw behavior markers."""
    lfp, behavior = pair
    _, trials, _, *_ = load_behavior(behavior)
    trial = next(t for t in trials if not t['reasons'] and t['correct'] == t['inseq'] == 1)
    onset = trial['onset']
    window = slice(onset - 250, onset + 450)
    traces = {k: v[window] for k, v in selected_channels(lfp, ('T1', 'T2', 'T3', 'T23')).items()}
    fields = ('Odor' + str(trial['odor'] + 1), 'Position' + str(trial['position'] + 1), 'PokeEvents', 'InSeqLog')
    marks = {k: r[window] for k, r in zip(metadata(behavior)['keys'], rows(behavior), strict=True) if k in fields}
    return trial, traces, marks


def trial_quality(pair):
    """Recompute the Superchris example and training-only artifact limits."""
    lfp, behavior = pair
    time, trials, segments, *_ = load_behavior(behavior)
    assigned = [dict(t, reasons=list(t['reasons'])) for t in trials]
    intervals = partition_trials(assigned, segments, len(time), preprocess.CONFIG)
    ends = {(x['split'], x['segment']): x['end'] for x in intervals}
    candidates = [t for original, t in zip(trials, assigned, strict=True)
                  if not original['reasons'] and t['inseq'] == t['correct'] == 1]
    keys = channel_names(lfp)
    stats = preprocess._training_thresholds(lfp, candidates, ends, len(keys))
    c = keys.index('T1')
    center, limit = stats['robust_center'][c], stats['amplitude_limit'][c]
    row = selected_channels(lfp, ('T1',))['T1']
    examples = {t['trial_id']: row[t['onset']:t['onset']+500] for t in trials if t['trial_id'] in (0, 2)}
    return examples, center, limit, pd.DataFrame(assigned)


def notch_spectrum(pair):
    """Paired PSD for one real channel in the same continuous training interval."""
    lfp, behavior = pair
    time, trials, segments, *_ = load_behavior(behavior)
    intervals = partition_trials(trials, segments, len(time), preprocess.CONFIG)
    sos, info = notch_design(preprocess.CONFIG)
    pad = info['guard_samples']
    interval = next(v for v in intervals if v['split'] == 'train' and v['end'] - v['start'] >= WINDOW + 2*pad)
    a = interval['start']
    raw = selected_channels(lfp, ('T1',))['T1'][a:interval['end']]
    clean = sosfiltfilt(sos, raw, padtype='odd', padlen=pad)
    size = (len(raw) - 2*pad) // WINDOW * WINDOW
    hz, before = _window_psd(raw[pad:pad+size].reshape(-1, WINDOW))
    _, after = _window_psd(clean[pad:pad+size].reshape(-1, WINDOW))
    return hz, before, after


def transition_averages(pair):
    """Mitt correct InSeq odor and between-odor traces (exploratory reconstruction)."""
    lfp, behavior = pair
    _, raw_trials, segments, *_ = load_behavior(behavior)
    trials = [t for t in raw_trials if not t['reasons']]
    def good(t):
        return t['correct'] == t['inseq'] == 1

    # The departing odor, not the next odor's identity, determines the gap group.
    events = [(f'Odor {chr(65+t["odor"])}', t['poke_in'], min(1200, t['poke_out']-t['poke_in']), t['segment'])
              for t in trials if good(t)]
    for a, b in zip(raw_trials, raw_trials[1:]):
        if a['reasons'] or b['reasons'] or not good(a) or a['odor'] >= 4 or a['segment'] != b['segment']:
            continue
        gap = b['poke_in'] - a['poke_out']
        if 0 < gap <= 10 * FS:
            events.append((f'After odor {chr(65+a["odor"])}', a['poke_out'], min(4000, gap), a['segment']))
    summary = {}
    mean_lfp = np.zeros(metadata(lfp)['shape'][1])
    n_channels = 0
    for row in rows(lfp):
        mean_lfp += row
        n_channels += 1
    mean_lfp /= n_channels
    sos = butter(4, 30, fs=FS, btype='lowpass', output='sos')
    for name, onset, length, segment in events:
        lo, hi = segments[segment]
        if not (lo <= onset - 1000 and onset + length + 500 <= hi):
            continue
        x = sosfiltfilt(sos, mean_lfp[onset-1000:onset+length+500])[500:-500]
        x -= x[:500].mean()
        width = 1700 if name.startswith('Odor') else 4500
        if name not in summary:
            summary[name] = [np.zeros(width), np.zeros(width, dtype=int), 0]
        s, support, n = summary[name]
        s[:len(x)] += x
        support[:len(x)] += 1
        summary[name][2] = n + 1
    return {name: (np.divide(s, counts, where=counts > 0, out=np.full(len(s), np.nan)), n)
            for name, (s, counts, n) in summary.items()}
