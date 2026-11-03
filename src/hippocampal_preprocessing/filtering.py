"""Offline notch cleaning within independent split-owned, artifact-free blocks."""
import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, welch, freqz
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


def clean_trials(lfp, raw, candidates, intervals, raw_stats, config):
    """Return common filtered windows plus before/after training spectra.

    Raw quality fitting precedes this function. Final filtered-data quality and
    scaling are fitted separately, after edge and context-artifact exclusions.
    """
    sos, info = notch_design(config)
    guard = info['guard_samples']
    invalid, totals = artifact_mask(lfp, intervals, raw_stats, config)
    blocks, spans = [], []
    for interval in intervals:
        lo, hi = interval['start'], interval['end']
        for start, end in runs(invalid[lo:hi]):
            spans.append(dict(split=interval['split'], segment=interval['segment'],
                              start=int(lo+start), end=int(lo+end)))
        for start, end in runs(~invalid[lo:hi]):
            start, end = int(lo+start), int(lo+end)
            usable_start = min(end, start+guard)
            blocks.append(dict(block_id=len(blocks), split=interval['split'],
                               segment=interval['segment'], start=start, end=end,
                               usable_start=usable_start, usable_end=max(usable_start, end-guard),
                               usable=end-start > 2*guard))
    selected = []
    for i, trial in enumerate(candidates):
        if trial['reasons']:
            continue
        block = next((b for b in blocks if b['usable'] and
                      b['split'] == trial['split'] and
                      b['usable_start'] <= trial['start'] and trial['end'] <= b['usable_end']), None)
        if block is None:
            trial['reasons'].append('filter_artifact' if invalid[trial['start']:trial['end']].any()
                                    else 'filter_edge')
        else:
            trial.update(filter_block_id=block['block_id'], filter_source_start=block['start'],
                         filter_source_end=block['end'], filter_guard_samples=guard)
            selected.append(i)
    if not any(candidates[i]['split'] == 'train' for i in selected):
        raise ValueError('No training windows remain after filter context/edge screening')
    kept = [candidates[i] for i in selected]
    before = raw[selected]
    filtered = np.zeros_like(before)
    train_blocks = [b for b in blocks if b['usable'] and b['split'] == 'train']
    nperseg = min(int(round(4*config.sample_rate_hz)),
                 max(b['usable_end']-b['usable_start'] for b in train_blocks))
    step = max(1, nperseg//2)
    noverlap = nperseg-step
    freq = np.fft.rfftfreq(nperseg, 1/config.sample_rate_hz)
    before_psd = np.zeros((len(freq), raw.shape[-1]))
    after_psd = np.zeros_like(before_psd)
    weights = np.zeros(raw.shape[-1], dtype=np.int64)
    block_trials = {b['block_id']: [] for b in blocks}
    for i, trial in enumerate(kept):
        block_trials[trial['filter_block_id']].append(i)
    for ch, row in enumerate(rows(lfp)):
        if not raw_stats['channel_mask'][ch]:
            continue
        for block in blocks:
            if not block['usable']:
                continue
            indices = block_trials[block['block_id']]
            length = block['usable_end']-block['usable_start']
            psd_block = block['split'] == 'train' and length >= nperseg
            if not indices and not psd_block:
                continue
            start, end = block['start'], block['end']
            clean = sosfiltfilt(sos, row[start:end], padlen=guard)
            for i in indices:
                trial = kept[i]
                filtered[i, :, ch] = clean[trial['start']-start:trial['end']-start]
            if psd_block:
                left, right = block['usable_start'], block['usable_end']
                _, p0 = welch(row[left:right], fs=config.sample_rate_hz,
                              nperseg=nperseg, noverlap=noverlap)
                _, p1 = welch(clean[left-start:right-start], fs=config.sample_rate_hz,
                              nperseg=nperseg, noverlap=noverlap)
                weight = 1+(length-nperseg)//step
                before_psd[:, ch] += weight*p0
                after_psd[:, ch] += weight*p1
                weights[ch] += weight
    mask = raw_stats['channel_mask']
    before_psd[:, mask] /= weights[mask]
    after_psd[:, mask] /= weights[mask]
    info.update(raw_context_artifact_channel_sample_counts=totals,
                invalid_context_samples=int(invalid.sum()), flatline_ms=config.flatline_ms,
                psd_source='guarded continuous TRAIN blocks; before/after same source support',
                psd_nperseg=nperseg, psd_window_count=int(weights[mask][0]))
    spectrum = dict(frequency_hz=freq, before_power=before_psd, after_power=after_psd)
    return filtered, before, kept, info, blocks, spans, spectrum


def save_notch_qa(output, before, after, kept, keys, mask, info, spectrum, config, plots):
    """Original-unit spectra compare cleaning, independently of z-scoring."""
    import json
    freq = spectrum['frequency_hz']
    p0, p1 = spectrum['before_power'][:, mask], spectrum['after_power'][:, mask]
    nearest = int(np.argmin(np.abs(freq-config.notch_hz)))
    tiny = np.finfo(float).tiny
    channel_qa = []
    for ch, key in enumerate(keys):
        channel_qa.append(dict(channel=str(key), measured_frequency_hz=float(freq[nearest]),
            before_power=float(p0[nearest, ch]), after_power=float(p1[nearest, ch]),
            attenuation_db=float(10*np.log10(max(p1[nearest, ch], tiny)/max(p0[nearest, ch], tiny)))))
    info['channels'] = channel_qa
    (output/'notch_quality.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'notch_training_psd.npz', frequency_hz=freq,
                        before_power=p0, after_power=p1, channel_keys=keys)
    f, h = freqz(info['b'], info['a'], worN=16384, fs=config.sample_rate_hz)
    np.savez_compressed(output/'notch_response.npz', frequency_hz=f,
                        single_pass_amplitude=np.abs(h), effective_amplitude=np.abs(h)**2)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        axes[0].semilogy(freq[1:], np.median(p0[1:], axis=1), label='Before notch')
        axes[0].semilogy(freq[1:], np.median(p1[1:], axis=1), label='After notch')
        axes[0].set(xlim=(40, 80), xlabel='Hz', ylabel='Stored units²/Hz',
                    title='Continuous training blocks: median channel PSD')
        axes[0].legend()
        i = next(i for i, t in enumerate(kept) if t['split'] == 'train')
        time_ms = np.arange(before.shape[1])*1000/config.sample_rate_hz
        axes[1].plot(time_ms, before[i, :, 0], label='Before notch', lw=.8)
        axes[1].plot(time_ms, after[i, :, 0], label='After notch', lw=.8)
        axes[1].set(xlabel='Milliseconds after odor onset', ylabel='Stored units',
                    title=f'First retained training trial, {keys[0]}')
        axes[1].legend()
        fig.savefig(output/'notch_quality.png', dpi=140)
        plt.close(fig)
