"""Raw-only, per-session preprocessing with protected chronological partitions."""
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import ctypes
import sys
import json
import os
import platform
import shutil
import tempfile
import numpy as np
from .context import eligible_centers
from .events import load_behavior
from .io import metadata, rows


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

    def validate(self):
        values = (self.window_ms, self.sample_rate_hz, self.train_fraction,
                  self.validation_fraction, self.artifact_z)
        if not np.isfinite(values).all():
            raise ValueError('Configuration must be finite')
        if self.window_ms <= 0 or self.sample_rate_hz <= 0 or self.artifact_z <= 0:
            raise ValueError('Window, sample rate and artifact threshold must be positive')
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1-self.train_fraction:
            raise ValueError('Split fractions must leave nonempty train/validation/test')
        n = self.window_ms * self.sample_rate_hz / 1000
        if not np.isclose(n, round(n), atol=1e-8) or round(n) < 2:
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
        return int(round(n))


def partition_trials(trials, segments, n_samples, config):
    """Assign entire source intervals first; purge windows crossing a boundary."""
    window = config.validate()
    cuts = [0, int(n_samples*config.train_fraction),
            int(n_samples*(config.train_fraction+config.validation_fraction)), n_samples]
    split_names = ['train', 'validation', 'test']
    intervals = []
    for split, a, b in zip(split_names, cuts[:-1], cuts[1:]):
        for segment, (lo, hi) in enumerate(segments):
            if max(a, lo) < min(b, hi):
                intervals.append(dict(split=split, segment=segment, start=max(a, lo), end=min(b, hi)))
    for trial in trials:
        start, end = trial['onset'], trial['onset']+window
        owner = next(i for i in range(3) if cuts[i] <= start < cuts[i+1])
        trial.update(start=start, end=end, split=split_names[owner])
        if end > cuts[owner+1]:
            trial['reasons'].append('split_boundary')
        if end > segments[trial['segment']][1]:
            trial['reasons'].append('recording_boundary')
        # Exit is exclusive. [onset,onset+n) ending exactly at exit is valid.
        if trial['poke_out'] is not None and end > trial['poke_out']:
            trial['reasons'].append('early_withdrawal')
    return intervals


def unique_support(trials, selected):
    """Indices and first flattened positions; overlaps counted exactly once."""
    if len(selected) == 0:
        raise ValueError('No eligible training trials')
    indices = np.concatenate([np.arange(trials[i]['start'], trials[i]['end'], dtype=np.int64)
                              for i in selected])
    return np.unique(indices, return_index=True)


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


def quality_and_scale(data, trials, config):
    """data: candidate trials x samples x channels, with corresponding manifests.

    Fit robust thresholds on unique candidate train support. Reject any trial
    with a nonfinite/amplitude/jump violation on a retained channel. Then fit
    mean/SD on unique retained training samples; held-out values never fit rules.
    """
    train = np.array([i for i, t in enumerate(trials) if t['split'] == 'train'], dtype=int)
    support, first = unique_support(trials, train)
    fit = data[train].reshape(-1, data.shape[-1])[first].astype(np.float64)
    c = data.shape[-1]
    center, amplitude_scale, jump_center, jump_scale = [np.zeros(c) for _ in range(4)]
    channel_mask = np.zeros(c, dtype=bool)
    adjacent = np.diff(support) == 1
    for ch in range(c):
        center[ch], amplitude_scale[ch] = robust_stats(fit[:, ch])
        jump_center[ch], jump_scale[ch] = robust_stats(np.diff(fit[:, ch])[adjacent])
        channel_mask[ch] = amplitude_scale[ch] > 0 and np.isfinite(fit[:, ch]).any()
    if not channel_mask.any():
        raise ValueError('No nonconstant training channels')
    amplitude_limit = config.artifact_z*amplitude_scale
    jump_limit = config.artifact_z*jump_scale
    finite = np.isfinite(data[:, :, channel_mask]).all(axis=(1, 2))
    amplitude = (np.abs(data[:, :, channel_mask]-center[channel_mask]) > amplitude_limit[channel_mask]).any(axis=(1, 2))
    jump = (np.abs(np.diff(data[:, :, channel_mask], axis=1)-jump_center[channel_mask]) > jump_limit[channel_mask]).any(axis=(1, 2))
    accepted = finite & ~amplitude & ~jump
    candidate_z = np.array(sorted(set([10.0, 15.0, 20.0, config.artifact_z])))
    candidate_counts = []
    for z in candidate_z:
        amp_bad = (np.abs(data[train][:, :, channel_mask]-center[channel_mask]) > z*amplitude_scale[channel_mask]).any(axis=(1, 2))
        jump_bad = (np.abs(np.diff(data[train][:, :, channel_mask], axis=1)-jump_center[channel_mask]) > z*jump_scale[channel_mask]).any(axis=(1, 2))
        candidate_counts.append(int((~finite[train] | amp_bad | jump_bad).sum()))
    for i, trial in enumerate(trials):
        if not finite[i]:
            trial['reasons'].append('nonfinite_lfp')
        if amplitude[i]:
            trial['reasons'].append('amplitude_artifact')
        if jump[i]:
            trial['reasons'].append('jump_artifact')
    accepted_train = np.array([i for i in train if accepted[i]], dtype=int)
    scale_support, scale_first = unique_support(trials, accepted_train)
    scale_fit = data[accepted_train].reshape(-1, c)[scale_first].astype(np.float64)
    mean, sd = np.zeros(c), np.ones(c)
    mean[channel_mask] = scale_fit[:, channel_mask].mean(axis=0)
    sd[channel_mask] = scale_fit[:, channel_mask].std(axis=0)
    # A channel can become constant after rejected trials are removed.
    channel_mask &= sd > 0
    if not channel_mask.any():
        raise ValueError('No varying channels after training quality exclusions')
    sd[~channel_mask] = 1
    scaled = ((data[accepted][:, :, channel_mask]-mean[channel_mask])/sd[channel_mask]).astype(np.float32)
    stats = dict(artifact_z_candidates=candidate_z, artifact_candidate_train_rejections=np.array(candidate_counts),
                 threshold_fit_finite_counts=np.isfinite(fit).sum(axis=0), mean=mean, sd=sd, channel_mask=channel_mask, robust_center=center,
                 amplitude_scale=amplitude_scale, amplitude_limit=amplitude_limit,
                 jump_center=jump_center, jump_scale=jump_scale, jump_limit=jump_limit,
                 threshold_fit_raw_indices=support, scaling_fit_raw_indices=scale_support,
                 threshold_fit_trial_ids=np.array([trials[i]['trial_id'] for i in train]),
                 scaling_fit_trial_ids=np.array([trials[i]['trial_id'] for i in accepted_train]))
    return accepted, scaled, stats


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                         for k, v in r.items()} for r in records)


def counts(trials):
    return {key: dict(sorted(Counter(str(t[key]) for t in trials).items()))
            for key in ('odor', 'position', 'inseq', 'correct', 'split')}


def coverage(trials, config):
    result = {}
    for split in ('train', 'validation', 'test'):
        subset = [t for t in trials if t['split'] == split]
        result[split] = {str(ms): sum(t['sampling_duration_ms'] is not None and
                                      t['sampling_duration_ms'] >= ms for t in subset)
                         for ms in (300, 500)}
        result[split]['all_onsets'] = len(subset)
    return result


def diagnostics(output, raw, scaled, kept, channel_keys, stats, config, plots):
    """PSD uses independent retained TRAIN trials; no artificial joins."""
    from scipy.signal import welch
    train = [i for i, t in enumerate(kept) if t['split'] == 'train']
    train_raw = raw[train]
    freq, psd = welch(train_raw, fs=config.sample_rate_hz,
                      nperseg=min(500, train_raw.shape[1]), axis=1)
    power = psd.mean(axis=0)
    band = (freq >= 58) & (freq <= 62)
    sides = ((freq >= 52) & (freq <= 56)) | ((freq >= 64) & (freq <= 68))
    diagnostic_available = bool(band.any() and sides.any())
    ratio = (power[band].mean(axis=0)/np.maximum(power[sides].mean(axis=0), np.finfo(float).tiny)
             if diagnostic_available else None)
    np.savez_compressed(output/'training_psd.npz', frequency_hz=freq, power=power,
                        channel_keys=channel_keys)
    summary = [dict(channel=str(key), training_min=float(train_raw[:, :, ch].min()),
                    training_max=float(train_raw[:, :, ch].max()),
                    training_sd=float(train_raw[:, :, ch].std()),
                    line_60hz_to_neighbor_power=float(ratio[ch]) if ratio is not None else None,
                    line_noise_status='available' if diagnostic_available else 'insufficient frequency resolution') for ch, key in enumerate(channel_keys)]
    _json(output/'channel_quality.json', summary)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
        time = np.arange(train_raw.shape[1])/config.sample_rate_hz
        for ch in range(min(3, len(channel_keys))):
            axes[0].plot(time, train_raw[0, :, ch], label=str(channel_keys[ch]), lw=.7)
            axes[1].plot(time, scaled[train[0], :, ch], lw=.7)
        axes[0].set(title='First retained training trial: supplied voltage', ylabel='Stored units')
        axes[0].legend()
        axes[1].set(title='Same samples: frozen training scaling', ylabel='Z score', xlabel='Seconds after onset')
        axes[2].semilogy(freq[1:], power[1:])
        axes[2].set(title='Mean PSD of independent retained training clips', xlabel='Hz', ylabel='Stored units²/Hz')
        fig.savefig(output/'quality.png', dpi=140)
        plt.close(fig)
    return summary


def process_session(raw_root, session, output, config, plots=True):
    output.mkdir()
    source = raw_root/session
    bvr, lfp = source/f'{session}_bvr.npz', source/f'{session}_lfp.npz'
    time, trials, segments, anomalies, bvr_meta, behavior_events = load_behavior(bvr, config.sample_rate_hz)
    lfp_meta = metadata(lfp)
    if lfp_meta['shape'][1] != len(time):
        raise ValueError('LFP and behavior sample counts differ')
    intervals = partition_trials(trials, segments, len(time), config)
    for interval in intervals:
        interval['start_s'] = float(time[interval['start']])
        interval['end_exclusive_s'] = float(time[interval['end']-1]+1/config.sample_rate_hz)
    coverage_summary = coverage(trials, config)
    candidates = [t for t in trials if not t['reasons']]
    if not candidates:
        raise ValueError(f'No eligible windows for {session}')
    n = config.validate()
    data = np.empty((len(candidates), n, lfp_meta['shape'][0]), dtype=np.float64)
    for ch, row in enumerate(rows(lfp)):
        for i, trial in enumerate(candidates):
            data[i, :, ch] = row[trial['start']:trial['end']]
    accepted, scaled, stats = quality_and_scale(data, candidates, config)
    kept = [t for i, t in enumerate(candidates) if accepted[i]]
    for t in trials:
        t['retained'] = not bool(t['reasons'])
    mask = np.ones(scaled.shape[:2], dtype=bool)
    centers = eligible_centers(mask, config.left_context, config.right_context, 0)
    pair_centers = eligible_centers(mask, config.left_context, config.right_context, config.positive_offset)
    raw_indices = np.stack([np.arange(t['start'], t['end']) for t in kept])
    trial_ids = np.array([t['trial_id'] for t in kept], dtype=np.int64)
    splits = np.array([t['split'] for t in kept])
    keys = np.array(lfp_meta['keys'])[stats['channel_mask']]
    np.savez_compressed(output/'odor_trials.npz', lfp=scaled, sample_valid=mask,
                        valid_centers=centers, valid_pair_centers=pair_centers,
                        raw_indices=raw_indices, timestamps_s=time[raw_indices],
                        trial_ids=trial_ids, split=splits, channel_keys=keys,
                        sample_rate_hz=config.sample_rate_hz)
    np.savez_compressed(output/'labels.npz', trial_ids=trial_ids,
                        **{key: np.array([t[key] for t in kept])
                           for key in ('odor', 'position', 'inseq', 'correct')})
    train = splits == 'train'
    # No behavior labels, splits, status or source files enter this artifact.
    np.savez_compressed(output/'cebra_train_sequences.npz', lfp=scaled[train],
                        sample_valid=mask[train], valid_centers=centers[train],
                        valid_pair_centers=pair_centers[train], trial_ids=trial_ids[train],
                        raw_indices=raw_indices[train], timestamps_s=time[raw_indices[train]],
                        channel_keys=keys, sample_rate_hz=config.sample_rate_hz,
                        lengths=np.full(train.sum(), n), offsets=np.arange(train.sum())*n,
                        left_context=config.left_context, right_context=config.right_context,
                        positive_offset=config.positive_offset,
                        context_contract_selected=bool(config.left_context or config.right_context),
                        pair_contract_status='illustrative; modeling adapter and encoder must be verified')
    np.savez_compressed(output/'scaling.npz', **stats)
    _json(output/'raw_behavior_events.json', {key: [dict(raw_index=index, time_s=float(time[index]), value=value) for index, value in sorted(values.items())] for key, values in behavior_events.items()})
    _csv(output/'trial_manifest.csv', trials)
    _csv(output/'split_manifest.csv', intervals)
    quality = diagnostics(output, data[accepted][:, :, stats['channel_mask']], scaled,
                          kept, keys, stats, config, plots)
    exclusions = Counter(reason for t in trials for reason in t['reasons'])
    summary = dict(session=session, raw_onsets=len(trials), retained=len(kept),
                   coverage_ms=coverage_summary, before=counts(trials), after=counts(kept),
                   by_split={split: counts([t for t in kept if t['split'] == split])
                             for split in ('train', 'validation', 'test')},
                   exclusions=dict(exclusions), poke_anomalies=anomalies,
                   channels_total=len(lfp_meta['keys']), channels_kept=int(stats['channel_mask'].sum()),
                   dropped_channel_keys=np.array(lfp_meta['keys'])[~stats['channel_mask']].tolist(),
                   dropped_channel_reasons={key: ('no_finite_training_samples' if stats['threshold_fit_finite_counts'][ch] == 0 else 'constant_training_channel')
                                            for ch, key in enumerate(lfp_meta['keys']) if not stats['channel_mask'][ch]},
                   artifact_candidate_train_rejections=dict(zip(stats['artifact_z_candidates'].astype(str).tolist(), stats['artifact_candidate_train_rejections'].tolist())),
                   withdrawal_coverage_by_status={str(ms): counts([t for t in trials if t['sampling_duration_ms'] is not None and t['sampling_duration_ms'] >= ms]) for ms in (300,500)},
                   threshold_unique_samples=len(stats['threshold_fit_raw_indices']),
                   scaling_unique_samples=len(stats['scaling_fit_raw_indices']),
                   segments=segments, timestamp_gaps=[dict(after_raw_index=b-1, from_s=float(time[b-1]), to_s=float(time[b]), jump_s=float(time[b]-time[b-1])) for _, b in segments[:-1]], split_intervals=intervals,
                   sources=dict(lfp=lfp_meta, behavior=bvr_meta), config=asdict(config),
                   line_noise_diagnostic='58-62 Hz mean power / mean 52-56 and 64-68 Hz power; no notch',
                   channel_quality=quality)
    _json(output/'preprocessing_config.json', dict(schema_version=1, **asdict(config),
        window_policy='fixed [odor onset, onset+window); exclude early withdrawals',
        split_policy='contiguous raw sample index blocks; purge crossing windows; no verified sequence groups',
        quality_policy='unique candidate TRAIN samples; median/MAD (SD fallback); reject whole trial on any retained-channel nonfinite/amplitude/jump violation',
        scaling_policy='unique retained TRAIN samples, population mean/SD',
        signal_processing='no filtering, rereferencing, resampling or imputation',
        context_contract_selected=bool(config.left_context or config.right_context),
        odor_offset='unknown', units='unknown supplied voltage units', acquisition_filter='unknown',
        acquisition_reference='unknown'))
    _json(output/'summary.json', summary)
    (output/'preprocessing_qa.md').write_text(
        f'# {session} preprocessing QA\n\n'
        f'{len(trials)} chronological raw odor onsets; {len(kept)} retained. '
        f'{stats["channel_mask"].sum()}/{len(lfp_meta["keys"])} channels retained.\n\n'
        f'Exclusion reasons (may overlap): `{dict(exclusions)}`.\n\n'
        f'300/500 ms withdrawal coverage by raw assigned split: `{coverage_summary}`.\n\n'
        f'Scaling uses {len(stats["scaling_fit_raw_indices"])} unique raw training samples. '
        'See scaling.npz for exact support and fit trial IDs.\n\n'
        'All accepted windows lie within one recording segment, one split, and one verified poke. '
        'No filtering or new temporal mixing was applied. Context/pairs remain within each trial.\n\n'
        'summary.json gives class/status/position/split counts, full source provenance and channel diagnostics. '
        'training_psd.npz is computed independently per retained training clip; quality.png shows before/after scaling.\n\n'
        'Odor offset, physical units, acquisition reference/filtering remain unknown. '
        'The window is a sampling proxy. Fixed-window exclusions can alter behavior/class balance. '
        'Residual 60 Hz peaks are preserved; consult channel_quality.json and the PSD before selecting a shared optional notch branch. '
        'Zero context is a placeholder until an encoder is selected. This export is not a fitted CEBRA model or adapter.\n')
    return summary


def validate_output(raw_root, output):
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if not raw_root.is_dir():
        raise ValueError('Input raw directory does not exist')
    # Also protect sibling epoched_data and ancestors; allow data/processed only.
    protected = [raw_root, raw_root.parent/'epoched_data']
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError('Output overlaps a protected source tree')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new run directory: {output}')
    return raw_root, output


def publish_exclusive(staging, output):
    """Atomically publish without replacing even a concurrently created directory.

    Fail closed on platforms without a supported exclusive rename primitive.
    macOS RENAME_EXCL is defined in sys/stdio.h; Linux uses RENAME_NOREPLACE.
    """
    if sys.platform == 'win32':
        os.rename(staging, output)  # Windows rename refuses an existing destination.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staging), os.fsencode(output)
    if sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source, target, 0x00000004)
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, source, -100, target, 1)
    else:
        raise RuntimeError('Atomic exclusive directory publication is unsupported on this platform')
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def run(raw_root, output, sessions=None, config=None, plots=True):
    config = config or Config()
    config.validate()
    raw_root, output = validate_output(raw_root, output)
    sessions = sessions or sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    if not sessions or len(set(sessions)) != len(sessions):
        raise ValueError('Provide unique session names')
    for name in sessions:
        if name in ('.', '..') or Path(name).name != name or not (raw_root/name).is_dir():
            raise ValueError(f'Invalid session: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.staging-', dir=output.parent))
    try:
        summaries = []
        for session in sessions:
            summary = process_session(raw_root, session, staging/session, config, plots)
            summaries.append(summary)
            print(f'{session}: {summary["retained"]}/{summary["raw_onsets"]} trials; exclusions={summary["exclusions"]}', flush=True)
        import scipy
        _json(staging/'run_manifest.json', dict(schema_version=1, raw_root=str(raw_root),
            config=asdict(config), sessions=sessions, python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__, summaries=summaries))
        (staging/'preprocessing_qa.md').write_text('# Raw LFP preprocessing run\n\n' + '\n'.join(
            f'- {s["session"]}: {s["retained"]}/{s["raw_onsets"]} trials, '
            f'{s["channels_kept"]}/{s["channels_total"]} channels; '
            f'exclusions: {s["exclusions"]}' for s in summaries) + '\n\nSee each session for manifests, frozen training statistics, PSD and trace QA.\n')
        # Recheck so an existing directory is never intentionally replaced.
        if output.exists():
            raise FileExistsError(f'Output appeared during processing: {output}')
        publish_exclusive(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summaries
