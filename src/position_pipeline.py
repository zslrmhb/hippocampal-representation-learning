"""Shared raw-LFP preprocessing and boundary-safe CEBRA position experiment.

All labels are zero-based internally. No behavioral labels enter CEBRA training.
Uses CEBRA's encoder and InfoNCE implementation with an explicit temporal sampler.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import zipfile
import random
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from numpy.lib import format as npformat
import torch
import cebra
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class Config:
    fs: int = 1000
    window_ms: int = 500
    artifact_z: float = 20.
    train_fraction: float = .70
    validation_fraction: float = .15
    architecture: str = 'offset10-model'
    positive_offset_ms: int = 100
    latent_dim: int = 8
    hidden_units: int = 32
    iterations: int = 1000
    batch_size: int = 512
    learning_rate: float = 3e-4
    seed: int = 42
    n_null: int = 20
    c_grid: tuple = (.01, .1, 1.)


def npz_rows(path):
    """Stream C-order channels x samples arrays without loading the full ZIP."""
    with np.load(path, allow_pickle=False) as z:
        keys = z['keys'].tolist()
    with zipfile.ZipFile(path) as z, z.open('data.npy') as f:
        version = npformat.read_magic(f)
        if version == (1, 0):
            shape, order, dtype = npformat.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, order, dtype = npformat.read_array_header_2_0(f)
        else:
            raise ValueError(f'Unsupported NPY version: {version}')
        assert len(shape) == 2 and not order and len(keys) == shape[0]
        assert len(set(keys)) == len(keys) and dtype.kind in 'fiu'
        for key in keys:
            raw = f.read(shape[1] * dtype.itemsize)
            if len(raw) != shape[1] * dtype.itemsize:
                raise ValueError('Truncated NPZ row')
            yield key, np.frombuffer(raw, dtype=dtype)


def reconstruct_trials(path, cfg):
    events = {}
    for key, row in npz_rows(path):
        if key == 'TimeBin':
            time = row.copy()
        elif key in ['PokeEvents', 'InSeqLog', 'PerformanceLog'] or key.startswith(('Odor', 'Position')):
            assert np.isfinite(row).all()
            ids = np.flatnonzero(row)
            events[key] = dict(zip(ids.tolist(), row[ids].tolist()))
    dt = np.diff(time)
    assert np.isfinite(time).all() and (dt > 0).all()
    regular = dt <= 1.5 / cfg.fs
    assert np.allclose(dt[regular], 1 / cfg.fs, rtol=1e-4, atol=1e-8)
    breaks = np.flatnonzero(~regular) + 1
    segment = np.searchsorted(breaks, np.arange(len(time)), side='right')
    onsets = sorted(set().union(*(events[f'Odor{i}'] for i in range(1, 6))))
    brackets, active, bad = [], None, False
    for k, value in sorted(events['PokeEvents'].items()):
        if active is not None and segment[k] != segment[active]:
            active, bad = None, False
        if value == 1:
            if active is None:
                active, bad = k, False
            else:
                bad = True
        elif value == -1:
            if active is not None:
                brackets.append((active, k, bad))
            active, bad = None, False
        elif active is not None:
            bad = True
    records = []
    for trial_id, onset in enumerate(onsets):
        reasons = []
        od = [i-1 for i in range(1, 6) if events[f'Odor{i}'].get(onset) == 1]
        po = [i-1 for i in range(1, 6) if events[f'Position{i}'].get(onset) == 1]
        if len(od) != 1 or len(po) != 1:
            raise ValueError('Ambiguous labels: resolve before assigning sequence IDs')
        if any(onset+d in events[f'Odor{od[0]+1}'] for d in [-1, 1]):
            reasons.append('odor_not_single_sample_pulse')
        ins, correct = (events[k].get(onset, 0) for k in ['InSeqLog', 'PerformanceLog'])
        if ins not in [-1, 1] or correct not in [-1, 1]:
            reasons.append('invalid_status')
        assert (ins == 1) == (od[0] == po[0])
        matches = [(a, b, bad) for a, b, bad in brackets if a <= onset < b]
        entry, exit_ = onset, onset
        if len(matches) != 1:
            reasons.append('missing_poke_bracket')
        else:
            entry, exit_, bad = matches[0]
            if bad or sum(entry <= k < exit_ for k in onsets) != 1:
                reasons.append('ambiguous_poke_bracket')
        records.append(dict(trial_id=trial_id, onset=onset, odor=od[0], position=po[0],
                            inseq=ins == 1, correct=correct == 1, poke_in=entry, poke_out=exit_,
                            sampling_duration_ms=(exit_-onset)*1000/cfg.fs,
                            segment=int(segment[onset]), reasons=';'.join(reasons)))
    df = pd.DataFrame(records)
    # Conservative groups start only at a recorded position-1 event. Missing
    # starts merge sequences instead of splitting a possibly dependent stretch.
    # Keep the initial partial sequence in one group as well.
    starts = df.position.eq(0)
    starts.iloc[0] = True
    df['sequence_id'] = starts.cumsum().astype(int)-1
    df['sequence_group_irregular'] = False
    for _, g in df.groupby('sequence_id'):
        regular_group = np.array_equal(g.position, np.arange(len(g))) and len(g) <= 5 and g.segment.nunique() == 1
        df.loc[g.index, 'sequence_group_irregular'] = not regular_group
    nseq = df.sequence_id.nunique()
    ntrain = int(nseq*cfg.train_fraction)
    nval = max(1, int(nseq*cfg.validation_fraction))
    assert ntrain > 0 and ntrain+nval < nseq
    df['split'] = np.select([df.sequence_id < ntrain, df.sequence_id < ntrain+nval],
                            ['train', 'validation'], default='test')
    boundaries = [0, int(df.loc[df.sequence_id.eq(ntrain), 'poke_in'].iloc[0]),
                  int(df.loc[df.sequence_id.eq(ntrain+nval), 'poke_in'].iloc[0]), len(time)]
    win = round(cfg.fs*cfg.window_ms/1000)
    for i, row in df.iterrows():
        end = row.onset+win
        why = row.reasons.split(';') if row.reasons else []
        owner = ['train', 'validation', 'test'].index(row.split)
        if end > boundaries[owner+1]: why.append('split_boundary')
        if end > len(time) or segment[min(end-1, len(time)-1)] != row.segment: why.append('recording_boundary')
        if end > row.poke_out: why.append('early_withdrawal')
        df.at[i, 'reasons'] = ';'.join(why)
    assert df.groupby('sequence_id').split.nunique().max() == 1
    return df, time, breaks, boundaries


def robust(x):
    x = x[np.isfinite(x)]
    if len(x) == 0: return 0., 0.
    med = np.median(x)
    return med, (1.4826*np.median(np.abs(x-med)) or np.std(x))


def preprocess(data_root, session, out, cfg):
    out.mkdir(parents=True, exist_ok=True)
    df, time, breaks, boundaries = reconstruct_trials(data_root/session/f'{session}_bvr.npz', cfg)
    n = len(time)
    win = round(cfg.fs*cfg.window_ms/1000)
    candidates = df[df.reasons.eq('')].copy()
    ids = candidates.index.to_numpy()
    indices = candidates.onset.to_numpy()[:, None]+np.arange(win)
    train = candidates.split.eq('train').to_numpy()
    support = np.unique(indices[train].ravel())
    assert len(support)
    # Temporary disk-backed array avoids keeping full float64 recordings in memory.
    channel_names, stats, masks, trials = [], [], [], []
    cache = None
    for ch, (key, row) in enumerate(npz_rows(data_root/session/f'{session}_lfp.npz')):
        if cache is None:
            with np.load(data_root/session/f'{session}_lfp.npz', allow_pickle=False) as z:
                nch = len(z['keys'])
            cache = np.lib.format.open_memmap(out/'recording_cache.npy', mode='w+', dtype='float32', shape=(n, nch))
        assert len(row) == n
        center, amp = robust(row[support])
        adjacent = np.diff(support) == 1
        # Do not include derivatives across actual timestamp gaps.
        adjacent &= np.isclose(np.diff(time[support]), 1/cfg.fs, rtol=1e-4, atol=1e-8)
        jc, js = robust(np.diff(row[support])[adjacent])
        if amp <= 0 or js <= 0:
            continue
        window = row[indices].copy()
        bad = (~np.isfinite(window)).any(1)
        bad_amp = (np.abs(window-center) > cfg.artifact_z*amp).any(1)
        bad_jump = (np.abs(np.diff(window, axis=1)-jc) > cfg.artifact_z*js).any(1)
        channel_names.append(key)
        masks.append(np.stack([bad, bad_amp, bad_jump], axis=1))
        trials.append(window)
        stats.append([center, amp, jc, js])
        cache[:, len(channel_names)-1] = row
    assert channel_names
    masks = np.stack(masks).any(0)
    accepted = ~masks.any(1)
    for j, reason in enumerate(['nonfinite_lfp', 'amplitude_artifact', 'jump_artifact']):
        for i in ids[masks[:, j]]:
            df.at[i, 'reasons'] = ';'.join(filter(None, [df.at[i, 'reasons'], reason]))
    kept = candidates.loc[ids[accepted]].copy()
    assert all((kept.split == s).any() for s in ['train', 'validation', 'test'])
    assert set(kept.loc[kept.split.eq('train'), 'position']) == set(range(5))
    X = np.stack(trials, axis=-1)[accepted]
    tr = kept.split.eq('train').to_numpy()
    train_indices = np.unique(indices[accepted][tr].ravel())
    # All accepted samples are distinct here; require this instead of weighting duplicates.
    assert len(train_indices) == tr.sum()*win
    mu = X[tr].mean((0, 1))
    sd = X[tr].std((0, 1))
    assert np.isfinite(mu).all() and (sd > 0).all()
    X = ((X-mu)/sd).astype('float32')
    df['retained'] = df.reasons.eq('')
    df.to_csv(out/'trial_manifest.csv', index=False)
    pd.crosstab([df.split, df.position], df.retained).to_csv(out/'retention_by_position.csv')
    pd.crosstab([df.split, df.inseq], df.retained).to_csv(out/'retention_by_trial_type.csv')
    pd.crosstab(kept.odor, kept.position).reindex(index=range(5), columns=range(5), fill_value=0).to_csv(out/'odor_position_counts.csv')
    np.savez_compressed(out/'shared_trials.npz', lfp=X, trial_ids=kept.trial_id,
                        sequence_ids=kept.sequence_id, split=kept.split.to_numpy(dtype=str),
                        odor=kept.odor, position=kept.position, inseq=kept.inseq, correct=kept.correct,
                        channel_keys=np.array(channel_names), sample_rate_hz=cfg.fs)
    np.savez(out/'scaling.npz', mean=mu, std=sd, artifact_stats=np.array(stats), fit_raw_indices=train_indices)
    # Additional unlabeled training recording. No validation/test samples included.
    stop = boundaries[1]
    recording = np.asarray(cache[:stop, :len(channel_names)]).copy()
    stats = np.array(stats)
    good = np.isfinite(recording).all(1)
    good &= (np.abs(recording-stats[:, 0]) <= cfg.artifact_z*stats[:, 1]).all(1)
    good[1:] &= (np.abs(np.diff(recording, axis=0)-stats[:, 2]) <= cfg.artifact_z*stats[:, 3]).all(1)
    good[breaks[breaks < stop]] = False
    recording = ((recording-mu)/sd).astype('float32')
    recording[~good] = 0
    cache.flush()
    del cache
    (out/'recording_cache.npy').unlink()
    edges = np.diff(np.r_[False, good, False].astype(int))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return df, kept.reset_index(drop=True), X, recording, runs, boundaries


class SegmentSampler:
    """Uniform valid centers; context and the full positive bridge remain in one run."""
    def __init__(self, runs, left, right, lag, seed):
        assert lag > 0 and left >= 0 and right > 0
        a = np.array([s+left for s, e in runs], dtype=np.int64)
        b = np.array([e-right-lag+1 for s, e in runs], dtype=np.int64)
        valid = b > a
        self.a, self.b = a[valid], b[valid]
        self.total = np.cumsum(self.b-self.a)
        assert len(self.total), 'No segments long enough for encoder context and lag'
        self.rng = np.random.default_rng(seed)
        self.left, self.right, self.lag = left, right, lag

    def draw(self, n):
        flat = self.rng.integers(0, self.total[-1], n)
        group = np.searchsorted(self.total, flat, side='right')
        prev = np.r_[0, self.total[:-1]]
        return self.a[group]+flat-prev[group], group

    def triplet(self, n):
        ref, group = self.draw(n)
        neg, _ = self.draw(n)
        pos = ref+self.lag
        assert (ref >= self.a[group]).all() and (pos < self.b[group]+self.lag).all()
        return ref, pos, neg


def train_cebra(neural, runs, cfg, device):
    """Official CEBRA encoder/loss; custom sampler avoids artificial temporal joins."""
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    model = cebra.models.init(cfg.architecture, num_neurons=neural.shape[1],
                             num_units=cfg.hidden_units, num_output=cfg.latent_dim).to(device)
    offset = model.get_offset()
    lag = round(cfg.positive_offset_ms*cfg.fs/1000)
    sampler = SegmentSampler(runs, offset.left, offset.right, lag, cfg.seed)
    signal = torch.from_numpy(np.ascontiguousarray(neural))
    relative = torch.arange(-offset.left, offset.right)
    criterion = cebra.models.FixedCosineInfoNCE(temperature=1.).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_history = []
    model.train()
    for step in range(cfg.iterations):
        refs, pos, neg = sampler.triplet(cfg.batch_size)
        patches = [signal[torch.from_numpy(i)[:, None]+relative].permute(0, 2, 1).to(device)
                   for i in [refs, pos, neg]]
        embeddings = [model(x) for x in patches]
        loss, _, _ = criterion(*embeddings)
        assert torch.isfinite(loss), 'Nonfinite CEBRA loss'
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        if (step+1) % 250 == 0:
            print(f'  CEBRA {step+1}/{cfg.iterations}, loss {loss_history[-1]:.3f}', flush=True)
    return model.eval(), loss_history, {'valid_pair_centers': int(sampler.total[-1]),
                                       'left_context': offset.left, 'right_context_exclusive': offset.right,
                                       'positive_offset_samples': lag}


def embed_trials(model, X, device):
    result = []
    with torch.no_grad():
        for start in range(0, len(X), 32):
            x = torch.from_numpy(X[start:start+32]).permute(0, 2, 1).to(device)
            # Valid convolutions only. No padding or context from other trials.
            result.append(model(x).mean(-1).cpu().numpy())
    return np.concatenate(result)


def balanced(y, pred):
    """Mean recall over classes present; report n_classes alongside subset results."""
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def classifier(C, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced',
                         solver='lbfgs', max_iter=2000, random_state=seed))


def decode(features, kept, cfg, out, method):
    y = kept.position.to_numpy()
    tr, va, te = [kept.split.eq(s).to_numpy() for s in ['train', 'validation', 'test']]
    tuning = []
    for C in cfg.c_grid:
        model = classifier(C, cfg.seed).fit(features[tr], y[tr])
        tuning.append((balanced(y[va], model.predict(features[va])), C))
    # Stable grid order breaks ties in favor of stronger regularization.
    score, C = max(tuning, key=lambda v: v[0])
    model = classifier(C, cfg.seed).fit(features[tr], y[tr])
    pred = model.predict(features[te])
    test = kept.loc[te].copy()
    test['prediction'] = pred
    test.to_csv(out/f'{method}_predictions.csv', index=False)
    cm = confusion_matrix(y[te], pred, labels=range(5))
    pd.DataFrame(cm, index=range(1, 6), columns=range(1, 6)).to_csv(out/f'{method}_confusion.csv')
    rows = []
    for subset, mask in [('all', np.ones(te.sum(), bool)), ('inseq', test.inseq.to_numpy()),
                         ('outseq', ~test.inseq.to_numpy())]:
        yt, yp = y[te][mask], pred[mask]
        rows.append(dict(method=method, subset=subset, n=len(yt), n_classes=len(np.unique(yt)),
                         accuracy=accuracy_score(yt, yp) if len(yt) else np.nan,
                         balanced_accuracy=balanced(yt, yp) if len(yt) else np.nan,
                         macro_f1=f1_score(yt, yp, labels=range(5), average='macro', zero_division=0) if len(yt) else np.nan,
                         chosen_C=C, validation_balanced_accuracy=score))
    rng = np.random.default_rng(cfg.seed)
    null = []
    for _ in range(cfg.n_null):
        shuffled = rng.permutation(y[tr])
        nm = classifier(C, cfg.seed).fit(features[tr], shuffled)
        null.append(balanced(y[te], nm.predict(features[te])))
    # Descriptive benchmark only: not a sequence-aware permutation significance test.
    pd.DataFrame({'balanced_accuracy': null}).to_csv(out/f'{method}_descriptive_shuffle.csv', index=False)
    rows[0]['shuffle_mean'] = float(np.mean(null)) if null else np.nan
    rows[0]['shuffle_95_percentile'] = float(np.quantile(null, .95)) if null else np.nan
    return rows, cm


def run_session(data_root, session, output_root, cfg):
    out = output_root/session
    out.mkdir(parents=True, exist_ok=False)
    print(f'Processing {session}', flush=True)
    df, kept, X, recording, runs, boundaries = preprocess(data_root, session, out, cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    features = {'raw': X.reshape(len(X), -1)}
    train = kept.split.eq('train').to_numpy()
    training_windows = X[train]
    inputs = {
        'cebra_trial_only': (training_windows.reshape(-1, X.shape[-1]),
                             [(i*X.shape[1], (i+1)*X.shape[1]) for i in range(len(training_windows))]),
        'cebra_recording': (recording, runs),
    }
    losses, metadata = {}, {}
    for name, (signal, bounds) in inputs.items():
        print(f'{session}: {name}, {len(signal):,} samples, {len(bounds)} segments', flush=True)
        model, loss, meta = train_cebra(signal, bounds, cfg, device)
        features[name] = embed_trials(model, X, device)
        torch.save({'state_dict': model.cpu().state_dict(), 'config': asdict(cfg),
                    'input_channels': X.shape[-1], 'sampler': meta}, out/f'{name}.pt')
        losses[name], metadata[name] = loss, meta
        del model
    rows, confusion = [], {}
    for name, feat in features.items():
        print(f'{session}: decoding {name}', flush=True)
        result, cm = decode(feat, kept, cfg, out, name)
        rows.extend(result); confusion[name] = cm
    metrics = pd.DataFrame(rows).assign(session=session)
    metrics.to_csv(out/'metrics.csv', index=False)
    # A behavioral-only benchmark quantifies how well odor alone predicts position.
    mapping = kept[train].groupby('odor').position.agg(lambda x: x.mode().iloc[0]).to_dict()
    test = kept[kept.split.eq('test')]
    odor_pred = test.odor.map(mapping).to_numpy()
    shortcut = balanced(test.position.to_numpy(), odor_pred)
    pd.DataFrame({'position': test.position, 'odor': test.odor, 'odor_only_prediction': odor_pred}).to_csv(out/'odor_only_baseline.csv', index=False)
    np.savez_compressed(out/'features.npz', **features)
    pd.DataFrame(losses).to_csv(out/'cebra_losses.csv', index=False)
    metadata.update(config=asdict(cfg), device=device, boundaries=boundaries,
                    raw_trials=len(df), retained_trials=len(kept), odor_only_test_balanced_accuracy=shortcut,
                    versions={k: importlib.metadata.version(k) for k in ['cebra', 'torch', 'numpy', 'scikit-learn']},
                    python=platform.python_version(), policy='1000 Hz; no filtering/notch/resampling; training-only stats; all correctness types')
    (out/'run_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Finished {session}: {len(kept)}/{len(df)} trials', flush=True)
    return metrics, df.assign(session=session), confusion, losses
