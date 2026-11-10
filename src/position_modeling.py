"""Position decoding with the team's shared notch-filtered InSeq and OutSeq exports."""
from pathlib import Path
import json
import hashlib
import platform
import importlib.metadata
from dataclasses import asdict
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from threadpoolctl import threadpool_limits
import torch

from position_pipeline import Config, train_cebra
from position_decoders import DecoderConfig, select_decoder, coefficient_table

SNAPSHOT = '987145627a25fe4e7ddc0f5f29b04dc3119420e7'
SPLITS = ('train', 'validation', 'test')


def load_trials(folder):
    """Validate shared exports, align labels, and purge groups spanning splits."""
    policy = json.loads((folder / 'preprocessing_config.json').read_text())
    if (policy['profile'], policy['sample_rate_hz'], policy['window_ms'], policy['notch_hz']) != (
        'inseq-decoding', 1000, 500, 60
    ):
        raise ValueError(f'{folder.name}: expected the shared 60 Hz notch / 500 ms export')
    if not policy.get('include_outseq', False):
        raise ValueError('Regenerate exports with --include-outseq; old InSeq-only arrays are not accepted')
    with np.load(folder / 'position_decoding.npz', allow_pickle=False) as data:
        x, y, ids, split = (data[key] for key in ('X', 'y', 'trial_ids', 'split'))
        if str(data['target_name']) != 'position':
            raise ValueError('Expected position targets, not odor targets')
        raw = data['raw_indices']
        source_start, source_end = data['filter_source_start'], data['filter_source_end']
        if not data['sample_valid'].all() or float(data['sample_rate_hz']) != 1000:
            raise ValueError('Invalid samples or unexpected sampling rate')
    manifest = pd.read_csv(folder / 'trial_manifest.csv').sort_values('onset')
    manifest['group'] = manifest.position.eq(0).cumsum()
    if not manifest.trial_id.is_unique or len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate trial IDs')
    matched = manifest.set_index('trial_id').loc[ids]
    if not (
        x.ndim == 3 and x.shape[0] == len(ids) and x.shape[1] == 500
        and raw.shape == x.shape[:2] and np.isfinite(x).all()
        and matched.retained.all() and matched.inseq.isin([0, 1]).all()
        and np.array_equal(matched.position, y) and np.array_equal(matched.split, split)
        and np.array_equal(raw, matched.onset.to_numpy()[:, None] + np.arange(500))
        and set(split) == set(SPLITS)
    ):
        raise ValueError(f'{folder.name}: inconsistent exported trial data')
    # Half-open raw filter supports must be ordered and disjoint across splits.
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        if source_end[split == earlier].max() > source_start[split == later].min():
            raise ValueError(f'{folder.name}: filtering support crosses a split')
    with np.load(folder / 'labels.npz', allow_pickle=False) as labels:
        if not (np.array_equal(labels['trial_ids'], ids)
                and np.array_equal(labels['position'], y)
                and np.array_equal(labels['odor'], matched.odor)
                and np.array_equal(labels['inseq'], matched.inseq)):
            raise ValueError('Labels are not aligned with position export')
    groups = manifest.dropna(subset=['split']).groupby('group').split.nunique()
    keep = ~matched.group.isin(groups[groups > 1].index).to_numpy()
    return x[keep], y[keep], ids[keep], split[keep], int((~keep).sum()), matched.iloc[np.flatnonzero(keep)].reset_index()


def pool50(x):
    """Average each 50 ms bin; flatten in time-bin then channel order."""
    if x.ndim != 3 or x.shape[1] != 500:
        raise ValueError("Expected trials x 500 samples x channels")
    return x.reshape(len(x), 10, 50, x.shape[-1]).mean(axis=2).reshape(len(x), -1)


def encode(model, x, device):
    """Transform each trial independently, with edge padding inside that trial."""
    offset = model.get_offset()
    chunks = []
    with torch.inference_mode():
        for part in np.array_split(x, max(1, (len(x) + 31) // 32)):
            signal = torch.as_tensor(part, dtype=torch.float32, device=device).transpose(1, 2)
            # Align outputs to each of the 500 time samples; pad within each trial only.
            signal = torch.nn.functional.pad(signal, (offset.left, offset.right - 1), mode='replicate')
            chunks.append(model(signal).transpose(1, 2).cpu().numpy())
    z = np.concatenate(chunks)
    if z.shape[:2] != x.shape[:2] or not np.isfinite(z).all():
        raise ValueError('Invalid latent time alignment or values')
    return z


def run(raw_root, output, decoder_config=DecoderConfig()):
    """Run fixed five-rat protocol, preserving provenance and fitted models."""
    raw_root, output = Path(raw_root), Path(output)
    folders = sorted(p.parent for p in raw_root.glob('*/position_decoding.npz'))
    if len(folders) != 5:
        raise ValueError(f'Expected five rats in {raw_root}; see notebook setup')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_config.json').write_text(json.dumps(dict(
        preprocessing_commit=SNAPSHOT, input_root=str(raw_root.resolve()),
        target='position', include_outseq=True, window_ms=500,
        decoder=asdict(decoder_config), penalty='l1', solver='saga',
        python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','scipy','torch','cebra')},
        note='Training-only encoder and decoders; subgroup metrics descriptive, missing classes reported.'), indent=2))
    source_root = Path(__file__).parent
    provenance = {'inputs': {}, 'sources': {}}
    for source in ('position_modeling.py', 'position_decoders.py', 'position_pipeline.py'):
        content = (source_root / source).read_bytes()
        (output / source).write_bytes(content)
        provenance['sources'][source] = hashlib.sha256(content).hexdigest()
    for folder in folders:
        for name in ('position_decoding.npz','labels.npz','trial_manifest.csv','preprocessing_config.json'):
            provenance['inputs'][folder.name + '/' + name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = Config(positive_offset_ms=10, seed=decoder_config.seed)  # Keep temporal pairs within real trials.
    rows, predictions, subgroup_rows, audit_rows = [], [], [], []
    for folder in folders:
        x, y, ids, split, dropped, labels = load_trials(folder)
        for name in SPLITS:
            part = labels.loc[split == name]
            audit_rows.append(dict(rat=folder.name, split=name, n=len(part),
                                   inseq=int(part.inseq.eq(1).sum()), outseq=int(part.inseq.eq(0).sum())))
        if not labels.inseq.eq(0).any():
            raise ValueError(f'{folder.name}: no OutSeq trials survived; inspect preprocessing QA')
        tr, test = split == 'train', split == 'test'
        if any(set(y[split == name]) != set(range(5)) for name in SPLITS):
            raise ValueError(f'{folder.name}: class missing from a split')
        train = x[tr]
        runs = [(i * 500, (i + 1) * 500) for i in range(len(train))]
        print(f'{folder.name}: {len(train)} train, {test.sum()} test, {dropped} crossing-group trials dropped', flush=True)
        model, loss, sampler = train_cebra(train.reshape(-1, x.shape[-1]), runs, cfg, device)
        torch.save(model.state_dict(), output / f'{folder.name}_encoder_state.pt')
        (output / 'encoder_config.json').write_text(json.dumps(asdict(cfg), indent=2))
        labels.assign(split=split).to_csv(output / f'{folder.name}_retained_trials.csv', index=False)
        features = {'raw': pool50(x), 'latent': pool50(encode(model, x, device))}
        with threadpool_limits(limits=2):
            for method, (name, values) in product(('logistic', 'knn'), features.items()):
                score, decoder, setting, search = select_decoder(values, y, split, method, decoder_config)
                prefix = f"{folder.name}_{method}_{name}"
                search.to_csv(output / f"{prefix}_validation_search.csv", index=False)
                joblib.dump(decoder, output / f"{prefix}_decoder.joblib")
                if method == "logistic":
                    with np.load(folder / "position_decoding.npz", allow_pickle=False) as data:
                        channels = data["channel_keys"].tolist()
                    axes = channels if name == "raw" else [f"latent_{i+1}" for i in range(8)]
                    feature_names = [f"{t*50}-{(t+1)*50}ms:{axis}" for t in range(10) for axis in axes]
                    coefficient_table(decoder, feature_names).to_csv(output / f"{prefix}_coefficients.csv", index=False)
                pred = decoder.predict(values[test])
                rows.append(dict(rat=folder.name, decoder=method, representation=name,
                                 test_n=int(test.sum()), crossing_dropped=dropped, setting=setting,
                                 validation_balanced_accuracy=score,
                                 test_balanced_accuracy=recall_score(y[test], pred, labels=range(5),
                                                                      average='macro', zero_division=0)))
                test_labels = labels.loc[test].reset_index(drop=True)
                for trial_type, subset in (('all', np.ones(len(pred), dtype=bool)),
                                           ('inseq', test_labels.inseq.eq(1).to_numpy()),
                                           ('outseq', test_labels.inseq.eq(0).to_numpy())):
                    target = y[test][subset]
                    guess = pred[subset]
                    odor = test_labels.odor.to_numpy()[subset]
                    subgroup_rows.append(dict(
                        rat=folder.name, decoder=method, representation=name, trial_type=trial_type,
                        n=int(subset.sum()), classes_present=int(len(np.unique(target))),
                        position_accuracy=float(np.mean(guess == target)) if len(target) else np.nan,
                        position_balanced_accuracy_present_classes=float(np.mean([
                            np.mean(guess[target == c] == c) for c in np.unique(target)])) if len(target) else np.nan,
                        odor_match_rate=float(np.mean(guess == odor)) if len(target) else np.nan))
                predictions.extend(dict(rat=folder.name, decoder=method, representation=name,
                                        trial_id=i, position=t, prediction=p, odor=o, inseq=int(q))
                                   for i, t, p, o, q in zip(ids[test], y[test], pred,
                                                           test_labels.odor, test_labels.inseq))
                np.savetxt(output / f'{folder.name}_{method}_{name}_confusion.csv',
                           confusion_matrix(y[test], pred, labels=range(5)), fmt='%d', delimiter=',')
        pd.DataFrame(subgroup_rows).to_csv(output / 'subgroup_metrics.csv', index=False)
        pd.DataFrame(audit_rows).to_csv(output / 'trial_counts.csv', index=False)
        pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
        pd.DataFrame({'loss': loss}).to_csv(output / f'{folder.name}_cebra_loss.csv', index=False)
        (output / f'{folder.name}_sampler.json').write_text(json.dumps(sampler, indent=2))
    (output / 'COMPLETE').write_text('All five rats completed successfully.\n')
    return pd.DataFrame(rows)
