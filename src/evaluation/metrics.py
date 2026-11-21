"""Fixed five-class metrics and the TRAIN-only odor-to-position mapping."""
import math

import numpy as np

from data_processing.artifacts import MODELS, SESSIONS, TARGETS, job_key, require as _need

CLASSES = np.arange(1, 6)

def _integer(values, name):
    a = np.asarray(values)
    _need(a.ndim == 1 and a.dtype.kind in 'iu', f'Non-integral {name}')
    return a


def _classes(values, name):
    a = _integer(values, name)
    _need(np.isin(a, CLASSES).all(), f'{name} outside canonical classes')
    return a


def usual_positions(train):
    """Session-specific odor → usual position, identified by correct TRAIN InSeq only."""
    mask = (np.asarray(train['correct']) == 1) & (np.asarray(train['inseq']) == 1)
    if 'split' in train:
        _need(np.all(np.asarray(train['split']) == 'train'), 'Held-out rows passed to usual-position fit')
    mapping = set(zip(_classes(train['odor'], 'odor')[mask].tolist(), _classes(train['position'], 'position')[mask].tolist()))
    _need(len(mapping) == 5 and len({k for k, _ in mapping}) == len({v for _, v in mapping}) == 5,
          'Correct TRAIN InSeq does not establish a bijection')
    return dict(mapping)


def confusion(y, pred):
    y, pred = _classes(y, 'truth'), _classes(pred, 'prediction')
    _need(y.shape == pred.shape, 'Prediction count mismatch')
    return np.bincount(5 * (y.astype(np.intp, copy=False) - 1) + pred.astype(np.intp, copy=False) - 1,
                       minlength=25).reshape(5, 5)


def _ratio(x, y):
    return float(x / y) if y else None


def _mean(values):
    values = tuple(values)
    return math.fsum(values) / len(values) if values and None not in values else None


def score_confusion(c, *, position=False):
    """Fixed rows=truth and columns=prediction. Null ≠ zero on undefined denominators."""
    c = np.asarray(c)
    _need(c.shape == (5, 5) and c.dtype.kind in 'iu' and (c >= 0).all(), 'Expected nonnegative integer 5×5 counts')
    rows, cols, hits = c.sum(axis=1), c.sum(axis=0), c.diagonal()
    detail = [dict(support=int(n), predicted=int(k), tp=int(h), recall=_ratio(h, n), precision=_ratio(h, k),
                   f1=_ratio(2 * h, n + k)) for n, k, h in zip(rows, cols, hits)]
    n = int(rows.sum())
    accuracy = _ratio(hits.sum(), n)
    dist = np.abs(CLASSES[:, None] - CLASSES)
    return dict(n=n, accuracy=accuracy, ba=_mean(r['recall'] for r in detail),
                observed_macro_recall=_mean(r['recall'] for r in detail if r['support']),
                macro_f1=_mean(r['f1'] for r in detail) if rows.all() else None,
                mae=_ratio((dist*c).sum(), n) if position else None, exact_accuracy=accuracy if position else None,
                within_one_accuracy=_ratio(c[dist <= 1].sum(), n) if position else None), detail


def compare_metrics(out, spec, split):
    headline, outseq = {}, {}
    for t in TARGETS:
        for m in MODELS:
            per_rat = []
            mismatch = dict(n=0, follows_actual=0, follows_usual=0, neither=0)
            for s in SESSIONS:
                with np.load(out/'stages'/job_key(t, m, s, split)/'predictions.npz', allow_pickle=False) as f:
                    y, p = f['y'], f['pred']
                    score, _ = score_confusion(confusion(y, p), position=t == 'position')
                    per_rat.append(score)
                    if t == 'position':
                        u = f['usual']
                        selected = (f['inseq'] == 0) & (u != y)
                        n = int(selected.sum())
                        a, b = int((p[selected] == y[selected]).sum()), int((p[selected] == u[selected]).sum())
                        for k, v in zip(mismatch, (n, a, b, n-a-b), strict=True):
                            mismatch[k] += v
            n_rat_ba = sum(r['ba'] is not None for r in per_rat)
            record = dict(n=sum(r['n'] for r in per_rat), n_rat=len(per_rat), n_rat_ba=n_rat_ba,
                          observed_macro_recall=math.fsum(r['observed_macro_recall'] for r in per_rat)/len(per_rat),
                          ba=math.fsum(r['ba'] for r in per_rat)/len(per_rat) if n_rat_ba == len(per_rat) else None)
            key = f'{t}/{m}/{split}'
            expected = spec['headline'][key]
            _need(all(record[k] == expected[k] or (isinstance(record[k], float) and expected[k] is not None
                        and abs(record[k]-expected[k]) <= 1e-12) for k in record), f'Aggregate metric differs: {key}')
            headline[key] = record
            if t == 'position':
                key = f'{m}/{split}'
                _need(mismatch == spec['outseq'][key], f'OutSeq count vector differs: {key}')
                outseq[key] = mismatch
    return dict(headline=headline, outseq=outseq)
