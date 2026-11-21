"""Aggregate interpretation of the predeclared frozen models, using TRAIN/VAL only."""
from collections import defaultdict
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from data_processing.artifacts import ALIASES, commit, csv_write, read, require, sha
from data_processing.preprocess import FREQUENCIES
from modeling import design as dec
from .metrics import confusion, score_confusion

FILES = ('l1_weights.csv', 'l1_masking.csv', 'cebra_projection.csv')


def verify_interpretation(out, spec, spec_sha, code_sha):
    record = read(out/'interpretability_receipt.json')
    expected = {k: v for k, v in spec['jobs'].items() if k.endswith('/val')}
    require(record['status'] == 'verified' and record['spec_sha256'] == spec_sha
            and record['freeze_sha256'] == spec['freeze_sha256'] and record['validation_jobs'] == expected
            and record['validation_parity_sha256'] == sha(out/'val_parity.json')
            and record['code_sha256'] == code_sha and set(record['files']) == set(FILES),
            'Interpretability provenance differs')
    for name, value in record['files'].items():
        require(sha(out/name) == value, f'Interpretability file differs: {name}')


class Interpretation:
    def __init__(self):
        self.weights = []
        self.masking = []
        self.projection = []

    def add(self, session, target, model, design, head, X, train, val, pred):
        require(design._split == 'dev' and design.frame.iloc[train].split.eq('train').all()
                and design.frame.iloc[val].split.eq('val').all(), 'Interpretation must use only TRAIN/VAL')
        alias = ALIASES[session]
        if model == 'historical_anchor':
            self._mask(alias, target, design, head, X, val, pred)
            if alias == 'Mitt' and target == 'position':
                self._weights(design, head)
        elif alias == 'Mitt' and target == 'position':
            self._project(design.frame.iloc[val], X[train], X[val])

    def _weights(self, design, head):
        groups = defaultdict(list)
        for row in dec.weight_rows(design, 'amp_l1', 'bin50', head):
            if row['class_label'] == 3:
                groups[row['time_bin'], row['freq_index']].append(row['weight'])
        for (bin_index, freq), weights in sorted(groups.items()):
            require(len(weights) == len(design.channels), 'Coefficient support misses electrodes')
            self.weights.append(dict(bin_start_ms=50*bin_index, frequency_hz=float(FREQUENCIES[freq]),
                                     mean_weight=math.fsum(weights)/len(weights),
                                     n_nonzero=sum(w != 0 for w in weights), n_electrodes=len(weights)))

    def _mask(self, alias, target, design, head, X, val, pred):
        # Replace the Design view, not its underlying source/selector; mask_predict
        # reads this already-verified binned VAL matrix and never fits a scaler.
        from dataclasses import replace
        X_val = X[val]
        val_design = replace(design, frame=design.frame.iloc[val], _indices=design._indices[val])
        val_design._matrices['amp', 'bin50'] = X_val
        y = val_design.frame[target].to_numpy()
        intact, detail = score_confusion(confusion(y, pred))
        scaler, estimator = (step for _, step in head.steps)
        for group, (lo, hi) in {**dec.FREQUENCY_MASKS, **dec.TIME_MASKS}.items():
            if group in dec.FREQUENCY_MASKS:
                masked = dec.mask_predict(val_design, 'amp_l1', 'bin50', head, group)
            else:
                select = np.fromiter((lo <= b < hi for _, _, b, _, _ in design._columns('amp_l1', 'bin50')),
                                     dtype=bool, count=X_val.shape[1])
                standardized = scaler.transform(X_val)
                standardized[:, select] = 0
                masked = estimator.predict(standardized)
            score, _ = score_confusion(confusion(y, masked))
            self.masking.append(dict(target=target, session=alias, group=group, intact_ba=intact['ba'],
                                     masked_ba=score['ba'], delta=intact['ba']-score['ba'],
                                     n_classes=sum(r['support'] > 0 for r in detail), n=len(y)))

    def _project(self, frame, train, val):
        pca = PCA(n_components=2, svd_solver='full').fit(train)
        coords = pca.transform(val)
        for cohort, selected in [('all', np.ones(len(frame), dtype=bool)), ('outseq', frame.inseq.eq(0).to_numpy())]:
            for grouping in ('odor', 'position'):
                labels = frame[grouping].to_numpy()
                for label in np.unique(labels[selected]):
                    points = coords[selected & (labels == label)]
                    x, y = points.mean(axis=0)
                    self.projection.append({'cohort': cohort, 'grouping': grouping, 'class': int(label),
                                            'pc1': float(x), 'pc2': float(y), 'n': len(points),
                                            'pc1_variance': float(pca.explained_variance_ratio_[0]),
                                            'pc2_variance': float(pca.explained_variance_ratio_[1])})

    def finish(self, out, spec, spec_sha, val_sha):
        require(len(self.weights) == 120 and len(self.masking) == 50 and len(self.projection) > 10,
                'Incomplete interpretation')
        files = {}
        for name, rows in zip(FILES, (self.weights, self.masking, self.projection), strict=True):
            path = out/name
            csv_write(path, rows)
            files[name] = sha(path)
        commit(out/'interpretability_receipt.json', dict(status='verified', spec_sha256=spec_sha,
               freeze_sha256=spec['freeze_sha256'], validation_parity_sha256=val_sha,
               code_sha256=sha(Path(__file__)), files=files,
               validation_jobs={k: v for k, v in spec['jobs'].items() if k.endswith('/val')}))
