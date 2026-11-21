"""Export all-rat poster interpretation from a frozen validation replay cache."""
import argparse
from pathlib import Path

import numpy as np

from . import replay
from .interpretability import Interpretation
from data_processing.artifacts import ALIASES, digest, job_key, read, require, sha
from data_processing.storage import write_json


def build(cache, checkpoints, spec_path):
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    import torch
    from modeling import design as dec, models

    spec = replay.load_replay_spec(spec_path)
    replay.assert_runtimes(spec['python'], spec['packages'])
    identity = read(cache / 'identity.json')
    require(identity['inputs'] == spec['inputs'], 'Replay inputs differ from the reviewed specification')
    parity = replay.verified_stage(cache, spec, digest(identity), 'val')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = dec.Run(cache)
    output = dict(spec_sha256=sha(spec_path), validation_receipt_sha256=parity,
                  weights=[], latents=[])
    masking = Interpretation()
    with threadpool_limits(1):
        for session in replay.SESSIONS:
            for target in replay.TARGETS:
                key = f'{target}/{session}'
                design = dec.prepare(run, session, target, 500)
                unit = spec['units'][key]
                require(digest(replay.qa_reference(read(run.cache(session, target)/'qa.json'))) == unit['qa_sha256'],
                        f'{key}: cache QA differs')
                cutoff = int(design.frame.split.eq('train').sum())
                train, val = slice(None, cutoff), slice(cutoff, None)
                train_frame, val_frame = design.frame.iloc[train], design.frame.iloc[val]
                for split, frame in [('train', train_frame), ('val', val_frame)]:
                    require(replay.cohort_reference(frame, target) == spec['cohorts'][f'{key}/{split}'], f'{key}: cohort differs')
                require(replay.transform_reference(design._transform_path, train_frame.trial_id.to_numpy()) == unit['transform'],
                        f'{key}: scaler differs')
                name = replay.checkpoint_name(session, target)
                features = dec.matrices(design, checkpoints/name, spec['inputs']['checkpoints'][name], np.arange(cutoff))
                for model, x in features.items():
                    ref = spec['jobs'][job_key(target, model, session, 'val')]
                    require(replay.matrix_sha(x[val]) == ref['feature_sha256'], f'{key}/{model}: validation features differ')
                    head = models.fit_l1(x[train], train_frame[target].to_numpy(), unit['C'][model], 42)
                    pred = head.predict(x[val])
                    require(head.converged_ and replay.prediction_sha(val_frame.trial_id.to_numpy(), val_frame[target].to_numpy(), pred)
                            == ref['prediction_sha256'], f'{key}/{model}: prediction parity failed')
                    rat = ALIASES[session]
                    if model == 'historical_anchor':
                        # Report only within-rat channel summaries, not channel identities.
                        rows = dec.weight_rows(design, 'amp_l1', 'bin50', head)
                        w = np.array([r['weight'] for r in rows]).reshape(5, 10, len(design.channels), dec.N_FREQ)
                        raw = head.named_steps['logisticregression'].coef_.reshape(w.shape)
                        output['weights'].append(dict(rat=rat, target=target, n_channels=len(design.channels),
                            signed=w.mean(axis=2).round(10).tolist(), magnitude=np.abs(w).mean(axis=2).round(10).tolist(),
                            nonzero=np.count_nonzero(raw, axis=2).tolist()))
                        masking._mask(rat, target, design, head, x, val, pred)
                    else:
                        pca = PCA(n_components=2, svd_solver='full').fit(x[train])
                        xy = pca.transform(x[val])
                        groups = []
                        for cohort, select in [('all', np.ones(len(val_frame), dtype=bool)),
                                               ('outseq', val_frame.inseq.eq(0).to_numpy())]:
                            for label in ('odor', 'position'):
                                labels = val_frame[label].to_numpy()
                                for k in np.unique(labels[select]):
                                    points = xy[select & (labels == k)]
                                    groups.append(dict(cohort=cohort, label=label, cls=int(k), n=len(points),
                                                       xy=points.mean(axis=0).round(10).tolist()))
                        output['latents'].append(dict(rat=rat, target=target, n_val=len(val_frame),
                            explained=pca.explained_variance_ratio_.round(10).tolist(), groups=groups))
                print(f'Validated interpretation: {session}/{target}', flush=True)
    output['masking'] = masking.masking
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'checkpoints', 'spec', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if any(out.is_relative_to(p.resolve()) for p in (args.cache, args.checkpoints)) or out == args.spec.resolve():
        parser.error('Output must not overwrite replay inputs')
    write_json(out, build(args.cache, args.checkpoints, args.spec))


if __name__ == '__main__':
    main()
