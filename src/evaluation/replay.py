"""Replay the frozen 500 ms poster comparison on authorized private inputs."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys

import numpy as np

from data_processing import storage
from data_processing.artifacts import (
    CLASSES, MODELS, SESSIONS, TARGETS, commit, digest, job_key, plain, read, require, sha, within,
)

SOURCE = Path(__file__).resolve().parents[1]

THREADS = dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'PYTHONNOUSERSITE'), '1')
THREADS['CUDA_VISIBLE_DEVICES'] = ''

SCHEMA = 'poster-replay-v1'
CODE_SCHEMA = 'poster-replay-code-v5'
INTERPRETATION_CODE = 'evaluation/interpretability.py'
REVIEWED_INTERPRETATION_SHA = '480e8d91c94cac317b387a91ab3d34eb290e4260a42a684beaacc1811e195ff7'
SPLITS = ('train', 'val', 'test')
CODE = (
    f'src/evaluation/replay.py', f'src/{INTERPRETATION_CODE}',
    'src/data_processing/artifacts.py', 'src/data_processing/storage.py', 'pyproject.toml', 'uv.lock',
    'src/modeling/design.py', 'src/modeling/models.py', 'src/evaluation/metrics.py',
    'src/data_processing/preprocess.py',
    *(f'src/data_processing/vendor/{name}' for name in (
        '__init__.py', 'events.py', 'filtering.py', 'io.py', 'quality.py',
    )),
)



def matrix_sha(x):
    h = hashlib.sha256(f'{x.shape}:{x.dtype.str}\n'.encode())
    h.update(memoryview(np.ascontiguousarray(x)).cast('B'))
    return h.hexdigest()

def checkpoint_name(s, t):
    return f'{s}-{t}-0-500-cebra_combined-offset10-model-42/step_1000.pt'


def raw_name(s, kind):
    return f'{s}/{s}_{kind}.npz'


def qa_reference(qa):
    return {k: qa[k] for k in ('channel_keys', 'notch_guard_samples', 'morlet_guard_samples',
                                'amp_feature_order', 'phase_feature_order')}


def transform_reference(path, train_ids):
    """Verify the scaler against the exact ordered training rows."""
    with np.load(path, allow_pickle=False) as f:
        require(set(f.files) == {'mean', 'sd', 'train_id'}, 'Unexpected amplitude scaler members')
        ids = f['train_id']
        require(ids.dtype == train_ids.dtype and np.array_equal(ids, train_ids),
                'Amplitude scaler ordered TRAIN reference differs')
        mean, sd = f['mean'], f['sd']
        require(mean.dtype == sd.dtype == np.float64 and mean.ndim == 1 and mean.shape == sd.shape
                and np.isfinite(mean).all() and np.isfinite(sd).all() and (sd >= 0).all(),
                'Invalid frozen amplitude scaler')
        return {k: matrix_sha(x) for k, x in [('mean', mean), ('sd', sd), ('train_id', ids)]}


def prediction_sha(trial_id, y, pred):
    return digest(list(zip(trial_id.tolist(), y.tolist(), pred.tolist(), strict=True)))


def cohort_reference(frame, target):
    from data_processing.preprocess import META
    fields = {name: frame[name].to_numpy(dtype=str if name == 'split' else
                                         np.float64 if name == 'sampling_duration_ms' else np.int64)
              for name in META}
    return dict(sha256=digest({k: v.tolist() for k, v in fields.items()}),
                n=len(frame), classes=np.unique(fields[target]).tolist())


def validate_spec(spec):
    require(spec['schema'] == SCHEMA and spec['sessions'] == list(SESSIONS) and spec['classes'] == list(CLASSES)
            and spec['encoder_seed'] == spec['head_seed'] == 42 and spec['window_ms'] == 500,
            'Unexpected replay settings or domain')
    require(set(spec['jobs']) == {job_key(t, m, s, split) for t in TARGETS for m in MODELS
                                 for s in SESSIONS for split in ('val', 'test')}, 'Unexpected frozen job roster')
    require(set(spec['cohorts']) == {f'{t}/{s}/{split}' for t in TARGETS for s in SESSIONS for split in SPLITS}
            and set(spec['units']) == {f'{t}/{s}' for t in TARGETS for s in SESSIONS},
            'Incomplete cohort or unit roster')
    require(set(spec['inputs']) == {'raw', 'checkpoints'}
            and set(spec['inputs']['raw']) == {raw_name(s, k) for s in SESSIONS for k in ('lfp', 'bvr')}
            and set(spec['inputs']['checkpoints']) == {checkpoint_name(s, t) for s in SESSIONS for t in TARGETS},
            'Unexpected raw or checkpoint paths')
    for unit in spec['units'].values():
        require(set(unit['C']) == set(MODELS) and all(x in (.001, .01, .1, 1, 10) for x in unit['C'].values()),
                'Invalid frozen C')


def load_replay_spec(path):
    spec = read(path)
    if spec.get('schema') in (CODE_SCHEMA, 'poster-replay-code-v4', 'poster-replay-code-v3', 'poster-replay-code-v2'):
        require(set(spec) == {'schema', 'reviewed_spec_sha256', 'code'}, 'Unexpected replay code revision fields')
        reviewed = within(path.parent, 'evidence/replay-spec.json')
        require(sha(reviewed) == spec['reviewed_spec_sha256'], 'Reviewed replay specification digest differs')
        base = read(reviewed)
        pins = spec['code']
        if spec['schema'] == CODE_SCHEMA:
            require(set(pins) == set(CODE), 'Incomplete or unexpected checkout source pins')
        spec = {**base, 'code': pins}
    validate_spec(spec)
    return spec


def assert_runtimes(python, packages):
    require('.'.join(map(str, sys.version_info[:3])) == python, f'Expected scientific Python {python}')
    for package, version in packages.items():
        require(importlib.metadata.version(package) == version, f'Expected {package}=={version}')


def nonoverlap(a, b):
    require(not (a.is_relative_to(b) or b.is_relative_to(a)), f'Output overlaps protected path: {b}')


def output_boundary(out):
    repo = SOURCE.parent
    results = repo/'results'
    archived = (p for p in results.iterdir() if p.name.startswith(('rat-', 'poster-'))
                and not p.name.startswith('poster-release-')) if results.exists() else ()
    for root in (repo/'.research-archive', repo/'ablation_results', repo/'data',
                 repo/'poster', repo/'notebooks', repo/'src', *archived):
        nonoverlap(out, root)


def checked_paths(spec, roots):
    paths = {kind: {name: within(roots[kind], name) for name in pins} for kind, pins in spec['inputs'].items()}
    missing = [f'{kind}: {name}' for kind, members in paths.items() for name, p in members.items() if not p.is_file()]
    require(not missing, 'Missing private inputs (relative to --raw or --checkpoints):\n' + '\n'.join(missing))
    failures = []
    for kind, members in paths.items():
        for name, path in members.items():
            if sha(path) != spec['inputs'][kind][name]:
                failures.append(f'{kind}: {name}')
    require(not failures, 'input SHA-256 mismatch:\n' + '\n'.join(failures))
    return paths


def inventory(root):
    """Walk without traversing a symlink; plain also rejects hard-linked leaves."""
    for base, dirs, files in os.walk(root):
        for name in (*dirs, *files):
            p = plain(Path(base)/name)
            if name in files:
                require(p.is_file(), f'Not a regular artifact: {p}')
                yield p


def file_bindings(root, files):
    return {p.relative_to(root).as_posix(): sha(p) for p in files}


def verify_files(root, files):
    for name, ref in files.items():
        require(sha(within(root, name)) == ref, f'Local artifact digest differs: {name}')


def job_refs(spec, split):
    return {key: job for key, job in spec['jobs'].items() if key.endswith('/'+split)}


def parity_records(out, spec, identity, split):
    refs = job_refs(spec, split)
    receipts = {}
    for key, expected in refs.items():
        path = out/'stages'/key/'job.json'
        record = read(path)
        require(record['identity'] == identity and record['result'] == expected,
                f'{split} parity receipt differs: {key}')
        base = path.parent
        filename = ('predictions.npz', 'head.joblib') if split == 'val' else ('predictions.npz',)
        require(set(record['files']) == set(filename), f'Unexpected {split} private artifacts: {key}')
        verify_files(base, record['files'])
        with np.load(base/'predictions.npz', allow_pickle=False) as predictions:
            ids, truth, pred = (predictions[name] for name in ('trial_id', 'y', 'pred'))
            require(ids.shape == truth.shape == pred.shape == (expected['n'],)
                    and prediction_sha(ids, truth, pred) == expected['prediction_sha256'],
                    f'{split} ordered prediction digest differs: {key}')
        receipts[key] = sha(path)
    return receipts


def verified_stage(out, spec, identity, split):
    path = out/f'{split}_parity.json'
    record = read(path)
    require(record == {'identity': identity, 'jobs': parity_records(out, spec, identity, split)},
            f'{split} aggregate parity receipt differs')
    return sha(path)


def after_validation(out, spec, identity, run_test):
    return run_test(verified_stage(out, spec, identity, 'val'))


def record_job(out, spec, identity, key, X, frame, target, head, usual):
    from modeling import models
    ids, y = frame.trial_id.to_numpy(), frame[target].to_numpy()
    expected = spec['jobs'][key]
    require(expected['feature_sha256'] == matrix_sha(X), f'{key}: held-out feature digest differs')
    pred = head.predict(X)
    require(expected['prediction_sha256'] == prediction_sha(ids, y, pred) and expected['n'] == len(y),
            f'{key}: ordered prediction digest differs')
    base = out/'stages'/key
    paths = [base/'predictions.npz']
    fields = dict(trial_id=ids, y=y, pred=pred, inseq=frame.inseq.to_numpy())
    if usual is not None:
        fields['usual'] = np.asarray([usual[o] for o in frame.odor])
    storage.write_npz(paths[0], **fields)
    if key.endswith('/val'):
        paths.append(base/'head.joblib')
        models.atomic_joblib(head, paths[-1])
    commit(base/'job.json', dict(identity=identity, result=expected, files=file_bindings(base, paths)))
    return pred


def execute(out, spec_path, spec, roots, *, interpretability_only):
    # Basic environment must be set before starting the scientific Python process.
    require(all(os.environ.get(k) == v for k, v in THREADS.items()) and sys.flags.no_user_site,
            'Export the one-thread scientific environment shown in README.md before launch')
    assert_runtimes(spec['python'], spec['packages'])
    require(set(spec['code']) == set(CODE), 'Run requires the checkout replay specification')
    verify_files(SOURCE.parent, spec['code'])
    output_boundary(out)
    for p in (spec_path, *roots.values()):
        nonoverlap(out, p)
    paths = checked_paths(spec, roots)
    identity_record = dict(spec_sha256=sha(spec_path), inputs=spec['inputs'])
    identity = digest(identity_record)
    with storage.lock(plain(out.with_name('.'+out.name+'.lock'))):
        if out.exists():
            require(out.is_dir(), 'Output must be a directory')
            for _ in inventory(out):
                pass
            require(read(out/'identity.json') == identity_record, 'Stale replay output or different inputs')
        else:
            out.mkdir()
            commit(out/'identity.json', identity_record)
        if (out/'replay_receipt.json').exists():
            verify_replay(spec_path, out)
            return
        state = {'stage': 'runtime', 'session': None, 'target': None}
        try:
            from threadpoolctl import threadpool_limits
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            with threadpool_limits(1):
                run_stages(out, spec, identity_record, identity, paths, roots, state, interpretability_only)
        except Exception as exc:
            storage.write_json(out/'failed_verification.json', {**state, 'error': str(exc), 'identity': identity})
            raise
        else:
            (out/'failed_verification.json').unlink(missing_ok=True)


def run_stages(out, spec, identity_record, identity, paths, roots, state, interpretability_only):
    import joblib
    from data_processing import preprocess
    from modeling import design as dec, models
    from .interpretability import Interpretation, verify_interpretation
    from .metrics import compare_metrics, usual_positions

    run = dec.Run(out)
    val_done = (out/'val_parity.json').exists()
    if val_done:
        verified_stage(out, spec, identity, 'val')
    interpret = (out/'interpretability_receipt.json').exists()
    code_sha = spec['code'][f'src/{INTERPRETATION_CODE}']
    if interpret:
        verify_interpretation(out, spec, identity_record['spec_sha256'], code_sha)
    collector = None if interpret else Interpretation()
    for s in SESSIONS:
        state.update(stage='preprocessing', session=s, target=None)
        preprocess.prepare(s, tuple(paths['raw'][raw_name(s, k)] for k in ('lfp', 'bvr')), out)
        if val_done and interpret:
            continue
        for t in TARGETS:
            state.update(stage='validation', target=t)
            design = dec.prepare(run, s, t, 500, record_transform=True)
            unit = spec['units'][f'{t}/{s}']
            require(digest(qa_reference(read(run.cache(s, t)/'qa.json'))) == unit['qa_sha256'], 'Channel order or guards differ')
            train, val = (np.flatnonzero(design.frame.split.eq(split)) for split in ('train', 'val'))
            tf, vf = design.frame.iloc[train], design.frame.iloc[val]
            for split, frame in [('train', tf), ('val', vf)]:
                ref = cohort_reference(frame, t)
                require(ref == spec['cohorts'][f'{t}/{s}/{split}'] and ref['classes'] == list(CLASSES),
                        f'{split} ordered cohort or class support differs')
            design._amplitude()
            require(transform_reference(design._transform_path, tf.trial_id.to_numpy()) == unit['transform'],
                    'Numerical TRAIN amplitude scaler differs')
            name = checkpoint_name(s, t)
            features = dec.matrices(design, paths['checkpoints'][name], spec['inputs']['checkpoints'][name], train)
            usual = usual_positions(tf) if t == 'position' else None
            for m, X in features.items():
                key = job_key(t, m, s, 'val')
                base = out/'stages'/key
                if (base/'job.json').exists():
                    record = read(base/'job.json')
                    require(record['identity'] == identity and record['result'] == spec['jobs'][key],
                            f'Committed validation job differs: {key}')
                    verify_files(base, record['files'])
                    require(set(record['files']) == {'head.joblib', 'predictions.npz'}, 'Incomplete validation job receipt')
                    head = joblib.load(base/'head.joblib')
                    require(matrix_sha(X[val]) == spec['jobs'][key]['feature_sha256'], f'{key}: resumed features differ')
                    with np.load(base/'predictions.npz', allow_pickle=False) as f:
                        pred = f['pred']
                else:
                    head = models.fit_l1(X[train], tf[t].to_numpy(), C=unit['C'][m], seed=42)
                    require(head.converged_, f'{key}: frozen L1 fit did not converge')
                    pred = record_job(out, spec, identity, key, X[val], vf, t, head, usual)
                if collector is not None:
                    collector.add(s, t, m, design, head, X, train, val, pred)
            print(f'Validated {s}/{t}', flush=True)
    state.update(stage='validation-parity', session=None, target=None)
    if not val_done:
        commit(out/'val_parity.json', dict(identity=identity, jobs=parity_records(out, spec, identity, 'val')))
    val_sha = verified_stage(out, spec, identity, 'val')
    val_metrics = compare_metrics(out, spec, 'val')
    if collector is not None:
        state.update(stage='interpretability')
        collector.finish(out, spec, identity_record['spec_sha256'], val_sha)
    verify_interpretation(out, spec, identity_record['spec_sha256'], code_sha)
    if interpretability_only:
        print('Exact validation parity and interpretation: 20 jobs; test not scored', flush=True)
        return

    def test_stage(validation_sha):
        for s in SESSIONS:
            for t in TARGETS:
                state.update(stage='test', session=s, target=t)
                keys = {m: job_key(t, m, s, 'test') for m in MODELS}
                if all((out/'stages'/key/'job.json').exists() for key in keys.values()):
                    continue
                design = dec.prepare(run, s, t, 500, split='test')
                require(cohort_reference(design.frame, t) == spec['cohorts'][f'{t}/{s}/test'], 'Test ordered cohort differs')
                name = checkpoint_name(s, t)
                Xs = dec.matrices(design, paths['checkpoints'][name], spec['inputs']['checkpoints'][name])
                source = dec.trial_frame(run, s, t)
                usual = usual_positions(source.loc[source.split.eq('train')]) if t == 'position' else None
                for m, key in keys.items():
                    if (out/'stages'/key/'job.json').exists():
                        continue
                    base = out/'stages'/job_key(t, m, s, 'val')
                    verify_files(base, read(base/'job.json')['files'])
                    head = joblib.load(base/'head.joblib')
                    record_job(out, spec, identity, key, Xs[m], design.frame, t, head, usual)
                print(f'Retrospective test parity {s}/{t}', flush=True)
        state.update(stage='test-parity', session=None, target=None)
        commit(out/'test_parity.json', dict(identity=identity, jobs=parity_records(out, spec, identity, 'test')))
        test_sha = verified_stage(out, spec, identity, 'test')
        test_metrics = compare_metrics(out, spec, 'test')
        state.update(stage='release-receipt')
        stage_files = [out/f for f in ('identity.json', 'val_parity.json', 'test_parity.json', 'interpretability_receipt.json')]
        # Include numerical caches, scale files and visible coverage in the private receipt.
        for name in ('preprocessing', 'transforms', 'coverage'):
            stage_files.extend(inventory(out/name))
        commit(out/'replay_receipt.json', dict(status='verified', identity=identity, spec_sha256=identity_record['spec_sha256'],
               sessions=list(SESSIONS), validation_before_test_sha256=validation_sha, test_sha256=test_sha,
               val=val_metrics, test=test_metrics, files=file_bindings(out, stage_files)))
        print('Exact replay parity: five sessions, 20 validation then 20 previously viewed test jobs', flush=True)

    after_validation(out, spec, identity, test_stage)


def verify_replay(spec_path, out):
    from .interpretability import verify_interpretation
    from .metrics import compare_metrics
    spec = load_replay_spec(spec_path)
    ref = read(out/'replay_receipt.json')
    identity_record = read(out/'identity.json')
    spec_sha = sha(spec_path)
    identity = digest(identity_record)
    require(ref['status'] == 'verified' and ref['sessions'] == list(SESSIONS) and ref['identity'] == identity
            and ref['spec_sha256'] == spec_sha and identity_record == dict(spec_sha256=spec_sha, inputs=spec['inputs']),
            'Replay receipt does not match this specification')
    verify_files(out, ref['files'])
    require(verified_stage(out, spec, identity, 'val') == ref['validation_before_test_sha256']
            and verified_stage(out, spec, identity, 'test') == ref['test_sha256'], 'Bound parity receipts differ')
    code_sha = next((spec['code'][name] for name in (
        f'src/{INTERPRETATION_CODE}', INTERPRETATION_CODE, 'science/evaluation/interpretability.py', 'interpretability.py',
    ) if name in spec['code']), REVIEWED_INTERPRETATION_SHA)
    verify_interpretation(out, spec, spec_sha, code_sha)
    for split in ('val', 'test'):
        require(compare_metrics(out, spec, split) == ref[split], f'Bound {split} metrics differ')
    print('Verified private replay receipt and file digests; no raw arrays or fits used', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('run', 'verify'):
        sub = commands.add_parser(name)
        sub.add_argument('--spec', required=True, type=Path, help='Full replay specification or checkout code revision')
        if name == 'run':
            sub.add_argument('--raw', required=True, type=Path)
            sub.add_argument('--checkpoints', required=True, type=Path)
            sub.add_argument('--interpretability-only', action='store_true')
    for sub in commands.choices.values():
        sub.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        out = plain(args.output)
        if args.action == 'verify':
            verify_replay(plain(args.spec), out)
        else:
            spec_path = plain(args.spec)
            spec = load_replay_spec(spec_path)
            execute(out, spec_path, spec, {k: plain(getattr(args, k)) for k in ('raw', 'checkpoints')},
                    interpretability_only=args.interpretability_only)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Replay failed: {exc}\n')


if __name__ == '__main__':
    main()
