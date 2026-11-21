"""Validate and load the reviewed poster aggregates (standard library only)."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

SCHEMA = 'poster-evidence-v1'
ALIASES = dict(zip(('080718_mitt', '081106_barat', '090212_stella', '090212_superchris', '090420_buchanan'),
                   ('Mitt', 'Barat', 'Stella', 'Superchris', 'Buchanan'), strict=True))
TARGETS = ('odor', 'position')
MODELS = ('historical_anchor', 'cebra_combined_mean')
SPLITS = ('val', 'test')
CLASSES = set(range(1, 6))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (*path.parents, path)), f'Symlink forbidden: {path}')
    if path.is_file():
        require(path.stat().st_nlink == 1, f'Hard-linked evidence member: {path}')
    return path


def sha(path):
    with plain(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(plain(path).read_bytes())


FIELDS = {name: tuple(fields.split()) for name, fields in {
    'headline': 'target model split ba observed_macro_recall n n_rat n_rat_ba',
    'session_scores': 'target model split session ba observed_macro_recall n n_rat n_rat_ba',
    'class_support': 'target model split session class support predicted tp recall',
    'slide_check': 'target model current_ba slide_ba slide_window_ms slide_configuration n_rat',
    'outseq': 'model split scope session n actual_support usual_support follows_actual follows_usual neither',
    'matrix': 'target encoder decoder is_anchor input_view window_ms status val_ba validation_rank test_n test_n_rat test_n_rat_ba recommended',
    'position_confusions': 'target model split subset actual predicted count',
    'signal_qa': 'session status legacy_status legacy_windows legacy_peak_hz legacy_attenuation_db high_resolution_status windows peak_hz attenuation_db evidence_status',
    'l1_weights': 'bin_start_ms frequency_hz mean_weight n_nonzero n_electrodes',
    'l1_masking': 'target session group intact_ba masked_ba delta n_classes n',
    'cebra_projection': 'cohort grouping class pc1 pc2 n pc1_variance pc2_variance',
}.items()}
MEMBERS = {name+'.csv' for name in FIELDS} | {'replay-spec.json'}
INTEGER = set('n n_rat n_rat_ba class support predicted tp slide_window_ms follows_actual follows_usual neither window_ms validation_rank test_n test_n_rat test_n_rat_ba actual count legacy_windows windows bin_start_ms n_nonzero n_electrodes n_classes'.split())
FLOAT = set('ba observed_macro_recall recall current_ba slide_ba val_ba legacy_peak_hz legacy_attenuation_db peak_hz attenuation_db frequency_hz mean_weight intact_ba masked_ba delta pc1 pc2 pc1_variance pc2_variance'.split())
BOOLEAN = {'is_anchor', 'recommended'}
VECTOR = {'actual_support', 'usual_support'}
KEYS = {(t, m, s) for t in TARGETS for m in MODELS for s in SPLITS}
PUBLISHED = {
    ('odor', MODELS[0]): (47.1, 56.6, .489, 500),
    ('odor', MODELS[1]): (44.7, 49.5, .460, 500),
    ('position', MODELS[0]): (50.8, 45.4, .400, 500),
    ('position', MODELS[1]): (44.3, 38.8, .287, 200),
}


def value(name, text):
    if text == '':
        return None
    if name in INTEGER:
        return int(text)
    if name in FLOAT:
        number = float(text)
        require(math.isfinite(number), f'Nonfinite {name}')
        return number
    if name in BOOLEAN:
        require(text in ('True', 'False'), f'Invalid boolean: {name}')
        return text == 'True'
    if name in VECTOR:
        vector = json.loads(text)
        require(isinstance(vector, list) and len(vector) == 5 and all(type(n) is int and n >= 0 for n in vector),
                f'Invalid class counts: {name}')
        return vector
    return text


def project(name, row, **extra):
    return {k: (extra[k] if k in extra else value(k, row[k])) for k in FIELDS[name]}


def read_table(root, name):
    with plain(root/(name+'.csv')).open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        require(tuple(reader.fieldnames or ()) == FIELDS[name], f'Unexpected {name} header')
        out = []
        for row in reader:
            require(tuple(row) == FIELDS[name] and None not in row.values(), f'Malformed {name} row')
            out.append(project(name, row))
        return out


def close(x, y):
    return x == y or (x is not None and y is not None and abs(x-y) <= 1e-12)


def mean(values):
    seq = list(values)
    return math.fsum(seq)/len(seq) if seq else None


def indexed(items, *fields):
    result = {tuple(row[k] for k in fields): row for row in items}
    require(len(result) == len(items), f'Duplicate aggregate records: {fields}')
    return result


def check_aggregates(e):
    headlines = indexed(e['headline'], 'target', 'model', 'split')
    sessions = indexed(e['session_scores'], 'target', 'model', 'split', 'session')
    classes = indexed(e['class_support'], 'target', 'model', 'split', 'session', 'class')
    require(set(headlines) == KEYS and len(sessions) == 40 and len(classes) == 200, 'Incomplete headline support')
    for (t, m, split), r in headlines.items():
        key = (t, m, split)
        other = MODELS[1] if m == MODELS[0] else MODELS[0]
        all_scores = []
        for s in ALIASES.values():
            current = sessions[(*key, s)]
            all_scores.append(current)
            stats = [classes[(*key, s, c)] for c in sorted(CLASSES)]
            for item in stats:
                n, p, tp = (item[k] for k in ('support', 'predicted', 'tp'))
                require(0 <= tp <= min(n, p) and close(item['recall'], tp/n if n else None)
                        and n == classes[t, other, split, s, item['class']]['support'], f'Invalid paired class support: {key}, {s}')
            observed = [item['recall'] for item in stats if item['support']]
            complete = not (split == 'test' and s == 'Buchanan')
            require(sum(item['support'] for item in stats) == sum(item['predicted'] for item in stats) == current['n']
                    and len(observed) == 4+complete and current['n_rat'] == 1 and current['n_rat_ba'] == int(complete)
                    and close(mean(observed), current['observed_macro_recall'])
                    and close(mean(observed) if complete else None, current['ba']), f'Incorrect per-rat denominator: {key}, {s}')
        require(r['n'] == sum(x['n'] for x in all_scores) and r['n_rat'] == 5
                and r['n_rat_ba'] == sum(x['n_rat_ba'] for x in all_scores) == (5 if split == 'val' else 4)
                and close(r['observed_macro_recall'], mean(x['observed_macro_recall'] for x in all_scores))
                and close(r['ba'], mean(x['ba'] for x in all_scores) if split == 'val' else None)
                and round(100*r['observed_macro_recall'], 1) == PUBLISHED[t,m][split == 'test'], f'Incorrect equal-rat aggregate: {key}')
    slide = indexed(e['slide_check'], 'target', 'model')
    require(set(slide) == set(PUBLISHED), 'Incomplete slide check')
    for key, r in slide.items():
        require(r['n_rat'] == 4 and (r['slide_ba'], r['slide_window_ms']) == PUBLISHED[key][2:]
                and close(r['current_ba'], mean(sessions[(*key, 'test', s)]['ba']
                                               for s in ALIASES.values() if s != 'Buchanan')), f'Incorrect slide reconciliation: {key}')
    outseq = indexed(e['outseq'], 'model', 'split', 'scope', 'session')
    require(len(outseq) == 24, 'Incomplete OutSeq support')
    for m in MODELS:
        for split in SPLITS:
            pooled = outseq[m, split, 'pooled_trials', 'pooled']
            per_rat = [outseq[m, split, 'session', s] for s in ALIASES.values()]
            count_keys = ('n', 'follows_actual', 'follows_usual', 'neither')
            for r in [*per_rat, pooled]:
                n, *counts = (r[k] for k in count_keys)
                require(min(counts) >= 0 and sum(counts) == n == sum(r['actual_support']) == sum(r['usual_support']),
                        'OutSeq denominator differs')
            for k in count_keys:
                require(pooled[k] == sum(r[k] for r in per_rat), 'OutSeq pooled counts differ')
            for k in VECTOR:
                require(pooled[k] == [sum(v) for v in zip(*(r[k] for r in per_rat), strict=True)], 'OutSeq pooled supports differ')
            other = outseq[MODELS[1] if m == MODELS[0] else MODELS[0], split, 'pooled_trials', 'pooled']
            require(all(pooled[k] == other[k] for k in ('n', *VECTOR)), 'OutSeq model supports differ')
            if split == 'test':
                require(tuple(pooled[k] for k in count_keys) == ((8, 1, 4, 3) if m == MODELS[0] else (8, 0, 2, 6)),
                        'Eight-trial retrospective check differs')
    conf = indexed(e['position_confusions'], 'model', 'split', 'subset', 'actual', 'predicted')
    require(set(conf) == {(m, s, sub, a, p) for m in MODELS for s in SPLITS
                         for sub in ('all', 'outseq') for a in CLASSES for p in CLASSES}, 'Incomplete pooled confusion grid')
    for m in MODELS:
        for s in SPLITS:
            for a in CLASSES:
                truth = sum(classes['position', m, s, rat, a]['support'] for rat in ALIASES.values())
                require(sum(conf[m,s,'all',a,p]['count'] for p in CLASSES) == truth
                        and all(0 <= conf[m,s,'outseq',a,p]['count'] <= conf[m,s,'all',a,p]['count']
                                and conf[m,s,'all',a,p]['target'] == conf[m,s,'outseq',a,p]['target'] == 'position'
                                for p in CLASSES), 'Invalid position confusion counts')
    pairs = indexed(e['matrix'], 'target', 'encoder', 'decoder', 'is_anchor')
    require(len(pairs) == 392 and sum(r['is_anchor'] for r in pairs.values()) == 2
            and {r['status'] for r in pairs.values()} == {'complete', 'not_scheduled'}, 'Matrix roster differs')
    for t in TARGETS:
        grid = [r for r in pairs.values() if r['target'] == t and not r['is_anchor']]
        a = pairs[t, 'amp_bin50', 'linear_l1', True]
        require(len(grid) == 195 and len({r['encoder'] for r in grid}) == 15
                and len({r['decoder'] for r in grid}) == 13 and sum(r['status'] == 'complete' for r in grid) == 66
                and a['validation_rank'] is None and not a['recommended'] and a['status'] == 'complete', 'Invalid target matrix grid')
        for r in [*grid, a]:
            require(r['window_ms'] == 500 and r['input_view'], 'Matrix input view/window differs')
            if r['status'] == 'not_scheduled':
                require(not r['recommended'] and all(r[k] is None for k in ('val_ba', 'validation_rank', 'test_n',
                                                                          'test_n_rat', 'test_n_rat_ba')), 'Unscheduled row has a score')
            else:
                require(0 <= r['val_ba'] <= 1 and r['test_n'] == headlines[t, MODELS[0], 'test']['n']
                        and (r['test_n_rat'], r['test_n_rat_ba']) == (5, 4), 'Matrix seed-42 test support differs')
        winner, = [r for r in grid if r['recommended']]
        require((winner['encoder'], winner['decoder']) == (('amp_mean', 'hist_boost') if t == 'odor' else ('wave_tangent', 'rbf_svm'))
                and winner['validation_rank'] == 1, 'Frozen recommendation differs')
    return dict(matrix_slots=len(pairs), scheduled_pairs=132, historical_anchors=2, terminal_tuning_trials=824)


def check_interpretability(e):
    weights = indexed(e['l1_weights'], 'bin_start_ms', 'frequency_hz')
    require(len(weights) == 120 and {k[0] for k in weights} == set(range(0, 500, 50)) and len({k[1] for k in weights}) == 12
            and all(3.9999999 <= r['frequency_hz'] <= 150.0000001 and 0 <= r['n_nonzero'] <= r['n_electrodes']
                    and r['n_electrodes'] > 0 for r in weights.values()), 'Invalid weight grid/support')
    masks = indexed(e['l1_masking'], 'target', 'session', 'group')
    groups = {'freq_0_4', 'freq_4_8', 'freq_8_12', 'time_0_5', 'time_5_10'}
    require(set(masks) == {(t, s, g) for t in TARGETS for s in ALIASES.values() for g in groups}, 'Incomplete active masks')
    scores = indexed(e['session_scores'], 'target', 'model', 'split', 'session')
    for (t, s, _), r in masks.items():
        val = scores[t, MODELS[0], 'val', s]
        require(r['n_classes'] == 5 and r['n'] == val['n'] and close(r['intact_ba'], val['ba'])
                and 0 <= r['masked_ba'] <= 1 and close(r['delta'], r['intact_ba']-r['masked_ba']), 'Masking score differs')
    centroids = indexed(e['cebra_projection'], 'cohort', 'grouping', 'class')
    ev = {(r['pc1_variance'], r['pc2_variance']) for r in centroids.values()}
    require(len(ev) == 1 and all(0 < b <= a <= 1 and a+b <= 1 for a, b in ev)
            and {(c,g) for c,g,_ in centroids} == {(c,g) for c in ('all', 'outseq') for g in TARGETS}, 'Invalid centroid axes')
    for c, g, k in centroids:
        r = centroids[c,g,k]
        require(k in CLASSES and 0 < r['n'] <= centroids['all',g,k]['n'], 'Invalid centroid support')
    for g in TARGETS:
        require(all(('all',g,k) in centroids for k in CLASSES)
                and sum(r['n'] for (c,grp,_), r in centroids.items() if (c,grp) == ('all',g)) == 45
                and sum(r['n'] for (c,grp,_), r in centroids.items() if (c,grp) == ('outseq',g)) == 3, 'Incorrect Mitt validation cohort')


def load_evidence(root):
    root = plain(root)
    manifest = read_json(root/'manifest.json')
    require(manifest['schema'] == SCHEMA and set(manifest['members']) == MEMBERS
            and set(p.name for p in root.iterdir()) == MEMBERS | {'manifest.json'}, 'Unexpected evidence schema or inventory')
    for name, expected in manifest['members'].items():
        require(sha(root/name) == expected, f'Evidence SHA-256 mismatch: {name}')
    require(all(re.fullmatch('[0-9a-f]{64}', v) and not Path(k).is_absolute() and '..' not in Path(k).parts
                for k, v in manifest['inputs'].items()), 'Invalid input alias or SHA-256')
    evidence = {name: read_table(root, name) for name in FIELDS}
    spec = read_json(root/'replay-spec.json')
    require(spec['schema'] == 'poster-replay-v1' and spec['encoder_seed'] == spec['head_seed'] == 42
            and spec['sessions'] == list(ALIASES) and spec['classes'] == sorted(CLASSES)
            and spec['window_ms'] == 500 and len(spec['jobs']) == 40
            and all(spec[k+'_sha256'] == manifest['inputs'][f'matrix/{f}']
                    for k, f in [('review','review_receipt.json'), ('freeze','freeze.json'),
                                 ('test_evaluation','test_evaluation.json'), ('matrix_source','manifest.json')]),
            'Replay specification refers to a different reviewed run')
    require(check_aggregates(evidence) == manifest['counts'], 'Evidence counts differ')
    require({r['session']: r['evidence_status'] for r in evidence['signal_qa']} ==
            {s: ('independently_reviewed_mitt' if s == 'Mitt' else 'runner_reported') for s in ALIASES.values()}
            and len(evidence['signal_qa']) == 5 and all(r['status'] == r['legacy_status'] == r['high_resolution_status'] == 'done'
                                                     for r in evidence['signal_qa']), 'Invalid spectral review scope')
    check_interpretability(evidence)
    return {**evidence, 'manifest': manifest, 'replay-spec': spec}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = load_evidence(args.evidence)['manifest']
        print(f'{manifest["schema"]}: {len(manifest["members"])} hash-verified members; {manifest["counts"]}')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Evidence rejected: {exc}\n')


if __name__ == '__main__':
    main()
