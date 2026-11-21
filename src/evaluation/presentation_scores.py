"""Rebuild the presentation's published aggregates from archived score tables."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

RUN = 'run-d99f49d7dfac72c9'
SOURCE_HASHES = {
    'selected_validation.csv': '0957418eda96ceb731d6d2f6f8b7fd71ac05afc060aff70b80f351228c2bbbcb',
    'locked_test_scores.csv': 'ec057e9b0d1f2837674e862c5e519ff047ccfbd543e4f630883d2b2d8318de45',
}
DIMS = ['target', 'start', 'duration', 'rep', 'pool', 'steps', 'arch', 'seed', 'decoder']
SESSIONS = ('080718_mitt', '081106_barat', '090212_stella', '090212_superchris', '090420_buchanan')


def archived_table(archive, name):
    path = archive / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError(f'{path} is not the reviewed {RUN} table')
    return pd.read_csv(path, keep_default_na=False)


def aggregates(archive):
    val = archived_table(archive, 'selected_validation.csv')
    test = archived_table(archive, 'locked_test_scores.csv')
    if not val.full_val.eq(1).all() or set(val.session) != set(SESSIONS):
        raise ValueError('Validation must include full five-class support in all five rats')
    test['score'] = pd.to_numeric(test.five_class_balanced_accuracy.replace('', None))
    if (test.score.notna() != test.n_classes_test.eq(5)).any():
        raise ValueError('Strict test balanced accuracy requires all five classes')
    # The slide 34 pooling panel uses the same four sessions for both splits.
    paired = val[val.session.isin(test.loc[test.target.eq('position') & test.score.notna(), 'session'])]
    paired = paired.query('target == "position" and start == 0 and duration == 200 and '
                          'rep == "cebra_combined" and seed == 42 and steps == 1000 and decoder == "l1"')
    tables = []
    for split, frame, stages, n in [('validation', val, True, 5), ('paired_validation', paired, True, 4),
                                     ('historical_test', test.dropna(subset=['score']), False, 4)]:
        keys = ['stage', *DIMS] if stages else DIMS
        summary = frame.groupby(keys, sort=True).score.agg(mean_ba='mean', n_rat='size').reset_index()
        if not summary.n_rat.eq(n).all() or frame.duplicated([*keys, 'session']).any():
            raise ValueError(f'Unexpected per-recipe session roster: {split}')
        tables.append(summary.assign(split=split, stage=summary.stage if stages else 0))
    return pd.concat(tables, ignore_index=True)[['split', 'stage', *DIMS, 'n_rat', 'mean_ba']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    table = aggregates(args.archive)
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        table.to_csv(stream, index=False, float_format='%.12g')
    print(f'Wrote {len(table)} aggregate recipe rows to {args.output}')


if __name__ == '__main__':
    main()
