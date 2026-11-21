"""Render a one-page landscape poster and its matching wireframe."""
import argparse
import json
from os.path import relpath
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .report import HERE, RATS, TARGETS, _atomic_write
from .evidence_tools import load_evidence

ROOT = HERE.parents[1]
REPO = 'https://github.com/zslrmhb/hippocampal-representation-learning'
PIPELINES = {'historical_anchor': 'Direct wavelet amplitude', 'cebra_combined_mean': 'Wavelet CEBRA-Time'}
SOURCES = {
    'methods': [('Preprocessing notebook', 'notebooks/presentation_preprocessing.ipynb'),
                ('Signal pipeline', 'src/data_processing/preprocess.py'), ('Model design', 'src/modeling/design.py')],
    'scores': [('Poster notebook', 'notebooks/poster_results.ipynb'), ('Headline scores', 'poster/evidence/headline.csv'),
               ('Per-rat scores', 'poster/evidence/session_scores.csv')],
    'interpretation': [('All-rat aggregates', 'poster/interpretation-all-rats.json'),
                       ('Interpretation notebook', 'notebooks/poster_interpretation.ipynb'),
                       ('Validation export', 'src/evaluation/poster_interpretation.py')],
    'outseq': [('Out-of-sequence counts', 'poster/evidence/outseq.csv'), ('Metric code', 'src/evaluation/metrics.py')],
}
GROUPS = {'freq_0_4': '4–11 Hz', 'freq_4_8': '15–40 Hz', 'freq_8_12': '56–150 Hz',
          'time_0_5': '0–250 ms', 'time_5_10': '250–500 ms'}
CLASS_COLORS = ['#245699', '#ac491a', '#207761', '#85388a', '#866700']


def panel_data(supplement, evidence):
    expected = {(r, t) for r in RATS for t in TARGETS}
    panels = [{(r['rat'], r['target']): r for r in supplement[key]} for key in ('weights', 'latents')]
    if any(len(supplement[key]) != len(expected) or set(panel) != expected
           for key, panel in zip(('weights', 'latents'), panels, strict=True)):
        raise ValueError('Poster interpretation must cover each rat and target exactly once')
    if {(r['session'], r['target'], r['group']) for r in supplement['masking']} != {
            (rat, target, group) for rat, target in expected for group in GROUPS} or len(supplement['masking']) != 50:
        raise ValueError('Incomplete or duplicated grouped masking evidence')
    if supplement['masking'] != evidence['l1_masking']:
        # The release CSV has a different row order and may round at readback.
        a, b = (sorted(rows, key=lambda r: (r['target'], r['session'], r['group']))
                for rows in (supplement['masking'], evidence['l1_masking']))
        for x, y in zip(a, b, strict=True):
            if any(x[k] != y[k] for k in ('target', 'session', 'group', 'n', 'n_classes')) or not np.allclose(
                    [x[k] for k in ('intact_ba', 'masked_ba', 'delta')],
                    [y[k] for k in ('intact_ba', 'masked_ba', 'delta')], rtol=0, atol=1e-12):
                raise ValueError('Grouped masking disagrees with reviewed poster evidence')
    weights, latent = panels
    # Equal weight by rat and class, after averaging within rat over channels.
    heatmaps = {t: np.mean([weights[r, t]['magnitude'] for r in RATS], axis=(0, 1)).T for t in TARGETS}
    maximum = max(a.max() for a in heatmaps.values())
    cells = {t: [[int(np.rint(100*v/maximum)) for v in row] for row in a[::-1]] for t, a in heatmaps.items()}
    masking = []
    for target in TARGETS:
        lookup = {(r['group'], r['session']): -100*r['delta'] for r in supplement['masking'] if r['target'] == target}
        for key, label in GROUPS.items():
            values = [lookup[key, rat] for rat in RATS]
            masking.append(dict(target=TARGETS[target], key=key, group=label, mean=np.mean(values), values=values))
    scatter = {}
    for target in TARGETS:
        charts = []
        for rat in RATS:
            row = latent[rat, target]
            marks = [p for p in row['groups'] if p['cohort'] == 'all' and p['label'] == target]
            xy = np.array([p['xy'] for p in marks]); bounds = np.abs(xy).max(axis=0).clip(min=1e-12)
            points = [dict(x=49+39*x/bounds[0], y=35-24*y/bounds[1], color=CLASS_COLORS[m['cls']-1],
                           label='ABCDE'[m['cls']-1] if target == 'odor' else str(m['cls']),
                           n=m['n']) for (x,y),m in zip(xy,marks,strict=True)]
            charts.append(dict(rat=rat, n=row['n_val'], variance=round(100*sum(row['explained'])), points=points))
        scatter[target] = charts
    return dict(cells=cells, maximum=maximum, masking=masking, scatter=scatter, class_colors=CLASS_COLORS,
                low_band=[row for row in masking if row['key'] == 'freq_0_4'])

def decoding_rows(evidence):
    """Pair strict test scores on a common cohort; keep all rats on validation."""
    scores = {(r['target'], r['model'], r['split'], r['session']): r['ba']
              for r in evidence['session_scores']}
    test_rats = tuple(r for r in RATS if r != 'Buchanan')
    panels = []
    for target, title in TARGETS.items():
        rows = []
        for model, label in PIPELINES.items():
            test = {r: scores[target, model, 'test', r] for r in RATS}
            if tuple(r for r, score in test.items() if score is not None) != test_rats:
                raise ValueError('Strict test balanced accuracy must use the same four rats')
            rows.append(dict(model=label, test=np.mean([test[r] for r in test_rats]),
                             validation=np.mean([scores[target, model, 'val', r] for r in RATS])))
        panels.append(dict(target=title, rows=rows))
    return panels



def render(output):
    from playwright.sync_api import sync_playwright

    evidence = load_evidence(ROOT/'poster/evidence')
    supplement = json.loads((ROOT/'poster/interpretation-all-rats.json').read_text())
    data = panel_data(supplement, evidence)
    headline = decoding_rows(evidence)
    score_limit = max(60, int(np.ceil(max(r['test'] for p in headline for r in p['rows'])*5))*20)
    counts = {(r['split'], r['model']): r for r in evidence['outseq'] if r['scope'] == 'pooled_trials'}
    outseq = [dict(counts[split, m], label=label, split_label=title)
              for split, title in [('test', 'Test'), ('val', 'Validation')] for m, label in PIPELINES.items()]
    env = Environment(loader=FileSystemLoader(HERE), autoescape=select_autoescape(('html', 'jinja')),
                      trim_blocks=True, lstrip_blocks=True)
    template = env.get_template('poster_landscape.html.jinja')
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport=dict(width=1587, height=1122))
            page.emulate_media(media='print')
            for variant in ('wireframe', 'results'):
                path = output/f'poster-{variant}.html'
                context = dict(wireframe=variant=='wireframe', repo=REPO, sources=SOURCES,
                               headline=headline, score_limit=score_limit, outseq=outseq, rats=RATS, targets=TARGETS, **data)
                html = template.render(source_url=lambda p: relpath(ROOT/p, output), **context)
                _atomic_write(path, html.encode())
                page.set_content(template.render(source_url=lambda p: f'{REPO}/blob/HEAD/{p}', **context))
                if page.locator('article').bounding_box()['height'] > 297*96/25.4:
                    raise ValueError(f'{variant} does not fit one A3 landscape page')
                _atomic_write(path.with_suffix('.pdf'), page.pdf(print_background=True, prefer_css_page_size=True))
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/poster-landscape')
    output = parser.parse_args().output.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT/p) for p in ('poster','src','data','notebooks','presentation')):
        parser.error('Choose a generated-output directory (e.g. results/poster-landscape)')
    render(output)


if __name__ == '__main__':
    main()
