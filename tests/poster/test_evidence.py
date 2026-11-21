"""The aggregate release must load offline with fixed numeric and privacy boundaries."""
import json
import math
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
POSTER = ROOT/'poster'
sys.path.insert(0, str(ROOT/'src'))
from evaluation.evidence_tools import load_evidence


class ReviewedEvidence(unittest.TestCase):
    def test_copy_loads_identically_and_tampered_or_extra_members_refuse(self):
        reference = load_evidence(POSTER/'evidence')
        with TemporaryDirectory() as temp:
            packet = Path(temp)/'evidence'
            shutil.copytree(POSTER/'evidence', packet)
            self.assertEqual(load_evidence(packet), reference)
            path = packet/'headline.csv'
            original = path.read_bytes()
            at = original.index(b'0.4')
            path.write_bytes(original[:at]+b'1'+original[at+1:])
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch: headline.csv'):
                load_evidence(packet)
            path.write_bytes(original)
            extra = packet/'not-reviewed.csv'
            extra.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                load_evidence(packet)
            extra.unlink()
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'inventory'):
                load_evidence(packet)

    def test_equal_rat_observed_recall_and_outseq_counts(self):
        data = load_evidence(POSTER/'evidence')
        truth = {('odor', 'historical_anchor'): (.47104761904761905, .566063492063492, 86),
                 ('odor', 'cebra_combined_mean'): (.44676190476190475, .4953333333333333, 86),
                 ('position', 'historical_anchor'): (.5078614718614718, .4536190476190476, 94),
                 ('position', 'cebra_combined_mean'): (.4428658008658008, .3883888888888889, 94)}
        for (target, model), (validation, retrospective, n) in truth.items():
            val, = [r for r in data['headline'] if (r['target'], r['model'], r['split']) == (target, model, 'val')]
            test, = [r for r in data['headline'] if (r['target'], r['model'], r['split']) == (target, model, 'test')]
            self.assertAlmostEqual(val['ba'], validation, places=12)
            self.assertEqual((test['n'], test['n_rat'], test['n_rat_ba'], test['ba']), (n, 5, 4, None))
            rat_rows = [r for r in data['session_scores'] if (r['target'], r['model'], r['split']) == (target, model, 'test')]
            self.assertEqual({r['session'] for r in rat_rows if r['ba'] is None}, {'Buchanan'})
            self.assertAlmostEqual(math.fsum(r['observed_macro_recall'] for r in rat_rows)/5, retrospective, places=12)
            self.assertAlmostEqual(test['observed_macro_recall'], retrospective, places=12)
        counts = {r['model']: tuple(r[k] for k in ('n', 'follows_actual', 'follows_usual', 'neither'))
                  for r in data['outseq'] if (r['split'], r['scope']) == ('test', 'pooled_trials')}
        self.assertEqual(counts, {'historical_anchor': (8, 1, 4, 3), 'cebra_combined_mean': (8, 0, 2, 6)})
        matrix = data['matrix']
        self.assertEqual(len(matrix), 392)
        self.assertEqual(sum(r['is_anchor'] for r in matrix), 2)
        self.assertEqual(sum(not r['is_anchor'] and r['status'] == 'complete' for r in matrix), 132)

    def test_portable_packet_does_not_export_private_rows_or_absolute_paths(self):
        root = POSTER/'evidence'
        for p in root.iterdir():
            text = p.read_text(encoding='utf-8')
            self.assertIn(p.suffix, ('.csv', '.json'))
            self.assertNotRegex(text, r'(^|[\"\s])/(home|Users|mnt|var|tmp)/|[A-Z]:\\')
            if p.suffix == '.csv':
                self.assertNotRegex(text.splitlines()[0], r'\b(trial_id|y|pred|proba|onset|model_file|channel_id)\b')
        spec = json.loads((root/'replay-spec.json').read_text(encoding='utf-8'))
        self.assertEqual(set(spec['inputs']), {'raw', 'checkpoints'})
        for files in spec['inputs'].values():
            self.assertEqual(len(files), 10)
            self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', v) for v in files.values()))
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}', r['prediction_sha256']) for r in spec['jobs'].values()))
