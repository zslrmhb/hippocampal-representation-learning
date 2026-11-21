"""A foreign checkpoint and a divergent VAL receipt must never open science gates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REVIEWED_SPEC = ROOT/'poster/evidence/replay-spec.json'
CHECKOUT_SPEC = ROOT/'poster/replay-code.json'

SOURCE = ROOT/'src'
sys.path.insert(0, str(SOURCE))
from data_processing.artifacts import canonical
from evaluation import replay


class ReplayGates(unittest.TestCase):
    def test_checkout_spec_preserves_reviewed_science(self):
        original = replay.load_replay_spec(REVIEWED_SPEC)
        current = replay.load_replay_spec(CHECKOUT_SPEC)
        self.assertNotEqual(current['code'], original['code'])
        replay.verify_files(ROOT, current['code'])
        self.assertEqual({**current, 'code': original['code']}, original)

    def test_foreign_checkpoint_refuses_before_materialization(self):
        with TemporaryDirectory() as temp:
            local = Path(temp)
            spec = replay.load_replay_spec(CHECKOUT_SPEC)
            fresh = b'private input fixture'
            expected = hashlib.sha256(fresh).hexdigest()
            for kind, members in spec['inputs'].items():
                for name in members:
                    path = local/kind/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(fresh)
                    members[name] = expected
            foreign = next(iter(spec['inputs']['checkpoints']))
            (local/'checkpoints'/foreign).write_bytes(b'not a checkpoint')
            spec_path = local/'spec.json'
            spec_path.write_text(json.dumps(spec), encoding='utf-8')
            out = local/'replay'
            # This gate is independent of package versions. A standard root
            # environment should not need the suite's CUDA packages to test it.
            entry = ('from unittest.mock import patch\n'
                     'from evaluation import replay\n'
                     'with patch.object(replay, "assert_runtimes"):\n'
                     '    replay.main()\n')
            actual = subprocess.run([sys.executable, '-B', '-c', entry, 'run',
                                     '--spec', str(spec_path), '--raw', str(local/'raw'),
                                     '--checkpoints', str(local/'checkpoints'), '--output', str(out)],
                                    env={**os.environ, **replay.THREADS, 'PYTHONPATH': str(SOURCE)},
                                    capture_output=True, text=True, check=False)
            self.assertEqual(actual.returncode, 1, actual.stdout + actual.stderr)
            self.assertIn('input SHA-256 mismatch', actual.stderr)
            self.assertIn('checkpoints: '+foreign, actual.stderr)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name('.'+out.name+'.lock').exists())

    def test_wrong_validation_digest_keeps_test_selector_closed(self):
        key = 'odor/historical_anchor/080718_mitt/val'
        fields = dict(trial_id=np.array([3, 7]), y=np.array([1, 5]), pred=np.array([2, 5]))
        expected = dict(prediction_sha256=replay.prediction_sha(*fields.values()), feature_sha256='c'*64, n=2)
        testing = []
        with TemporaryDirectory() as temp:
            out = Path(temp)
            base = out/'stages'/key
            base.mkdir(parents=True)
            (base/'head.joblib').write_bytes(b'private fixture')

            def write_pair(arrays):
                np.savez(base/'predictions.npz', **arrays)
                payload = {name: replay.sha(base/name) for name in ('head.joblib', 'predictions.npz')}
                (base/'job.json').write_bytes(canonical(dict(identity='identity', result=expected, files=payload)))
                (out/'val_parity.json').write_bytes(
                    canonical(dict(identity='identity', jobs={key: replay.sha(base/'job.json')})))

            write_pair(fields)
            altered = {**expected, 'prediction_sha256': 'b'*64}
            with self.assertRaisesRegex(ValueError, 'val parity receipt differs'):
                replay.after_validation(out, {'jobs': {key: altered}}, 'identity', testing.append)
            self.assertEqual(testing, [])
            replay.after_validation(out, {'jobs': {key: expected}}, 'identity', testing.append)
            self.assertEqual(testing, [replay.sha(out/'val_parity.json')])
            testing.clear()
            write_pair({name: values[::-1] for name, values in fields.items()})
            with self.assertRaisesRegex(ValueError, 'val ordered prediction digest differs'):
                replay.after_validation(out, {'jobs': {key: expected}}, 'identity', testing.append)
            self.assertEqual(testing, [])

    def test_divergent_features_are_not_scored(self):
        import pandas as pd

        key = 'position/historical_anchor/080718_mitt/test'
        frame = pd.DataFrame({'trial_id': [12], 'position': [1]})

        class NotAScorer:
            def predict(self, X):
                raise AssertionError('Divergent held-out features were scored')

        with TemporaryDirectory() as temp:
            out = Path(temp)
            spec = {'jobs': {key: {'feature_sha256': '0'*64}}}
            with self.assertRaisesRegex(ValueError, 'held-out feature digest differs'):
                replay.record_job(out, spec, 'identity', key, np.ones((1, 3)), frame, 'position', NotAScorer(), None)
            self.assertFalse((out/'stages').exists())
