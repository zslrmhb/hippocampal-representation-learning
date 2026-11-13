"""Signal preservation, context isolation and matched target export checks."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
import numpy as np
from scipy.signal import sosfiltfilt
from hippocampal_preprocessing.filtering import notch_design
from hippocampal_preprocessing.pipeline import Config, run
from test_preprocessing import event_fixture


def decoding_fixture(root, gap=False, artifacts=False):
    session = 'synthetic'
    folder = root/session
    folder.mkdir(parents=True)
    onsets = (4000, 11000, 20000, 35000, 44000, 60000, 72000, 76000)
    time, events = event_fixture(n=80000, onsets=onsets)
    events['InSeqLog'] = {onset: -1 if i == 1 else 1 for i, onset in enumerate(onsets)}
    events['Position3'][onsets[1]] = events['Position2'].pop(onsets[1])
    if gap:
        time[32000:] += 10
    keys = ['TimeBin']+list(events)
    behavior = np.zeros((len(keys), len(time)))
    behavior[0] = time
    for row, key in enumerate(keys[1:], 1):
        for index, value in events[key].items():
            behavior[row, index] = value
    t = np.arange(len(time))/1000
    neural = .25*np.sin(2*np.pi*8*t)+.2*np.sin(2*np.pi*30*t)+.1*np.sin(2*np.pi*70*t)
    source = np.random.default_rng(12).normal(scale=.05, size=(3, len(time)))
    source[:2] += neural+3*np.sin(2*np.pi*60*t)
    source[2] = 7
    if artifacts:
        source[0, 30000] = np.nan  # Outside trial windows; cannot poison a filter block.
        source[0, 42000] = 10000  # Guard must also remove the otherwise clean 44000 trial.
        source[1, 48000:48200] = 0  # Detect a local flatline in filter context.
    np.savez_compressed(folder/f'{session}_bvr.npz', data=behavior, keys=np.array(keys))
    np.savez_compressed(folder/f'{session}_lfp.npz', data=source,
                        keys=np.array(['one', 'two', 'constant']))
    return session, source


class NotchTests(unittest.TestCase):
    def test_line_suppression_passbands_and_guard_sensitivity(self):
        config = Config(profile='inseq-decoding')
        sos, info = notch_design(config)
        t = np.arange(20000)/1000
        x = sum(np.sin(2*np.pi*f*t) for f in (8, 30, 60, 70))
        clean = sosfiltfilt(sos, x, padlen=info['guard_samples'])
        selected = slice(5000, 15000)
        for f in (8, 30, 70):
            amplitude = 2*np.mean(clean[selected]*np.sin(2*np.pi*f*t[selected]))
            self.assertAlmostEqual(amplitude, 1, delta=.02)
        line = 2*np.mean(clean[selected]*np.sin(2*np.pi*60*t[selected]))
        self.assertLess(abs(line), 1e-6)
        # Removing external context only changes guarded interior at small scale.
        short = sosfiltfilt(sos, x[2000:18000], padlen=info['guard_samples'])
        g = info['guard_samples']
        np.testing.assert_allclose(short[g:-g], clean[2000+g:18000-g], atol=1e-5, rtol=0)
        impulse = np.zeros_like(x)
        impulse[10000] = 1
        response = sosfiltfilt(sos, impulse, padlen=g)
        self.assertLess(np.max(np.abs(response[:10000-g])), 1e-6)
        self.assertLess(np.max(np.abs(response[10001+g:])), 1e-6)

    def test_invalid_filter_configuration(self):
        for kwargs in (dict(notch_hz=500), dict(notch_hz=0), dict(notch_q=0),
                       dict(filter_tolerance=1), dict(filter_tolerance=0),
                       dict(flatline_ms=0), dict(notch_q=float('nan'))):
            with self.assertRaises(ValueError):
                Config(profile='inseq-decoding', **kwargs).validate()


class InSeqExportTests(unittest.TestCase):
    def test_include_outseq_retains_mismatched_targets_with_shared_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw_data', Path(tmp)/'processed'
            session, _ = decoding_fixture(root)
            config = Config(profile='inseq-decoding', include_outseq=True)
            summary = run(root, output, [session], config, plots=False)[0]
            folder = output/session
            with (np.load(folder/'odor_decoding.npz') as odor,
                  np.load(folder/'position_decoding.npz') as position,
                  np.load(folder/'labels.npz') as labels):
                for key in ('X', 'trial_ids', 'raw_indices', 'split'):
                    np.testing.assert_array_equal(odor[key], position[key])
                np.testing.assert_array_equal(odor['trial_ids'], labels['trial_ids'])
                outseq = labels['inseq'] == 0
                np.testing.assert_array_equal(odor['trial_ids'][outseq], [1])
                np.testing.assert_array_equal(odor['y'][outseq], [1])
                np.testing.assert_array_equal(position['y'][outseq], [2])
            self.assertEqual(summary['after']['inseq'], {'0': 1, '1': 7})
            self.assertFalse(summary['target_arrays_identical'])

    def test_shared_x_real_targets_scaling_and_full_filter_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw_data', Path(tmp)/'processed'
            session, source = decoding_fixture(root)
            summary = run(root, output, [session], Config(profile='inseq-decoding'), plots=False)[0]
            folder = output/session
            with np.load(folder/'odor_decoding.npz') as odor, np.load(folder/'position_decoding.npz') as pos, np.load(folder/'labels.npz') as labels, np.load(folder/'scaling.npz') as scale:
                for key in odor.files:
                    if key not in ('y', 'target_name', 'class_names'):
                        np.testing.assert_array_equal(odor[key], pos[key])
                np.testing.assert_array_equal(odor['y'], labels['odor'])
                np.testing.assert_array_equal(pos['y'], labels['position'])
                self.assertTrue((labels['inseq'] == 1).all())
                self.assertIn(0, labels['correct'])  # Incorrect InSeq is eligible.
                self.assertNotIn(1, odor['trial_ids'])  # OutSeq excluded.
                self.assertEqual(odor['X'].shape, (7, 500, 2))
                self.assertEqual(odor['X'].dtype, np.float32)
                train = odor['split'] == 'train'
                np.testing.assert_allclose(odor['X'][train].mean(axis=(0, 1)), 0, atol=1e-6)
                np.testing.assert_allclose(odor['X'][train].std(axis=(0, 1)), 1, atol=1e-6)
                sos, info = notch_design(Config(profile='inseq-decoding'))
                for i in range(len(odor['trial_ids'])):
                    a, b = int(odor['filter_source_start'][i]), int(odor['filter_source_end'][i])
                    lo, hi = {'train': (0, 56000), 'validation': (56000, 68000),
                              'test': (68000, 80000)}[str(odor['split'][i])]
                    self.assertGreaterEqual(a, lo)
                    self.assertLessEqual(b, hi)
                    indices = odor['raw_indices'][i]
                    self.assertGreaterEqual(indices[0], a+info['guard_samples'])
                    self.assertLessEqual(indices[-1]+1, b-info['guard_samples'])
                    filtered = sosfiltfilt(sos, source[:2, a:b], axis=1, padlen=info['guard_samples'])
                    expected = (filtered[:, indices-a].T-scale['mean'][:2])/scale['sd'][:2]
                    np.testing.assert_allclose(odor['X'][i], expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(odor['timestamps_s'], odor['raw_indices']/1000)
            self.assertTrue(summary['target_arrays_identical'])
            self.assertTrue(all(ch['attenuation_db'] < -30 for ch in summary['filter']['channels']))

    def test_gap_nonfinite_jump_and_flatline_context_do_not_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw_data', Path(tmp)/'processed'
            session, _ = decoding_fixture(root, gap=True, artifacts=True)
            summary = run(root, output, [session], Config(profile='inseq-decoding'), plots=False)[0]
            with np.load(output/session/'odor_decoding.npz') as data:
                self.assertTrue(np.isfinite(data['X']).all())
                self.assertNotIn(4, data['trial_ids'])
                for a, b in zip(data['filter_source_start'], data['filter_source_end']):
                    self.assertFalse(a < 32000 < b)
                    self.assertFalse(a <= 30000 < b)
                    self.assertFalse(a <= 42000 < b)
                    self.assertFalse(a < 48200 and b > 48000)
            context = summary['filter']['raw_context_artifact_channel_sample_counts']
            self.assertGreater(context['nonfinite'], 0)
            self.assertGreater(context['flatline'], 0)
            self.assertGreater(summary['exclusions']['filter_edge'], 0)
            with (output/session/'trial_manifest.csv').open() as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 8)

    def test_heldout_perturbation_cannot_change_train_or_fitted_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw_data'
            session, source = decoding_fixture(root)
            config = Config(profile='inseq-decoding')
            first, second = Path(tmp)/'first', Path(tmp)/'second'
            run(root, first, [session], config, plots=False)
            source[:2, 56000:] *= 1.05
            source[0, 70000] = np.nan
            path = root/session/f'{session}_lfp.npz'
            np.savez_compressed(path, data=source, keys=np.array(['one', 'two', 'constant']))
            run(root, second, [session], config, plots=False)
            with np.load(first/session/'odor_decoding.npz') as a, np.load(second/session/'odor_decoding.npz') as b:
                for key in ('X', 'trial_ids', 'raw_indices', 'timestamps_s', 'y'):
                    np.testing.assert_array_equal(a[key][a['split'] == 'train'], b[key][b['split'] == 'train'])
            for filename in ('raw_quality.npz', 'scaling.npz', 'notch_training_psd.npz'):
                with np.load(first/session/filename) as a, np.load(second/session/filename) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])

    def test_position_export_reads_its_own_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw_data', Path(tmp)/'processed'
            session, _ = decoding_fixture(root)
            path = root/session/f'{session}_bvr.npz'
            with np.load(path) as data:
                bvr, keys = data['data'].copy(), data['keys']
            positions = [i for i, key in enumerate(keys) if str(key).startswith('Position')]
            # Deliberately nonstandard synthetic labels catch accidental odor reuse.
            bvr[positions] = np.roll(bvr[positions], 1, axis=0)
            np.savez_compressed(path, data=bvr, keys=keys)
            run(root, output, [session], Config(profile='inseq-decoding'), plots=False)
            with np.load(output/session/'odor_decoding.npz') as odor, np.load(output/session/'position_decoding.npz') as pos:
                np.testing.assert_array_equal(pos['y'], (odor['y']+1)%5)
                np.testing.assert_array_equal(odor['X'], pos['X'])
            summary = json.loads((output/session/'summary.json').read_text())
            self.assertFalse(summary['target_arrays_identical'])

    def test_no_guarded_training_windows_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp)/'raw_data', Path(tmp)/'processed'
            session, _ = decoding_fixture(root)
            with self.assertRaisesRegex(ValueError, 'No training windows'):
                run(root, output, [session], Config(profile='inseq-decoding', notch_q=1e6), plots=False)
            self.assertFalse(output.exists())
