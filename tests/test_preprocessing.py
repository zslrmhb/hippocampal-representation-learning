"""Regression checks of source/event/split/fit/export invariants, no models."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hippocampal_preprocessing.context import eligible_centers, pair_indices
from hippocampal_preprocessing.events import reconstruct, time_segments
from hippocampal_preprocessing.pipeline import (Config, partition_trials, quality_and_scale,
                                               run, validate_output, publish_exclusive)


def event_fixture(n=10000, onsets=(1000, 2000, 6500, 7500, 9000)):
    time = np.arange(n)/1000
    events = {f'{kind}{i}': {} for kind in ('Odor', 'Position') for i in range(1, 6)}
    events.update(PokeEvents={}, InSeqLog={}, PerformanceLog={})
    for i, onset in enumerate(onsets):
        events[f'Odor{i%5+1}'][onset] = 1
        events[f'Position{i%5+1}'][onset] = 1
        events['PokeEvents'].update({onset-10: 1, onset+600: -1})
        events['InSeqLog'][onset] = -1 if i%2 else 1
        events['PerformanceLog'][onset] = -1 if i%3 else 1
    return time, events


def write_fixture(root):
    session = 'synthetic'
    folder = root/session
    folder.mkdir(parents=True)
    time, events = event_fixture()
    keys = ['TimeBin']+list(events)+['XvalRatMazePosition', 'YvalRatMazePosition', 'WaterReward']
    behavior = np.zeros((len(keys), len(time)))
    behavior[0] = time
    for row, key in enumerate(keys[1:], 1):
        for index, value in events.get(key, {}).items():
            behavior[row, index] = value
    behavior[-1, 1700] = 1  # Reward AFTER the first exit; preserve full raw event timing.
    source = np.random.default_rng(12).normal(size=(3, len(time)))
    source[2] = 7  # Training-constant channel, same channel mask for heldout.
    np.savez_compressed(folder/f'{session}_bvr.npz', data=behavior, keys=np.array(keys))
    np.savez_compressed(folder/f'{session}_lfp.npz', data=source, keys=np.array(['one','two','constant']))
    return session, source


class EventTests(unittest.TestCase):
    def test_raw_labels_chronological_and_poke_convention(self):
        time, events = event_fixture()
        trials, segments, anomalies = reconstruct(time, events)
        self.assertEqual([t['onset'] for t in trials], [1000, 2000, 6500, 7500, 9000])
        self.assertEqual([t['odor'] for t in trials], list(range(5)))
        self.assertEqual(trials[1]['inseq_raw'], -1)
        self.assertEqual(trials[1]['correct'], 0)
        self.assertEqual(trials[0]['poke_in'], 990)
        self.assertEqual(trials[0]['sampling_duration_ms'], 600)
        self.assertEqual(segments, [(0, 10000)])
        self.assertEqual(anomalies, [])
        self.assertTrue(all(not t['reasons'] for t in trials))

    def test_ambiguous_poke_labels_and_pulses_are_excluded(self):
        time, events = event_fixture()
        events['PokeEvents'][995] = 1
        events['Position2'][1000] = 1
        events['Odor1'][1001] = 1
        trials, _, anomalies = reconstruct(time, events)
        self.assertIn('ambiguous_poke_bracket', trials[0]['reasons'])
        self.assertIn('ambiguous_position', trials[0]['reasons'])
        self.assertIn('odor_not_single_sample_pulse', trials[0]['reasons'])
        self.assertIn('multiple_onsets_in_poke', trials[0]['reasons'])
        self.assertEqual(anomalies[0]['reason'], 'repeated_entry')
        self.assertEqual(len(trials), 6)  # Unexpected pulse retained in manifest.

    def test_unmatched_exit_entry_and_gap_never_pair(self):
        time, events = event_fixture(onsets=(1000,))
        time[1200:] += 10
        events['PokeEvents'][100] = -1
        trials, segments, anomalies = reconstruct(time, events)
        self.assertEqual(len(segments), 2)
        self.assertIn('missing_poke_bracket', trials[0]['reasons'])
        self.assertEqual({a['reason'] for a in anomalies}, {'unmatched_entry','unmatched_exit'})

    def test_nonpositive_and_inconsistent_timestamp_fail(self):
        for kind in ('negative', 'irregular', 'nan'):
            time = np.arange(100)/1000
            if kind == 'negative': time[20] = time[19]
            if kind == 'irregular': time[20:] += .0001
            if kind == 'nan': time[20] = np.nan
            with self.assertRaises(ValueError): time_segments(time)

    def test_half_open_short_withdrawal_and_split_gap_purges(self):
        time, events = event_fixture(onsets=(1000, 2000, 6900, 8200))
        events['PokeEvents'].pop(1600)
        events['PokeEvents'][1500] = -1  # Exactly the exclusive endpoint: kept.
        events['PokeEvents'].pop(2600)
        events['PokeEvents'][2499] = -1
        trials, segments, _ = reconstruct(time, events)
        intervals = partition_trials(trials, segments, len(time), Config())
        self.assertFalse(trials[0]['reasons'])
        self.assertIn('early_withdrawal', trials[1]['reasons'])
        self.assertIn('split_boundary', trials[2]['reasons'])
        self.assertIn('split_boundary', trials[3]['reasons'])
        self.assertEqual(sum(i['end']-i['start'] for i in intervals), len(time))
        time[2300:] += 5
        # Brackets across gap are invalid, even if array indices look continuous.
        trials, segments, _ = reconstruct(time, events)
        partition_trials(trials, segments, len(time), Config())
        self.assertIn('recording_boundary', trials[1]['reasons'])

    def test_delayed_window_respects_onset_split_and_withdrawal(self):
        time, events = event_fixture(onsets=(1000, 2000, 6900))
        events['PokeEvents'].pop(1600)
        events['PokeEvents'][1350] = -1
        events['PokeEvents'].pop(2600)
        events['PokeEvents'][2349] = -1
        time[7100:] += 5
        trials, segments, _ = reconstruct(time, events)
        partition_trials(trials, segments, len(time), Config(window_start_ms=150, window_ms=200))
        self.assertEqual((trials[0]['start'], trials[0]['end']), (1150, 1350))
        self.assertEqual(trials[0]['reasons'], [])
        self.assertIn('early_withdrawal', trials[1]['reasons'])
        self.assertEqual(trials[2]['split'], 'train')
        self.assertIn('split_boundary', trials[2]['reasons'])
        self.assertIn('recording_boundary', trials[2]['reasons'])


class QualityTests(unittest.TestCase):
    def test_unique_overlap_statistics_constant_channel_and_holdout_isolation(self):
        source = np.column_stack([np.arange(30, dtype=float), np.ones(30)*7])
        trials = [dict(start=0,end=10,split='train',trial_id=0,reasons=[]),
                  dict(start=5,end=15,split='train',trial_id=1,reasons=[]),
                  dict(start=20,end=30,split='test',trial_id=2,reasons=[])]
        data = np.stack([source[t['start']:t['end']] for t in trials])
        accepted, scaled, stats = quality_and_scale(data, trials, Config(artifact_z=1000))
        self.assertTrue(accepted.all())
        self.assertEqual(stats['scaling_fit_raw_indices'].tolist(), list(range(15)))
        self.assertAlmostEqual(stats['mean'][0], np.arange(15).mean())
        self.assertAlmostEqual(stats['sd'][0], np.arange(15).std())
        self.assertEqual(stats['channel_mask'].tolist(), [True, False])
        modified = data.copy()
        modified[2, :, 0] *= 10000
        fresh = [dict(t, reasons=[]) for t in trials]
        changed, scaled2, stats2 = quality_and_scale(modified, fresh, Config(artifact_z=1000))
        self.assertFalse(changed[2])
        np.testing.assert_array_equal(scaled[:2], scaled2)
        for key in stats: np.testing.assert_array_equal(stats[key], stats2[key])

    def test_nonfinite_amplitude_and_jump_exclusions(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(6, 500, 2))
        data[3, 200, 0] = np.nan
        data[4, 200, 0] = 1000
        data[5, :, 0] = 0
        data[5, 250:, 0] = 10  # Below amplitude limit, above jump limit with z=5.
        trials = [dict(start=i*1000,end=i*1000+500,split='train' if i<3 else 'test',
                       trial_id=i,reasons=[]) for i in range(6)]
        accepted, _, _ = quality_and_scale(data, trials, Config(artifact_z=5))
        self.assertTrue(accepted[:3].all())
        self.assertIn('nonfinite_lfp', trials[3]['reasons'])
        self.assertIn('amplitude_artifact', trials[4]['reasons'])
        self.assertIn('jump_artifact', trials[5]['reasons'])

    def test_no_valid_train_or_channels_fails(self):
        trials = [dict(start=0,end=500,split='train',trial_id=0,reasons=[])]
        with self.assertRaisesRegex(ValueError, 'nonconstant'):
            quality_and_scale(np.ones((1,500,2)), trials, Config())
        trials[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'training'):
            quality_and_scale(np.ones((1,500,2)), trials, Config())


class ContextTests(unittest.TestCase):
    def test_no_trial_boundary_or_artifact_bridge(self):
        valid = np.ones((2, 12), dtype=bool)
        valid[0, 6] = False
        centers = eligible_centers(valid, 2, 1, 3)
        self.assertFalse(centers[0].any())
        self.assertEqual(np.flatnonzero(centers[1]).tolist(), list(range(2, 8)))
        pairs = pair_indices(valid, 0, 0, -2)
        self.assertTrue((pairs[:,1] >= 2).all())
        for trial, center, positive in pairs:
            self.assertTrue(valid[trial, min(center,positive):max(center,positive)+1].all())
        with self.assertRaises(ValueError): eligible_centers(valid, -1)


class ExportTests(unittest.TestCase):
    def test_raw_only_export_stats_support_and_unlabeled_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw_data'
            session, source = write_fixture(root)
            output = Path(tmp)/'processed'/'run'
            summaries = run(root, output, [session], plots=False)
            folder = output/session
            self.assertEqual(summaries[0]['raw_onsets'], 5)
            with np.load(folder/'odor_trials.npz') as x, np.load(folder/'scaling.npz') as stats, np.load(folder/'cebra_train_sequences.npz') as unlabeled:
                selected = x['split'] == 'train'
                np.testing.assert_array_equal(x['lfp'][selected], unlabeled['lfp'])
                np.testing.assert_array_equal(x['trial_ids'][selected], unlabeled['trial_ids'])
                self.assertFalse({'odor','position','inseq','correct','labels','split'} & set(unlabeled.files))
                ids = stats['scaling_fit_raw_indices']
                np.testing.assert_allclose(stats['mean'][:2], source[:2, ids].mean(axis=1))
                np.testing.assert_allclose(stats['sd'][:2], source[:2, ids].std(axis=1))
                np.testing.assert_allclose(x['lfp'][0], ((source[:2, x['raw_indices'][0]].T-stats['mean'][:2])/stats['sd'][:2]), rtol=1e-6, atol=1e-6)
                self.assertEqual(stats['channel_mask'].tolist(), [True, True, False])
                self.assertFalse(bool(unlabeled['context_contract_selected']))
            with (folder/'trial_manifest.csv').open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 5)
            self.assertEqual(records[0]['onset'], '1000')
            reward = json.loads((folder/'raw_behavior_events.json').read_text())['WaterReward']
            self.assertEqual(reward, [dict(raw_index=1700,time_s=1.7,value=1.0)])
            with self.assertRaises(FileExistsError): run(root, output, [session], plots=False)
            self.assertTrue((folder/'summary.json').exists())

    def test_safety_rejects_sources_ancestors_aliases_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'data'/'raw_data'
            raw.mkdir(parents=True)
            for output in (raw, raw/'bad', raw.parent, raw.parent/'epoched_data'/'bad'):
                with self.assertRaises(ValueError): validate_output(raw, output)
            alias = Path(tmp)/'alias'
            alias.symlink_to(raw, target_is_directory=True)
            with self.assertRaises(ValueError): validate_output(raw, alias/'bad')
            good = Path(tmp)/'data'/'processed'
            good.mkdir()
            with self.assertRaises(FileExistsError): validate_output(raw, good)

    def test_failed_run_has_no_partial_final_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw_data'
            (root/'broken').mkdir(parents=True)
            final = Path(tmp)/'processed'/'failed'
            with self.assertRaises(FileNotFoundError): run(root, final, plots=False)
            self.assertFalse(final.exists())
            self.assertEqual(list(final.parent.iterdir()), [])

    def test_short_windows_export_with_unavailable_line_noise_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw_data'
            session, _ = write_fixture(root)
            output = Path(tmp)/'processed'/'short'
            summaries = run(root, output, [session], Config(window_ms=100), plots=False)
            channel = summaries[0]['channel_quality'][0]
            self.assertIsNone(channel['line_60hz_to_neighbor_power'])
            self.assertEqual(channel['line_noise_status'], 'insufficient frequency resolution')

    def test_atomic_publication_preserves_concurrent_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw_data'
            session, _ = write_fixture(root)
            output = Path(tmp)/'processed'/'run'
            def inject_destination(staging, target):
                target.mkdir()  # Competing writer appears AFTER the exists check.
                publish_exclusive(staging, target)
            with patch('hippocampal_preprocessing.pipeline.publish_exclusive', side_effect=inject_destination):
                with self.assertRaises(FileExistsError):
                    run(root, output, [session], plots=False)
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_bad_configuration_fails_before_output(self):
        for config in (Config(window_ms=0), Config(window_ms=500.1), Config(train_fraction=.9,validation_fraction=.2), Config(left_context=500), Config(artifact_z=float('nan')), Config(positive_offset=0)):
            with self.assertRaises(ValueError): config.validate()
        for start in (-1, 100000.25, float('nan')):
            with self.assertRaises(ValueError):
                Config(window_start_ms=start).validate()


if __name__ == '__main__':
    unittest.main()
