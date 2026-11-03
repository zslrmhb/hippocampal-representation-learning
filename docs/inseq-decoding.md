# InSeq odor and sequence-position datasets

Use the same cleaned LFP input for two target definitions. Each rat gets two separate files: `odor_decoding.npz` and `position_decoding.npz`. Their `X`, trial IDs, splits, channels, timestamps, validity masks and filter source metadata are exactly equal. Only `y`, `target_name` and `class_names` represent different targets. Odor is A–E; position is ordinal sequence position 1–5, not physical X/Y location. Both targets are encoded 0–4.

In this fixed InSeq sequence, odor and position are paired one-to-one. The observed targets can therefore be numerically identical. These are separate dataset packages, not independent evidence for odor versus position coding. Keep the real labels; do not modify them to create a difference.

## Run

From the repository root, choose a new output directory:

```sh
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --profile inseq-decoding --output data/processed/my-inseq-decoding-run

# Optional one-rat pilot:
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --profile inseq-decoding --sessions 080718_mitt --output data/processed/my-inseq-pilot

PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The original `--profile baseline` remains the default and keeps the unfiltered all-status behavior. Both modes use raw LFP/behavior, preserve original data, refuse existing/protected output paths, and publish a complete run atomically. No model training occurs. Existing `raw-lfp-v2` outputs are preserved.

## Shared preprocessing

1. Reconstruct chronological trials and labels directly from raw behavior. Select InSeq (`InSeqLog == +1`), including incorrect InSeq trials; all raw onsets and exclusions stay in the manifest.
2. Partition the source timeline per rat: 70% train, 15% validation, 15% test by raw sample index, intersected with timestamp segments. These are source blocks, not verified whole-sequence groups or guaranteed class-balanced trial fractions.
3. Fit raw quality thresholds on unique eligible InSeq training-window samples. The existing median/MAD screen (SD fallback, default multiplier 20) removes training-constant channels and rejects trials with retained-channel nonfinite/amplitude/jump violations. These are screening heuristics, not hardware-rail clipping detection or a complete physiological artifact detector.
4. Screen all continuous filter context with those frozen raw thresholds. Exclude nonfinite/amplitude samples, both samples adjoining detected jumps, and exact local flatlines lasting at least 100 ms. Use the union across retained channels to form independent clean blocks. Differences and flatlines never bridge gaps or split boundaries. There is no interpolation/imputation.
5. Filter each block with a second-order 60 Hz IIR notch, Q=30, at the verified 1000 Hz sampling rate. Apply it forward/backward using second-order sections and odd-reflection padding. The single-pass -3 dB bandwidth is 2 Hz; the effective forward/backward response is squared and has different attenuation bandwidth. Save both responses. No automatic harmonic notches, broad bandpass, rereferencing or downsampling are applied.
6. Guard every block edge using `ceil(log(tolerance)/log(max_pole_radius))`, with a minimum of nine samples and default tolerance 1e-6. The default guard is 2199 samples (2199 ms) per edge. Padding uses the same number of samples. This is a relative pole-envelope decay criterion, not an absolute voltage-error guarantee. Synthetic impulse and boundary-sensitivity tests check the implementation.
7. Retain fixed `[odor onset, onset + 500 ms)` windows only if they fit inside a guarded block and the verified poke interval. No time warping, post-withdrawal padding or silent early-withdrawal inclusion. Refit quality rules on eligible filtered InSeq training windows, then compute final per-channel population mean/SD on unique retained training source samples. Freeze those statistics for held-out data and both target exports. A raw-stage removed channel stays removed.

`--notch-hz`, `--notch-q`, `--filter-tolerance` and `--flatline-ms` expose these settings alongside the original window, quality, split and context options. Freeze filter choices using training evidence; do not choose them from final test accuracy. There is no automatic claim that 60 Hz cleaning removes every electrical artifact. Genuine neural activity near 60 Hz is attenuated too.

## Temporal support

An IIR filter has infinite theoretical support. Each exported trial records the *entire independently processed block* as its actual source interval (`filter_source_start`, `filter_source_end`, exclusive end), not a fictional finite support of only the crop plus guard. No block spans partitions, timestamp gaps or detected artifacts.

Neighboring pre-onset, post-withdrawal and OutSeq activity within that block can contribute to a filtered InSeq window. This mode is therefore **offline context-supported decoding**, relaxing the original baseline plan's strict interval-contained processing rule. It does not establish strictly odor-period, odor-delivery-only, pre-decision or causal input. Actual odor offset and acquisition filter/reference/units remain unknown. Encoder context, if later used, must still respect exported trial boundaries; the existing CEBRA compatibility export is not a validated training adapter.

## Files

| File | Contents |
| --- | --- |
| `odor_decoding.npz` | Shared `X` (trials × samples × channels, float32), odor `y`, target/class names, and common metadata |
| `position_decoding.npz` | Exactly the same `X` and common metadata, sequence-position `y` and names |
| `raw_quality.npz` | Frozen raw quality rules, raw channel mask, candidate/retained training fit support |
| `scaling.npz` | Final filtered quality rules, channel mask, mean/SD, exact unique training support |
| `filter_blocks.csv` | Independent filter blocks, split/segment, guarded usable region and usability flag |
| `artifact_spans.csv` | Union of detected invalid context spans, if any |
| `notch_training_psd.npz` | Before/after PSD in original stored units on the same guarded continuous training support; up to four-second Welch windows, no disconnected joins |
| `notch_quality.json`, `notch_quality.png` | Filter settings/guard/context-artifact counts and measured training-spectrum attenuation; before/after spectra and a real training trial |
| `notch_response.npz` | Single-pass and effective forward/backward amplitude responses |
| Other baseline-format files | Trial/split manifests, sparse behavior events, compatibility neural/label/unlabeled arrays, config, summary and clip-based QA |

Common dataset metadata: `trial_ids`, `split`, `channel_keys`, `raw_indices`, `timestamps_s`, `sample_valid`, `sample_rate_hz`, `filter_block_ids`, `filter_source_start`, `filter_source_end`. The neural inputs contain no appended behavior labels. `labels.npz` separately preserves correctness and InSeq status. PSDs in `training_psd.npz` and `notch_training_psd.npz` have different source windows and frequency resolution; the paired notch PSD is the cleaning comparison.

```python
from pathlib import Path
import numpy as np

p = Path('data/processed/inseq-decoding-v1/080718_mitt')
with np.load(p/'odor_decoding.npz') as odor, np.load(p/'position_decoding.npz') as position:
    np.testing.assert_array_equal(odor['X'], position['X'])
    np.testing.assert_array_equal(odor['trial_ids'], position['trial_ids'])
    np.testing.assert_array_equal(odor['split'], position['split'])
    X = odor['X']
    y_odor, y_position = odor['y'], position['y']
    train = odor['split'] == 'train'
    X_train, y_train = X[train], y_odor[train]
```

## Validation

Regression tests cover synthetic 60 Hz attenuation, preservation of 8/30/70 Hz tones, impulse decay and guarded edge sensitivity, exact shared inputs, real target-field selection, training normalization, incorrect-InSeq retention, complete filter-source reconstruction, gap/nonfinite/jump/flatline boundaries, held-out perturbation isolation (features, raw/filtered fitted statistics and training PSD), and failed-run cleanup. The original baseline regression suite remains required.

## Actual run — November 3, 2026

Outputs are in `data/processed/inseq-decoding-v1/`, preceded by `inseq-decoding-pilot-v1/` for Mitt. Each session contains both target datasets.

| Rat | Raw InSeq | Retained | Train / validation / test | Channels |
| --- | ---: | ---: | --- | ---: |
| Mitt | 262 | 249 | 167 / 47 / 35 | 22 |
| Barat | 155 | 154 | 111 / 24 / 19 | 22 |
| Stella | 194 | 190 | 150 / 28 / 12 | 21 |
| Superchris | 210 | 187 | 138 / 31 / 18 | 21 |
| Buchanan | 226 | 208 | 174 / 19 / 15 | 20 |
| Total | 1047 | 988 | 740 / 149 / 99 | 106 |

InSeq exclusions are 13 early withdrawals, 44 filter-edge/context guards and two raw amplitude exclusions. Other exclusion totals in the manifests also include OutSeq trials, so reasons can overlap across the all-onset audit. Seventy-seven incorrect InSeq trials remain eligible. All 106 channels remain. Test sets are small; class counts are recorded in each summary.

At the 60 Hz bin of the paired continuous-training Welch spectra, power attenuation ranged from 30.35 to 47.40 dB across channels. Median attenuation by rat was 44.20, 30.60, 33.29, 40.10 and 31.83 dB respectively. This is measured spectral suppression at 60 Hz, not a claim of complete denoising or improved decoding. Median after/before band-power ratios across channels were about 0.99995–0.99996 for 4–12 Hz, 0.99889–0.99901 for 20–40 Hz and 0.98252–0.98453 for 65–100 Hz. Activity nearer the notch is attenuated more strongly.

All 22 regression tests passed. A separate full-output check verified every target pair's shared arrays, all retained raw labels/timestamps, reconstruction of every retained waveform using the independent direct-form filtering path, unique-source training scaling and complete filter-source containment within partitions. All 186 pre-existing source/output files retained their original SHA256, size and modification time. Both y arrays are numerically identical for all retained real InSeq trials, consistent with the fixed odor/position mapping.

Validation used the available Python 3.11.1, NumPy 1.24.1, SciPy 1.14.1 and Matplotlib 3.6.3. These are older than some declared dependency minimums; the locked project environment was not installed or validated. No models were trained. Original data and previous processed runs were preserved.
