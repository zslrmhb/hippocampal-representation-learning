# Raw LFP preprocessing

The original baseline is documented below. For the newer InSeq-only datasets with shared notch-cleaned X and separate odor/sequence-position targets, use `--profile inseq-decoding`; see [InSeq decoding datasets](inseq-decoding.md).

This pipeline prepares matched inputs for direct-voltage odor decoding and a future self-supervised representation comparison. It reads only `<session>_lfp.npz` and `<session>_bvr.npz` from the raw recording directory. It does not read the supplied epochs, external alignment tables, spikes, or research notes, and does not fit a model.

From the repository root:

```sh
# Mitt pilot; choose a new directory for every run.
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --sessions 080718_mitt --output data/processed/my-mitt-pilot

# All sessions.
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --output data/processed/my-raw-lfp-run

PYTHONPATH=src python3 -m unittest discover -s tests -v
```

NumPy and SciPy are required; Matplotlib is needed for the default QA PNGs (`--no-plots` disables PNGs). No additional dependencies were introduced. The first real run used Python 3.11.1, NumPy 1.24.1, SciPy 1.14.1 and Matplotlib 3.6.3. These are older than the repository's declared dependency minimums; the regression suite and real run passed in that environment, but the locked project environment was not installed or validated. Standard repository setup remains `uv sync`.

## Defaults and scientific interpretation

| Setting | Behavior |
| --- | --- |
| Signal | Supplied broadband voltage at 1000 Hz; no new filter, notch, rereference, downsampling, interpolation or per-trial normalization |
| Window | Fixed `[odor_onset, odor_onset + 500 samples)`; early withdrawals are excluded, never padded or time warped |
| Split | Per-session contiguous source-index blocks: first 70% train, next 15% validation, final 15% test; a crossing trial is purged |
| Quality | Thresholds fitted only on unique eligible training samples; reject a whole trial on any retained-channel nonfinite value or threshold violation |
| Scaling | Per-channel population mean/SD on unique retained training source samples, frozen for held-out trials; float32 output |
| Context | Left/right context default 0 is explicitly unselected; illustrative temporal-positive offset is 1 sample (1 ms), not a finalized CEBRA configuration |

The window is an **odor-sampling proxy**. Actual odor offset is unknown. The one-sample odor pulses are onset markers; returning to zero does not establish offset. Acquisition filtering, reference and physical voltage units are also unknown. No pre-decision or odor-delivery-only claim follows from this export.

`--window-ms`, `--artifact-z`, `--train-fraction`, `--validation-fraction`, `--sample-rate-hz`, `--left-context`, `--right-context` and `--positive-offset` expose the configuration. A window must contain an integer number of samples. TimeBin must verify the requested sample rate inside each segment. Different window/context settings are separate, explicit runs; no automatic duration choice or validation/test tuning occurs. The QA reports 300/500 ms withdrawal coverage irrespective of the selected window.

## Event reconstruction and source ownership

Every chronological raw odor onset appears in `trial_manifest.csv`, including excluded trials. Odor and position use 0–4 (A–E and positions 1–5); raw signed status markers are also retained. Correctness comes from PerformanceLog, not a newly inferred duration rule. Incorrect, OutSeq and first-position trials are initially eligible.

A valid trial has exactly one +1 odor pulse and one +1 position marker at onset, signed ±1 sequence/performance markers, and a single +1 poke-entry / -1 exit bracket within one timestamp segment. Repeated entries, missing exits, multiple odor onsets inside a poke, ambiguous labels and non-pulse odors receive explicit exclusion reasons. Timestamp gaps separate segments; nonfinite, nonpositive or irregular within-segment timing fails the session instead of guessing. Intervals use zero-based indices and exclusive ends. A 500-sample window ending exactly at withdrawal is valid.

Whole-sequence reconstruction has not been validated. The fallback therefore assigns the *entire source recording* to chronological blocks before quality fitting, rather than claiming sequence grouping or stratification. `split_manifest.csv` covers the source segments with exact split intervals and clock times. Actual trial windows must be wholly inside both their segment and their split. Any retained overlap is necessarily within a single split; unique source samples are counted once when fitting. Clock-time gaps mean these index fractions are not fractions of elapsed wall time, and trial fractions/classes need not match 70/15/15. No retained trials in the first all-session run crossed a split boundary.

## Quality policy

For each channel, fit the median and `1.4826 × median absolute deviation` on unique candidate training samples. When MAD is zero, use population SD as an explicit fallback. Fit the same robust statistics to adjacent differences only when their source indices are truly consecutive; unrelated trial boundaries are never differenced. The amplitude threshold is `abs(value - median) > artifact_z × robust_scale`, and the jump threshold applies the analogous rule to within-trial differences. The default multiplier is a conservative 20. It is a reproducible screening heuristic, not a validated physiological artifact detector or a clipping detector tied to unknown hardware rails.

Channels with no finite training samples or no training variation are removed for every split. Any nonfinite value or amplitude/jump violation on a retained channel excludes the whole trial. No valid values are imputed. Channel statistics are then fitted again using unique retained training samples. `scaling.npz` records both the quality-threshold and final scaling fit IDs and exact source support, all channel masks and thresholds. Any channel made constant after exclusions is removed too.

Training-only candidate rejection counts for multipliers 10/15/20 are recorded for sensitivity review. Mitt's counts were 73/3/0: a much lower multiplier would reject many otherwise eligible clips. Default 20 was fixed before the all-session run and was not chosen using held-out performance. In Superchris it rejects two training trials and one validation trial; other sessions have no quality exclusions. This permissive screen does not establish artifact-free recordings.

`training_psd.npz` averages Welch spectra computed independently within retained training clips. `channel_quality.json` includes a 60 Hz neighborhood diagnostic, not an automatic cleaning decision. Short windows without suitable frequency bins report a null ratio and insufficient resolution. `quality.png` shows the first retained training trial before/after scaling and all retained-channel mean training PSDs. At 500 ms the PSD spacing is 2 Hz. Some sessions have clear 60 Hz peaks; selecting a notch or restricted passband would require a separately justified comparison with shared settings for both models.

## Artifacts and consumption

Each session output contains:

| File | Contract |
| --- | --- |
| `trial_manifest.csv` | All raw onsets, labels, status, poke times/duration, onset XY, windows, split, reasons and retention; actual odor offset is blank |
| `raw_behavior_events.json` | All sparse raw odor/position/status/poke/reward markers with raw indices and TimeBin; keeps rewards outside poke intervals too; separate from neural input |
| `split_manifest.csv` | Entire source interval ownership by segment/split, zero-based exclusive ends and TimeBin clock values |
| `odor_trials.npz` | `lfp`: retained trials × samples × retained channels; `sample_valid`, `valid_centers`, `valid_pair_centers`, `raw_indices`, `timestamps_s`, `trial_ids`, `split`, `channel_keys`, rate |
| `labels.npz` | Trial IDs and odor/position/InSeq/correct labels, aligned by ID with `odor_trials.npz` |
| `cebra_train_sequences.npz` | Exact train subset of the same LFP, masks, IDs/times/source indices, separate trial boundaries, context metadata; no odor or behavior labels and no held-out activity |
| `scaling.npz` | Frozen statistics, thresholds, channel mask, exact unique fit support/IDs and training-only threshold sensitivity counts |
| `preprocessing_config.json`, `summary.json` | Resolved policy, provenance, exclusions, channel reasons, before/after class/status/position/split counts and 300/500 ms coverage |
| `preprocessing_qa.md`, `channel_quality.json`, `training_psd.npz`, `quality.png` | QA findings and diagnostics (`quality.png` omitted with `--no-plots`) |

The run root contains `run_manifest.json` (source paths/sizes, configuration and runtime versions) and combined QA. Datasets/generated files remain ignored by Git. Existing output paths, source trees, source ancestors and symlink aliases into protected trees are refused. Processing stages results beside the requested output and publishes only after every session succeeds using an atomic exclusive rename (macOS/Linux/Windows; other unsupported platforms fail closed); failures remove the staging directory. Source files are never opened for writing. Use a new version/run directory to retain a previous result.

A minimal load preserves row alignment and separates labels from pretraining:

```python
from pathlib import Path
import numpy as np

p = Path('data/processed/raw-lfp-v2/080718_mitt')
with np.load(p/'odor_trials.npz') as neural, np.load(p/'labels.npz') as labels:
    np.testing.assert_array_equal(neural['trial_ids'], labels['trial_ids'])
    X = neural['lfp']                # (282 trials, 500 samples, 22 channels) for Mitt
    y = labels['odor']               # A–E encoded 0–4
    train = neural['split'] == 'train'
    X_train, y_train = X[train], y[train]  # Supervised decoder inputs, 188 Mitt trials

with np.load(p/'cebra_train_sequences.npz') as unlabeled:
    pretraining_trials = unlabeled['lfp']  # Same train voltage, no behavior labels
    assert not bool(unlabeled['context_contract_selected'])  # Default export needs model contract
```

The exported voltage windows and channel mask are the common starting point for both decoder branches. The initial default exports whole valid trials, so `sample_valid` is all true; exclusions remain in the manifest rather than concealed as padding.

**CEBRA integration remains modeling work.** The artifact keeps a trial axis. Its `offsets`/`lengths` describe boundaries if a downstream representation uses flattened storage; offsets do not authorize flattening into a continuous time series. Default pair masks are illustrative: `context_contract_selected=False` and `pair_contract_status` explicitly state that an encoder and adapter remain to be verified. Do not feed concatenated trials into an unverified CEBRA adapter or train on validation/test arrays.

Select the encoder's actual left/right receptive field and temporal-positive offset, then use the boundary utility or rerun with those settings:

```python
import numpy as np
from hippocampal_preprocessing import eligible_centers, pair_indices

with np.load('data/processed/my-raw-lfp-run/080718_mitt/cebra_train_sequences.npz') as x:
    # Example only: replace these three settings with the selected model's contract.
    centers = eligible_centers(x['sample_valid'], left=10, right=10)
    pairs = pair_indices(x['sample_valid'], left=10, right=10, offset=20)
    # Each pair is (trial index, center sample, positive sample) in one trial.
```

The utility requires both full contexts and the interval connecting a pair to be valid inside the same trial. It never creates cross-trial positives. Temporal-positive offsets must be nonzero; `eligible_centers(..., offset=0)` is for individual encoder centers, not self-pair training. An adapter must enforce these constraints for all sampling and embedding operations. Direct voltage and latent decoders must use the corresponding source support, valid centers, trial eligibility and aggregation. Context trimming must not silently give either branch extra pre-onset/post-withdrawal data or different trials.

## First actual run — October 12, 2026

Outputs: `data/processed/raw-lfp-v2/` (all sessions), preceded by `data/processed/raw-lfp-pilot-v1/` (Mitt pilot). The pilot and v1 precede the complete sparse reward-event export; use the all-session run for modeling handoff.

| Session | Raw trials | Retained | Train / validation / test | Early withdrawal | Quality trials | OutSeq retained / raw |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Mitt | 292 | 282 | 188 / 55 / 39 | 10 | 0 | 30 / 30 |
| Barat | 176 | 176 | 125 / 28 / 23 | 0 | 0 | 21 / 21 |
| Stella | 222 | 210 | 165 / 32 / 13 | 12 | 0 | 18 / 28 |
| Superchris | 240 | 233 | 175 / 35 / 23 | 4 | 3 | 25 / 30 |
| Buchanan | 270 | 269 | 223 / 26 / 20 | 1 | 0 | 44 / 44 |
| Total | 1200 | 1170 | 876 / 176 / 118 | 27 | 3 | 138 / 153 |

All 106 recording channels remain. Superchris has three amplitude violations, one of which also violates the jump rule; reason counts therefore overlap. No event/poke anomalies or split/gap boundary exclusions occurred. At 300 ms, 1188 trials have withdrawal coverage versus 1173 at 500 ms; these are timing coverage counts before quality screening, not a second processed/model-tested run. The default is deliberately unchanged.

Exclusions alter behavior balance: Stella loses 10/28 OutSeq trials, whereas Mitt's 10 early withdrawals are incorrect InSeq trials. Report that bias in downstream decoding. Per-split counts are small, especially Stella's 13 test trials; a chronological split does not guarantee class or status coverage. Odor and position remain strongly confounded on InSeq trials. This preprocessing cannot establish odor-specific coding, pre-decision prediction, improved representation quality or cross-rat channel correspondence.

Regression coverage includes raw label reconstruction, ambiguous poke/pulse events, gaps and half-open window boundaries, split purges, unique overlap statistics, constant/nonfinite/artifact handling, held-out perturbation isolation, no temporal/context crossing, source/output protection, failed-run cleanup, short-window diagnostics and training-only unlabeled export. No models were trained and no original datasets were repaired.
