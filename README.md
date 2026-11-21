# Hippocampal Representation Learning

This project asks whether hippocampal local field potentials (LFPs) can predict which odor a rat smelled and where it appeared in a learned sequence. It compares direct wavelet features with learned representations across five rats. The repository contains the methods, reviewed results, and figure-building workflows; no slide deck is required.

## Start here

| I want to… | Go to | What I need |
| --- | --- | --- |
| View or rebuild the final poster results | [Quick start](#1-poster-rebuild-from-reviewed-aggregates) · [Poster notebook](notebooks/poster_results.ipynb) | Python 3.11 and uv; no recordings or checkpoints |
| Build the landscape poster or inspect its all-rat interpretation | [Poster layout and evidence map](poster/README.md) · [Interpretation notebook](notebooks/poster_interpretation.ipynb) | Checked-in aggregates; Chromium for PDF export |
| Put the poster materials in Deepnote or Jupyter | [Standalone `poster.ipynb`](notebooks/poster.ipynb) · [Upload instructions](poster/README.md#deepnote-working-notebook) | NumPy, pandas, SciPy, and Plotly 6; raw recordings and print export are optional |
| Explore the archived decoding comparisons | [Experiment guide](presentation/README.md) · [Decoding notebook](notebooks/presentation_decoding.ipynb) | The checked-in model score aggregates |
| Explore the recordings or regenerate preprocessing figures | [Exploratory notebooks](#exploratory-notebooks) | Authorized raw recordings |
| Recompute the fixed poster comparison from source data | [Private scientific replay](#private-scientific-replay) | Authorized recordings and frozen CEBRA checkpoints; Linux x86_64 |
| Inspect the historical experiment settings | [Ablation protocol](presentation/configs/ablation.json) · [Encoder–decoder matrix protocol](presentation/configs/encoder_decoder_matrix.json) | Nothing to install; the search and ablation runners are not included |

**Main finding.** With 500 ms windows after odor onset, direct wavelet amplitude led on sequence position, while odor scores were nearly tied on the four-rat test. This compares complete pipelines, not architectures in isolation. The test was previously viewed; it is descriptive, not independent confirmation.

**How to read the scores.** Balanced accuracy gives each class equal weight within a rat, then weights the rats equally. The landscape poster shows strict five-class test scores for four rats because Buchanan lacks one test class, and validation scores for all five. The detailed report also shows test recall averaged over the classes observed in each of the five rats; it is a different metric.

**Targets.** Odor predicts A–E on correct in-sequence trials. Position predicts order (1–5), not physical location, on correct in-sequence and out-of-sequence trials. For example, if C usually appears third but appears fifth, odor is C and position is 5.

## 1. Poster: rebuild from reviewed aggregates

From the repository root, with Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods), build a self-contained HTML report:

```bash
out=results/poster-release-artifact
uv sync --locked
uv run --locked python -B src/evaluation/report.py build \
  --evidence poster/evidence --output "$out"
```

Open `results/poster-release-artifact/poster-results.html` in a browser. To explore the figures interactively, run `uv run --locked jupyter lab`, open the [poster notebook](notebooks/poster_results.ipynb), select **Python 3 (ipykernel)**, and run all cells.

- **Executed notebook:** replace `build` with `notebook` in the command above.
- **Detailed report PDF:** run `uv run --locked playwright install chromium`, then add `--pdf` to the `build` command.
- **Single-page landscape poster and wireframe:** follow the [poster guide](poster/README.md) instead.

These routes render reviewed aggregates only: no recordings, checkpoints, PyTorch, or CUDA.

### Private scientific replay

Replay recomputes the fixed 500 ms comparison and the training/validation interpretability panels (L1 coefficients, grouped input masking, and CEBRA latent projection). It refits the fixed L1 heads but does not retrain CEBRA, search hyperparameters, or rerun the historical ablation or later encoder–decoder matrix. [`poster/replay-code.json`](poster/replay-code.json) pins the implementation and binds the [reviewed input and result specification](poster/evidence/replay-spec.json).

```bash
out=results/poster-release-artifact/replay
uv sync --locked --group replay --python 3.11.18
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src"
uv run --locked --group replay --python 3.11.18 python -B -m evaluation.replay run \
  --spec poster/replay-code.json --raw /path/to/authorized/raw_data \
  --checkpoints /path/to/authorized/encoders --output "$out"
uv run --locked --group replay --python 3.11.18 python -B -m evaluation.replay verify \
  --spec poster/replay-code.json --output "$out"
```

Use a fresh output directory. The full run checks exact validation parity before scoring the previously viewed test; `verify` checks a completed replay without reopening raw recordings or fitting models. For weights, masking, and latent projection only, add `--interpretability-only` to `run` and skip the full-receipt `verify` command.

## Archived decoding experiments

The [experiment guide](presentation/README.md) explains the four-stage ablation and the separate, later encoder–decoder matrix. For the ablation results, open the [decoding notebook](notebooks/presentation_decoding.ipynb) to compare decoders, time windows, and CEBRA settings. For the later experiment, see the [matrix results](poster/evidence/matrix.csv) and [frozen decoder settings](presentation/configs/encoder_decoder_selection.json). These are saved results, not new test runs.

## Repository layout

| Location | Contents |
| --- | --- |
| [`poster/`](poster/) | Reviewed evidence, replay pins, all-rat interpretation, and landscape poster guide |
| [`presentation/`](presentation/) | Archived decoding scores, experiment protocols, and reproduction guide |
| [`src/data_processing/`](src/data_processing/) | Recording reads, behavior alignment, signal quality checks, filtering, and saved intermediates |
| [`src/modeling/`](src/modeling/) | Amplitude features, frozen CEBRA inference, and training-only sparse logistic regression |
| [`src/evaluation/`](src/evaluation/) | Scores, actual versus usual position counts, interpretation, rendering, and replay |
| [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) | One repository-wide environment; `replay` adds the Linux x86_64 scientific dependencies |
| [`notebooks/`](notebooks/) | Poster figures, archived decoding results, preprocessing, and exploratory recording analyses |
| [`tests/`](tests/) | Evidence integrity and scientific replay gate tests |
| `results/` | Ignored local output; not part of the release |

[`src/data_processing/vendor/`](src/data_processing/vendor/) contains original in-repo recording, event, filtering, and QC helpers, not third-party dependencies. Keeping these files together preserves the reviewed preprocessing fingerprints.

## Exploratory notebooks

Launch JupyterLab as above. Place authorized `*_lfp.npz` and `*_bvr.npz` pairs under `data/raw_data/`; if an authorized Git LFS copy is published separately, use `git lfs pull` in that checkout. Keep derived caches and figures in ignored `results/`.

| Notebook | What it covers | Recordings |
| --- | --- | --- |
| [Dataset overview](notebooks/dataset_eda.ipynb) | Channels, behavioral markers, and timestamp gaps | Mitt |
| [Brain activity](notebooks/brain_activity_eda.ipynb) | Poke-aligned LFP by odor and channel; accuracy by recorded sample time. Set `pool_incorrect = True` for the pooled view. | Mitt |
| [Recording preprocessing](notebooks/presentation_preprocessing.ipynb) | Trial alignment and rejection, normalization, 60 Hz spectrum and notch filter, wavelet and signal-quality heatmaps | All five rats |

For the [raw-versus-epoched audit](src/data_processing/audit_recordings.py), place both recording formats under `data/raw_data/` and `data/epoched_data/`, respectively. Run `PYTHONPATH=src uv run --locked python -B -m data_processing.audit_recordings`; outputs go to `results/recording-audit/`.

## Verification

From the repository root:

```bash
uv run --locked python -B -m unittest discover -s tests/poster
uv run --locked --group replay --python 3.11.18 python -B -m unittest discover -s tests/scientific
```

## Data availability, references, and license

The [MIT License](LICENSE) covers the code. This release omits raw recordings, processed data, checkpoints, and trial-level predictions. Local recordings are ignored by default; Git LFS is configured for a future release of authorized `.npz` files under `data/raw_data/`.

The [Dryad deposit](https://doi.org/10.7280/D14X30) is licensed CC0, but the local `.npz` files have not been verified against it. Request any missing authorized replay inputs from the data custodian.

- **Recordings and task:** [Shahbaba et al. (2022)](https://doi.org/10.1038/s41467-022-28057-6).
- **CEBRA:** [Schneider, Lee, and Mathis (2023)](https://doi.org/10.1038/s41586-023-06031-6).
