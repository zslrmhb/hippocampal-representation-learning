# Hippocampal Representation Learning

Can hippocampal local field potentials (LFPs) tell us which odor a rat smelled and where it appeared in a learned sequence? We compare direct wavelet features with learned representations across five rats.

## Start here

| I want to… | Go to | What I need |
| --- | --- | --- |
| Rebuild the poster report | [Quick start](#1-poster-rebuild-from-reviewed-aggregates) · [Results notebook](notebooks/poster_results.ipynb) | Python 3.11 and uv; no recordings or checkpoints |
| Build the landscape poster | [Layout and evidence guide](poster/README.md) · [Interpretation notebook](notebooks/poster_interpretation.ipynb) | Chromium for PDF export only |
| Use the standalone poster workbook | [`poster.ipynb`](notebooks/poster.ipynb) · [Deepnote upload guide](poster/README.md#deepnote-working-notebook) | Jupyter Notebook or Deepnote; raw recordings are optional |
| Compare archived models | [Experiment guide](presentation/README.md) · [Decoding notebook](notebooks/presentation_decoding.ipynb) | Included score tables |
| Explore the raw signals | [Recording notebooks](#exploratory-notebooks) | Authorized recordings |
| Recompute the fixed comparison | [Private replay](#private-scientific-replay) | Authorized recordings and frozen CEBRA checkpoints; Linux x86_64 |

**Main finding.** With 500 ms windows after odor onset, direct wavelet amplitude led on sequence position, while odor scores were nearly tied on the four-rat test. This compares complete pipelines, not architectures in isolation. The test was previously viewed; it is descriptive, not independent confirmation.

**How to read the scores.** Balanced accuracy weights each class equally within a rat, then weights rats equally. Validation includes all five rats. The strict test includes four rats because Buchanan is missing one class. The report also shows test recall over observed classes for all five rats; it is not the same metric.

**Targets.** Odor predicts A–E on correct in-sequence trials. Position predicts order (1–5), not physical location, on correct in-sequence and out-of-sequence trials. For example, if C usually appears third but appears fifth, odor is C and position is 5.

## 1. Poster: rebuild from reviewed aggregates

From the repository root, with Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods), build a self-contained HTML report:

```bash
out=results/poster-release-artifact
uv sync --locked
uv run --locked python -B src/evaluation/report.py build \
  --evidence poster/evidence --output "$out"
```

Open `results/poster-release-artifact/poster-results.html` in a browser. For the interactive results, launch Jupyter Notebook from the repository root:

```bash
uv run --locked jupyter notebook notebooks/poster_results.ipynb
```

Select **Python 3 (ipykernel)** and run all cells.

- **Executed notebook file:** in the HTML build command, replace `report.py build` with `report.py notebook`.
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

Use a fresh output directory. Replay must match the recorded validation results before it scores the previously viewed test. `verify` checks the completed run without raw recordings or model fitting. For weights, masking, and latent projection only, add `--interpretability-only` to `run` and skip `verify`.

## Archived decoding experiments

The [experiment guide](presentation/README.md) documents the four-stage ablation and the later encoder–decoder matrix. The [decoding notebook](notebooks/presentation_decoding.ipynb) compares windows, features, and decoders using saved scores. For the later study, see the [matrix scores](poster/evidence/matrix.csv) and [selected settings](presentation/configs/encoder_decoder_selection.json).

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

[`src/data_processing/vendor/`](src/data_processing/vendor/) contains the original project’s recording readers and signal helpers, not third-party code. Their hashes are pinned for replay.

## Exploratory notebooks

From the repository root, run `uv run --locked jupyter notebook`. Put authorized `*_lfp.npz` and `*_bvr.npz` pairs in `data/raw_data/`. Generated caches and figures belong in `results/`.

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
