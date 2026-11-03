# Hippocampal Representation Learning

## Getting Started

### Requirements
- [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

### Setup:
1. Clone the repo.
2. Run "uv sync" in the root of repo.

Use "uv add \<package>" to add new dependencies.

### Repository Structure
- data/  (local dataset, not tracked by git)
    - raw_data
    - epoched_data
    - ... (processed data?)
- docs/ (any related files, literature, not tracked)
- notebooks/  (jupyter notebooks)
- src/  (reusable Python code)
- results/ (generated figures, metrics, model outputs)
## Raw LFP preprocessing

For the InSeq odor and sequence-position tasks, generate two datasets per rat with exactly the same cleaned `X` and shared trial/split metadata, and different target definitions:

```sh
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --profile inseq-decoding --output data/processed/my-inseq-decoding-run
```

This mode applies a 60 Hz notch within independent split-owned, artifact-free recording blocks, extracts fixed 0–500 ms windows, and fits channel scaling on retained InSeq training data. It exports `odor_decoding.npz` and `position_decoding.npz` per rat. See [the InSeq dataset contract](docs/inseq-decoding.md) for loading, filter QA and temporal-support limits. On the fixed InSeq sequence, odor and position labels coincide; the two exports do not disentangle the underlying codes.

The original unfiltered baseline remains available:

Reconstruct chronological odor trials directly from raw behavior and LFP, with protected train/validation/test blocks, training-only quality rules and scaling, and separate neural/label exports:

```sh
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --output data/processed/my-raw-lfp-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Use a new output directory for each run. See [the preprocessing contract and actual-run QA](docs/preprocessing.md) for window assumptions, exclusions, artifact rules, output formats, and the unfinished CEBRA adapter/context contract.
