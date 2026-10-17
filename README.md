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

Reconstruct chronological odor trials directly from raw behavior and LFP, with protected train/validation/test blocks, training-only quality rules and scaling, and separate neural/label exports:

```sh
MPLCONFIGDIR=/private/tmp/capstone-matplotlib PYTHONPATH=src python3 -m hippocampal_preprocessing \
  --output data/processed/my-raw-lfp-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Use a new output directory for each run. See [the preprocessing contract and actual-run QA](docs/preprocessing.md) for window assumptions, exclusions, artifact rules, output formats, and the unfinished CEBRA adapter/context contract.
