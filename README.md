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
## Position decoding

Start with [the current notebook](notebooks/position_decoding.ipynb). See [the analysis protocol](docs/position_decoding_protocol.md) and requirements-position.txt for reproduction.

Current verified L1 outputs: results/position_l1_verified_20261110/. Earlier runs are in results/archive/; the historical notebook is in notebooks/archive/. Generated source snapshots and model files inside result folders are audit artifacts.
