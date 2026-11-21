# Landscape poster guide

The **wireframe** shows which evidence belongs in each panel. The **results** version fills the same layout with reviewed scores and a separate all-rat validation interpretation. Both are standalone HTML and single-page **A3 landscape** PDFs. The detailed [poster results notebook](../notebooks/poster_results.ipynb) remains the reference for the full reviewed comparison; the [historical presentation](../presentation/README.md) is a separate reproduction track.

The study design sits at left, the test comparison and its limits anchor the center, and interpretation sits at right. The printed poster shows the clearest masking comparison; the HTML supplement preserves every group and all ten CEBRA-Time panels. The hierarchy and figure labels follow [MIT Communication Lab's poster guidance](https://mitcommlab.mit.edu/aeroastro/commkit/research-posters/) and [figure guidance](https://mitcommlab.mit.edu/aeroastro/commkit/figure-design/), and the focus on a single takeaway follows [Erren and Bourne's poster recommendations](https://doi.org/10.1371/journal.pcbi.0030102).

From the repository root:

```bash
uv sync --locked
uv run --locked playwright install chromium
PYTHONPATH=src uv run --locked python -B -m evaluation.poster_layout
```

Outputs: `results/poster-landscape/poster-{wireframe,results}.{html,pdf}`. HTML evidence links open local checkout files; PDF evidence links point to the GitHub default branch and require those files to be published there. The renderer and layout are in [`src/evaluation/`](../src/evaluation/): [`poster_layout.py`](../src/evaluation/poster_layout.py) and [`poster_landscape.html.jinja`](../src/evaluation/poster_landscape.html.jinja).

## Deepnote working notebook

Upload [`poster.ipynb`](../notebooks/poster.ipynb) beside a folder named `poster/` containing the authorized session directories and raw NPZ files. For the data upload, use the contents of `data/raw_data/`, not this evidence directory. The notebook is standalone and starts with a plain-language guide; Section 08 maps each poster panel to its output. It requires NumPy, pandas, SciPy, and Plotly 6. The reviewed aggregates are embedded; raw files are optional and only needed for the signal illustrations.

The workbook uses the Week 8 presentation’s mint-and-teal theme, with white chart panels. In the masking figures, red still means performance dropped.

Run all cells, then download `poster_assets.zip`. Extract the whole folder for offline interactive HTML (the figures share a JavaScript file) and the CSV tables. For SVG, PDF, and high-resolution PNG, install `kaleido>=1.1,<2` and provide Chrome as described in the first cell. The notebook does not require any other project files.

## Panel evidence

| Panel | Evidence | Code and notebook |
| --- | --- | --- |
| 01 · Question and methods | Split, signal-stage and feature checks in [`evidence/replay-spec.json`](evidence/replay-spec.json) | [Signal preprocessing](../src/data_processing/preprocess.py), [fixed model design](../src/modeling/design.py), [preprocessing notebook](../notebooks/presentation_preprocessing.ipynb) |
| 02 · Main result | [`session_scores.csv`](evidence/session_scores.csv): five-class balanced accuracy, test first (same four rats), then validation (all five). Five-rat test observed-class recall is in [`headline.csv`](evidence/headline.csv). | [Poster results notebook](../notebooks/poster_results.ipynb), [metric code](../src/evaluation/metrics.py) |
| 03 · Actual vs usual position | Test and validation rows of [`outseq.csv`](evidence/outseq.csv); per-rat denominators are in the same file | [Metric code](../src/evaluation/metrics.py), [poster results notebook](../notebooks/poster_results.ipynb) |
| 04 · Validation masking | [`interpretation-all-rats.json`](interpretation-all-rats.json), `masking`; cross-checked against [`l1_masking.csv`](evidence/l1_masking.csv). The print poster shows 4–11 Hz; the HTML supplement shows all groups. | [Masking implementation](../src/evaluation/interpretability.py), [interpretation notebook](../notebooks/poster_interpretation.ipynb) |
| 05 · Training coefficients | [`interpretation-all-rats.json`](interpretation-all-rats.json), `weights` | [Validation export](../src/evaluation/poster_interpretation.py), [interpretation notebook](../notebooks/poster_interpretation.ipynb) |
| HTML supplement · CEBRA-Time projections | [`interpretation-all-rats.json`](interpretation-all-rats.json), `latents` | [Validation export](../src/evaluation/poster_interpretation.py), [interpretation notebook](../notebooks/poster_interpretation.ipynb) |

The previously viewed test comes first when a matched score exists. Buchanan lacks a test class, so the strict five-class test summary uses the same four rats for both models and targets; all five contribute to validation. The five-rat test observed-class recall answers a different question and stays in the linked results notebook. Grouped masking and latent projections remain validation-only; training coefficients have no validation/test counterpart. No models were reapplied to the test set for this poster update.

## Regenerate the all-rat interpretation

The export covers **five rats × two targets**. It verifies the replay’s validation receipt, recomputes validation features from the guarded cache and authorized checkpoints, refits the frozen heads on training rows, and checks all 20 validation feature and prediction digests. It does not score or open the test prediction artifacts. The original evidence packet in `poster/evidence/` is unchanged.

- **Weights:** each cube is class (A–E or position 1–5) × ten 50 ms bins × twelve Morlet frequencies (ascending 4–150 Hz). `signed` and `magnitude` average class-centered coefficients over channels within a rat; `nonzero` counts the original nonzero L1 coefficients before centering. Do not pool anatomically by tetrode number. The poster uses mean magnitude across rats and classes; the notebook keeps sign and sparsity visible.
- **Masking:** mean replacement in fitted head units; readout stays fixed. The landscape poster and interpretation notebook show masked minus intact validation balanced accuracy (negative means worse). The source files retain the original intact-minus-masked sign. This is sensitivity, not causal importance.
- **Latents:** train-fit two-component PCA per rat and target. Export validation centroids and support for all trials and OutSeq where available, grouped by odor and position. Independent axes and limited support make the projection descriptive; use the validation OutSeq counts separately for the prediction diagnostic.

First run the [private replay](../README.md#private-scientific-replay) with `--interpretability-only`. Then use its cache and the same checkpoint root:

```bash
PYTHONPATH=src uv run --locked --group replay --python 3.11.18 python -B \
  -m evaluation.poster_interpretation \
  --cache results/poster-release-artifact/replay \
  --checkpoints /path/to/authorized/encoders \
  --spec poster/replay-code.json \
  --output results/poster-interpretation-all-rats.json
```

The distributed supplement was regenerated against the reviewed replay specification; a new cache can have different receipt metadata without changing the aggregate values.
