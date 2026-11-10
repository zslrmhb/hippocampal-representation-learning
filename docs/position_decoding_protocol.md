# Position-decoding analysis protocol

The current entry point is `notebooks/position_decoding.ipynb`.
Historical `notebooks/position_decoding_cebra.ipynb` retains the original L2 experiment and includes a separate summary of the latest L1 results.

## Reproduce

Use the existing CEBRA environment. Add repository `src` to PYTHONPATH.
Generate shared inputs with `python -m hippocampal_preprocessing --profile inseq-decoding --include-outseq --raw-root <raw-data> --output data/processed/position_inseq_outseq_9871456 --no-plots`.
Run the notebook from inside the repository, or call `position_modeling.run(input_directory, new_output_directory)`.
Output directories must be new. Only a run containing COMPLETE is finished.
Run decoder checks with `python -m unittest discover -s tests -p test_position_decoders.py -v`.

## Fixed analysis choices

Five rats analyzed separately; both correct and incorrect InSeq/OutSeq trials eligible. Position labels 0–4 internally, 1–5 in coefficient tables. Odor-aligned [0,500) ms at 1 kHz, 60 Hz notch, training-only quality/scaling and chronological 70/15/15 source blocks. Early withdrawals excluded. Inferred sequence groups crossing splits removed downstream; upstream training statistics stay frozen. Groups are inferred, not verified sequence IDs.

CEBRA uses training trials only: offset10-model, 8 dimensions, 32 hidden units, cosine InfoNCE temperature 1, 10 ms positive lag, 1,000 Adam updates, batch 512, learning rate .0003, seed 42. Temporal context and positive pairs never cross trials. Inference replicates edges within each trial. Ten 50 ms means yield 200–220 raw or 80 latent features.

Primary decoder: multinomial logistic regression, L1 penalty, SAGA, balanced class weights, training-only feature standardization. C grid .001,.01,.1,1,10 selected by validation balanced accuracy; ties favor smaller C. Maximum 20,000 iterations, tolerance .0001; nonconvergence fails. No validation refit. Secondary kNN unchanged: Euclidean distance, k 25,15,9,5,3,1, uniform/distance weights, validation selection.

## Audit outputs

Each run records complete encoder/decoder settings, package versions, SHA256 input/source hashes, modeling source snapshots, retained trial assignments, validation search tables, confusion matrices, per-trial predictions, subgroup coverage, CEBRA loss and sampler details. Encoder state dictionaries and trusted-local decoder joblib files are saved. Do not load untrusted pickle/joblib files. Source snapshots and package versions support reruns; bitwise reproducibility across hardware or dependency versions is not guaranteed.

Coefficient CSVs label class, time bin and channel/latent axis. Standardized coefficients express logit contributions per decoder-feature SD. Input-unit coefficients reconstruct scores before decoder scaling; raw inputs already underwent upstream normalization, so these are not physical-voltage effects. Exact nonzero values define selected features. Correlated predictors can exchange weights, and latent axes are learned coordinates. Selection stability across repeated training-only resamples must be assessed before claiming robust biological importance. L1 is a sparsity choice, not proof of interpretability.

## Reporting limits

Report mean per-rat five-class test balanced accuracy and individual-rat scores. Subgroup balanced accuracy averages only represented classes; always report class coverage and trial count. OutSeq position versus odor-match rates are paired descriptive checks, not a significance test. Only 15 OutSeq test trials exist in the current export. Withdrawal selection, recording drift and behavior may confound results. Previous test results have informed discussion; new analyses are exploratory, not a pristine confirmatory test. A publication needs a predeclared evaluation and uncertainty/stability analysis before strong scientific claims.

