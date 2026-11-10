"""Validation-only model selection and portable coefficient exports.

Coefficients describe associations conditional on other features, not causal
effects. L1 selections can be unstable when channels/time bins are correlated.
"""
from dataclasses import dataclass
from itertools import product
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DecoderConfig:
    seed: int = 42
    c_grid: tuple = (.001, .01, .1, 1., 10.)
    neighbors: tuple = (25, 15, 9, 5, 3, 1)
    weights: tuple = ('uniform', 'distance')
    max_iter: int = 20000
    tolerance: float = 1e-4


def select_decoder(x, y, split, method, config=DecoderConfig()):
    """Fit train only; select by validation macro recall, never test values.

    Deterministic ties favor smaller C (stronger L1), or larger k then uniform
    weights. The selected estimator is NOT refitted on validation data.
    A convergence warning fails the run rather than silently accepting a fit.
    """
    train, validation = split == 'train', split == 'validation'
    if not train.any() or not validation.any():
        raise ValueError('Nonempty train and validation sets required')
    if method == 'logistic':
        settings = [(c, None) for c in config.c_grid]
    elif method == 'knn':
        settings = list(product([k for k in config.neighbors if k <= train.sum()], config.weights))
    else:
        raise ValueError(method)
    best, search = None, []
    for strength_or_neighbors, weighting in settings:
        # scikit-learn >=1.8: l1_ratio=1 specifies a pure L1 penalty.
        model = (LogisticRegression(C=strength_or_neighbors, l1_ratio=1.0, solver='saga',
                 class_weight='balanced', random_state=config.seed,
                 max_iter=config.max_iter, tol=config.tolerance)
                 if method == 'logistic' else
                 KNeighborsClassifier(n_neighbors=strength_or_neighbors, weights=weighting, metric='euclidean'))
        estimator = make_pipeline(StandardScaler(), model)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(x[train], y[train])
        score = balanced_accuracy_score(y[validation], estimator.predict(x[validation]))
        setting = str(strength_or_neighbors) if weighting is None else f'{strength_or_neighbors}/{weighting}'
        search.append(dict(setting=setting, validation_balanced_accuracy=float(score),
                           max_iterations_used=int(np.max(model.n_iter_)) if method == 'logistic' else None))
        if best is None or score > best[0]:
            best = score, estimator, setting
    return (*best, pd.DataFrame(search))


def coefficient_table(decoder, feature_names):
    """Export coefficients in decoder-standardized and input-feature units.

    Input raw features are already channel-normalized upstream, so input-unit
    weights are NOT physical voltage weights. Latent axes lack anatomical meaning.
    """
    scaler, model = decoder.steps[0][1], decoder.steps[-1][1]
    rows = []
    intercept = model.intercept_ - (model.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
    for i, label in enumerate(model.classes_):
        for j, name in enumerate(feature_names):
            rows.append(dict(position=int(label)+1, feature=name,
                             coefficient_standardized=float(model.coef_[i,j]),
                             coefficient_input_units=float(model.coef_[i,j]/scaler.scale_[j]),
                             selected=bool(model.coef_[i,j] != 0),
                             intercept_input_units=float(intercept[i])))
    return pd.DataFrame(rows)

