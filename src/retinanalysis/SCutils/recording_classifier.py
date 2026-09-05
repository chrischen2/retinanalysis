"""Block-level classifier for cell-attached versus whole-cell recordings.

The model is trained only from epoch groups with an explicit
``recordingTechnique`` label. Multiple raw epochs are concatenated into one
feature vector per epoch block, while validation splits keep every block from
the same cell together. This prevents a classifier from appearing accurate by
seeing another block from the validation cell during training.

The checked-in model bundle also stores out-of-fold probabilities for its
labelled training blocks. Those probabilities, rather than in-sample fitted
values, are used when auditing and retroactively correcting recorded labels.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd


MODEL_VERSION = 1
DEFAULT_MODEL_PATH = Path(__file__).with_name(
    'recording_technique_classifier.joblib')
DEFAULT_REPORT_PATH = Path(__file__).with_name(
    'recording_technique_classifier_report.json')
DEFAULT_TRAINING_FEATURES_PATH = Path(__file__).with_name(
    'recording_technique_classifier_training.csv')
DEFAULT_MIN_CONFIDENCE = 0.90
DEFAULT_N_TRIALS = 6
DEFAULT_TRACE_SECONDS = 5.0

FEATURE_NAMES = (
    'raw_mean', 'between_epoch_mean_sd', 'log_sd', 'log_mad', 'log_iqr',
    'log_range_99', 'log_abs_q999', 'log_diff_sd', 'log_diff_q99',
    'log_diff_q999', 'log_ddiff_sd', 'crest_factor', 'diff_crest_factor',
    'skew', 'kurtosis', 'power_1_50', 'power_50_200',
    'power_200_1000', 'power_1000_4500', 'spectral_centroid_hz',
    'lag1_corr',
)


def _technique_family(value) -> str:
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return ''
    text = str(value).strip().lower().replace('_', '-').replace(' ', '-')
    if text in ('cell-attached', 'cellattached'):
        return 'cell-attached'
    if text in ('whole-cell', 'wholecell'):
        return 'whole-cell'
    return ''


def _log_positive(value: float) -> float:
    return float(np.log10(max(float(value), 1e-12)))


def extract_recording_block_features(amp_data, sample_rate: float) -> Dict[str, float]:
    """One protocol-agnostic feature vector from several epochs of one block.

    Epochs are individually median-centered for shape and spectral features,
    then concatenated. Absolute holding current and between-epoch mean spread
    remain separate features. The representation therefore captures narrow
    spikes versus broad currents without depending on stimulus alignment or a
    fixed epoch duration.
    """
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew

    data = np.asarray(amp_data, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or not data.size or data.shape[1] < 3:
        raise ValueError('amp_data must contain at least one trace with 3 samples')
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('sample_rate must be positive')

    epoch_means = np.nanmean(data, axis=1)
    centered = data - np.nanmedian(data, axis=1, keepdims=True)
    flat = centered.ravel()
    flat = flat[np.isfinite(flat)]
    differences = np.diff(centered, axis=1).ravel()
    differences = differences[np.isfinite(differences)]
    second_differences = np.diff(centered, n=2, axis=1).ravel()
    second_differences = second_differences[np.isfinite(second_differences)]
    if not len(flat) or not len(differences):
        raise ValueError('amp_data contains no finite trace differences')

    absolute = np.abs(flat)
    absolute_diff = np.abs(differences)
    standard_deviation = float(np.std(flat))
    difference_sd = float(np.std(differences))

    spectra = []
    frequencies = None
    for trace in centered:
        finite = np.asarray(trace[np.isfinite(trace)], dtype=float)
        if len(finite) < 3:
            continue
        frequencies, power = welch(
            finite, fs=sample_rate, nperseg=min(4096, len(finite)))
        spectra.append(power)
    if not spectra or frequencies is None:
        raise ValueError('amp_data contains no trace long enough for a spectrum')
    power = np.mean(spectra, axis=0)
    upper = min(sample_rate / 2, 4500.0)
    usable = (frequencies >= 1) & (frequencies <= upper)
    total_power = float(power[usable].sum())

    def quantile(values, probability):
        return float(np.quantile(values, probability))

    def band_power(low, high):
        selected = (frequencies >= low) & (frequencies < min(high, upper))
        return float(power[selected].sum() / total_power) if total_power else 0.0

    lag1 = (float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
            if len(flat) > 2 else np.nan)
    return {
        'raw_mean': float(np.nanmean(epoch_means)),
        'between_epoch_mean_sd': float(np.nanstd(epoch_means)),
        'log_sd': _log_positive(standard_deviation),
        'log_mad': _log_positive(
            np.median(np.abs(flat - np.median(flat)))),
        'log_iqr': _log_positive(
            quantile(flat, .75) - quantile(flat, .25)),
        'log_range_99': _log_positive(
            quantile(flat, .995) - quantile(flat, .005)),
        'log_abs_q999': _log_positive(quantile(absolute, .999)),
        'log_diff_sd': _log_positive(difference_sd),
        'log_diff_q99': _log_positive(quantile(absolute_diff, .99)),
        'log_diff_q999': _log_positive(quantile(absolute_diff, .999)),
        'log_ddiff_sd': _log_positive(np.std(second_differences)),
        'crest_factor': float(
            quantile(absolute, .999) / max(standard_deviation, 1e-12)),
        'diff_crest_factor': float(
            quantile(absolute_diff, .999) / max(difference_sd, 1e-12)),
        'skew': float(skew(flat, bias=False)),
        'kurtosis': float(kurtosis(flat, fisher=True, bias=False)),
        'power_1_50': band_power(1, 50),
        'power_50_200': band_power(50, 200),
        'power_200_1000': band_power(200, 1000),
        'power_1000_4500': band_power(1000, 4500),
        'spectral_centroid_hz': (
            float(np.sum(frequencies * power) / power.sum())
            if power.sum() else 0.0),
        'lag1_corr': lag1,
    }


def recording_block_feature_table(
        blocks: pd.DataFrame, *, n_trials: int = DEFAULT_N_TRIALS,
        trace_seconds: float = DEFAULT_TRACE_SECONDS,
        metadata_columns: Sequence[str] = (
            'cell_label', 'cell_id', 'group_id', 'start_time',
            'recording_technique', 'onlineAnalysis'),
        verbose: bool = True) -> pd.DataFrame:
    """Load raw responses and return exactly one feature row per epoch block."""
    from retinanalysis.SCutils.recording_mode import (
        _amp_response_table, _amp_trace_samples)

    required = {'exp_name', 'block_id'}
    missing = required.difference(blocks.columns)
    if missing:
        raise ValueError(f'blocks is missing {sorted(missing)}')
    one_per_block = blocks.drop_duplicates(['exp_name', 'block_id']).copy()
    response_table = _amp_response_table(one_per_block.block_id)
    rows = []
    experiments = list(one_per_block.groupby('exp_name', sort=False))
    for number, (exp_name, experiment) in enumerate(experiments, 1):
        if verbose:
            print(f'[{number}/{len(experiments)}] {exp_name}: '
                  f'{len(experiment)} block(s)')
        samples = _amp_trace_samples(
            experiment[['exp_name', 'block_id']],
            response_table=response_table, n_trials=int(n_trials),
            trace_seconds=float(trace_seconds), verbose=False)
        lookup = experiment.set_index('block_id')
        for block_id, (amp_data, sample_rate) in samples.items():
            source = lookup.loc[int(block_id)]
            features = extract_recording_block_features(amp_data, sample_rate)
            features.update({
                'exp_name': str(exp_name), 'block_id': int(block_id),
                'sample_rate': float(sample_rate),
                'n_epochs_sampled': int(np.asarray(amp_data).shape[0]),
            })
            for column in metadata_columns:
                if column in source:
                    features[column] = source[column]
            rows.append(features)
    return pd.DataFrame(rows)


def _candidate_models(random_state: int):
    from sklearn.ensemble import (
        ExtraTreesClassifier, GradientBoostingClassifier,
        HistGradientBoostingClassifier, RandomForestClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def scaled(model):
        return make_pipeline(SimpleImputer(strategy='median'),
                             StandardScaler(), model)

    def tree(model):
        return make_pipeline(SimpleImputer(strategy='median'), model)

    return {
        'logistic_C0.1': scaled(LogisticRegression(
            C=.1, class_weight='balanced', max_iter=5000,
            random_state=random_state)),
        'logistic_C1': scaled(LogisticRegression(
            C=1, class_weight='balanced', max_iter=5000,
            random_state=random_state)),
        'logistic_C10': scaled(LogisticRegression(
            C=10, class_weight='balanced', max_iter=5000,
            random_state=random_state)),
        'rbf_svc_C1': scaled(SVC(
            C=1, class_weight='balanced', probability=True,
            random_state=random_state)),
        'rbf_svc_C10': scaled(SVC(
            C=10, class_weight='balanced', probability=True,
            random_state=random_state)),
        'random_forest_leaf2': tree(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced', random_state=random_state, n_jobs=-1)),
        'random_forest_leaf5': tree(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=random_state, n_jobs=-1)),
        'extra_trees_leaf2': tree(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced', random_state=random_state, n_jobs=-1)),
        'extra_trees_leaf5': tree(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=random_state, n_jobs=-1)),
        'gradient_boosting': tree(GradientBoostingClassifier(
            n_estimators=100, max_depth=2, min_samples_leaf=5,
            random_state=random_state)),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, l2_regularization=1,
            random_state=random_state),
    }


def train_recording_technique_classifier(
        features: pd.DataFrame, *, technique_column: str = 'recording_technique',
        group_column: str = 'cell_id', random_state: int = 20260905
        ) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Select, evaluate, and fit a model using labelled epoch groups only.

    Returns ``(bundle, comparison, labelled_features)``. The comparison uses
    four-fold cell-grouped cross-validation inside a cell-held-out test split.
    The selected model is evaluated once on that untouched split, then fitted
    on all labelled blocks. ``labelled_features`` includes a leakage-free
    out-of-fold probability for every training block.
    """
    import scipy
    import sklearn
    from sklearn.base import clone
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, roc_auc_score)
    from sklearn.model_selection import (
        StratifiedGroupKFold, cross_val_predict)

    required = set(FEATURE_NAMES) | {
        technique_column, group_column, 'block_id'}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f'feature table is missing {sorted(missing)}')
    labelled = features.copy()
    labelled['target'] = labelled[technique_column].map(_technique_family)
    labelled = labelled[labelled.target.ne('')].reset_index(drop=True)
    if labelled.target.nunique() != 2:
        raise ValueError('labelled training data must contain both recording families')

    X = labelled.loc[:, list(FEATURE_NAMES)]
    y = labelled.target.eq('whole-cell').astype(int).to_numpy()
    groups = labelled[group_column].to_numpy()
    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state)
    development, holdout = next(outer.split(X, y, groups))
    inner = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=random_state + 1)
    candidates = _candidate_models(random_state)
    comparison_rows = []
    for name, candidate in candidates.items():
        fold_balanced_accuracy = []
        fold_roc_auc = []
        for train, validation in inner.split(
                X.iloc[development], y[development], groups[development]):
            model = clone(candidate).fit(
                X.iloc[development].iloc[train], y[development][train])
            predicted = model.predict(X.iloc[development].iloc[validation])
            probability = model.predict_proba(
                X.iloc[development].iloc[validation])[:, 1]
            fold_balanced_accuracy.append(balanced_accuracy_score(
                y[development][validation], predicted))
            fold_roc_auc.append(roc_auc_score(
                y[development][validation], probability))
        comparison_rows.append({
            'model': name,
            'cv_balanced_accuracy_mean': float(np.mean(fold_balanced_accuracy)),
            'cv_balanced_accuracy_sd': float(np.std(fold_balanced_accuracy)),
            'cv_roc_auc_mean': float(np.mean(fold_roc_auc)),
            'cv_roc_auc_sd': float(np.std(fold_roc_auc)),
        })
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ['cv_balanced_accuracy_mean', 'cv_roc_auc_mean'], ascending=False,
        kind='stable').reset_index(drop=True)
    selected_name = str(comparison.loc[0, 'model'])
    selected = clone(candidates[selected_name]).fit(
        X.iloc[development], y[development])
    holdout_probability = selected.predict_proba(X.iloc[holdout])[:, 1]
    holdout_prediction = holdout_probability >= .5

    full_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state)
    oof_probability = cross_val_predict(
        clone(candidates[selected_name]), X, y, groups=groups, cv=full_cv,
        method='predict_proba', n_jobs=1)[:, 1]
    labelled['oof_p_whole_cell'] = oof_probability
    labelled['oof_prediction'] = np.where(
        oof_probability >= .5, 'whole-cell', 'cell-attached')
    labelled['oof_disagrees'] = labelled.oof_prediction.ne(labelled.target)

    final_model = clone(candidates[selected_name]).fit(X, y)
    report = {
        'model_version': MODEL_VERSION,
        'selected_model': selected_name,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_blocks': int(len(labelled)),
        'training_cells': int(labelled[group_column].nunique()),
        'class_counts': {
            str(key): int(value)
            for key, value in labelled.target.value_counts().items()},
        'feature_names': list(FEATURE_NAMES),
        'grouping': group_column,
        'random_state': int(random_state),
        'holdout_blocks': int(len(holdout)),
        'holdout_cells': int(labelled.iloc[holdout][group_column].nunique()),
        'holdout_balanced_accuracy': float(balanced_accuracy_score(
            y[holdout], holdout_prediction)),
        'holdout_roc_auc': float(roc_auc_score(
            y[holdout], holdout_probability)),
        'holdout_confusion_matrix': confusion_matrix(
            y[holdout], holdout_prediction).tolist(),
        'oof_balanced_accuracy': float(balanced_accuracy_score(
            y, oof_probability >= .5)),
        'oof_roc_auc': float(roc_auc_score(y, oof_probability)),
        'software': {
            'python': platform.python_version(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        },
    }
    bundle = {
        'model_version': MODEL_VERSION,
        'model': final_model,
        'feature_names': list(FEATURE_NAMES),
        'report': report,
        'comparison': comparison.to_dict(orient='records'),
        'oof_p_whole_cell_by_block': {
            int(row.block_id): float(row.oof_p_whole_cell)
            for row in labelled[['block_id', 'oof_p_whole_cell']].itertuples(
                index=False)},
    }
    return bundle, comparison, labelled


def retrain_recording_technique_classifier(
        blocks: pd.DataFrame, *, model_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
        training_features_path: Optional[Path] = None,
        technique_column: str = 'recording_technique',
        group_column: str = 'cell_id', verbose: bool = True) -> dict:
    """Rebuild the checked-in classifier from explicitly labelled blocks.

    Unlabelled blocks are removed before any training features are constructed.
    The saved CSV contains the exact feature snapshot and cell-held-out
    predictions needed to reproduce and audit the model without rereading raw
    H5 responses.
    """
    import json

    if technique_column not in blocks:
        raise ValueError(f'blocks is missing {technique_column!r}')
    labelled_blocks = blocks[
        blocks[technique_column].map(_technique_family).ne('')].copy()
    if labelled_blocks.empty:
        raise ValueError('no explicitly labelled recordingTechnique blocks found')
    features = recording_block_feature_table(labelled_blocks, verbose=verbose)
    bundle, comparison, labelled = train_recording_technique_classifier(
        features, technique_column=technique_column,
        group_column=group_column)
    saved_model = save_recording_technique_classifier(bundle, model_path)
    saved_report = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    saved_training = (DEFAULT_TRAINING_FEATURES_PATH
                      if training_features_path is None
                      else Path(training_features_path))
    report = dict(bundle['report'])
    report['model_comparison'] = comparison.to_dict(orient='records')
    disagreements = labelled[
        labelled.oof_disagrees
        & ((labelled.oof_p_whole_cell >= DEFAULT_MIN_CONFIDENCE)
           | (labelled.oof_p_whole_cell <= 1 - DEFAULT_MIN_CONFIDENCE))]
    report['high_confidence_oof_disagreements'] = disagreements[[
        column for column in (
            'exp_name', 'cell_label', 'block_id', technique_column,
            'oof_prediction', 'oof_p_whole_cell')
        if column in disagreements]].to_dict(orient='records')
    saved_report.write_text(json.dumps(report, indent=2) + '\n')
    identity = [column for column in (
        'exp_name', 'cell_label', group_column, 'group_id', 'block_id',
        'start_time', technique_column, 'sample_rate', 'n_epochs_sampled')
        if column in labelled]
    audit = [
        'target', 'oof_p_whole_cell', 'oof_prediction', 'oof_disagrees']
    labelled[identity + list(FEATURE_NAMES) + audit].to_csv(
        saved_training, index=False)
    if verbose:
        print(f'saved classifier: {saved_model}')
        print(f'saved report: {saved_report}')
        print(f'saved labelled feature snapshot: {saved_training}')
    return bundle


def save_recording_technique_classifier(
        bundle: dict, path: Optional[Path] = None) -> Path:
    """Persist a trusted, versioned classifier bundle with joblib."""
    import joblib

    target = DEFAULT_MODEL_PATH if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    return target


def load_recording_technique_classifier(path: Optional[Path] = None) -> dict:
    """Load the trusted checked-in classifier and verify its schema/version."""
    import joblib
    import sklearn

    target = DEFAULT_MODEL_PATH if path is None else Path(path)
    bundle = joblib.load(target)
    if bundle.get('model_version') != MODEL_VERSION:
        raise ValueError(
            f'{target.name} has model version {bundle.get("model_version")}; '
            f'expected {MODEL_VERSION}')
    if tuple(bundle.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError(f'{target.name} feature schema does not match this code')
    trained_version = bundle.get('report', {}).get(
        'software', {}).get('scikit_learn')
    if trained_version and trained_version != sklearn.__version__:
        warnings.warn(
            f'{target.name} was trained with scikit-learn {trained_version}; '
            f'this environment has {sklearn.__version__}. Retrain before relying '
            'on predictions.', RuntimeWarning, stacklevel=2)
    return bundle


def predict_recording_techniques(
        features: pd.DataFrame, bundle: Optional[dict] = None, *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        use_training_oof: bool = True) -> pd.DataFrame:
    """Predict block families, retaining low-confidence calls as unresolved.

    A labelled block present in the model's training set uses its cell-held-out
    probability. New and unlabelled blocks use the final model fitted to all
    labelled data.
    """
    bundle = (load_recording_technique_classifier()
              if bundle is None else bundle)
    if not 0.5 <= float(min_confidence) <= 1:
        raise ValueError('min_confidence must be between 0.5 and 1')
    missing = set(FEATURE_NAMES).difference(features.columns)
    if missing:
        raise ValueError(f'feature table is missing {sorted(missing)}')
    result = features.copy()
    probability = bundle['model'].predict_proba(
        result.loc[:, list(FEATURE_NAMES)])[:, 1]
    source = np.full(len(result), 'model fitted to labelled blocks', dtype=object)
    if use_training_oof and 'block_id' in result:
        oof = bundle.get('oof_p_whole_cell_by_block', {})
        for position, block_id in enumerate(result.block_id):
            try:
                key = int(block_id)
            except (TypeError, ValueError):
                continue
            if key in oof:
                probability[position] = float(oof[key])
                source[position] = 'cell-held-out prediction'
    confidence = np.maximum(probability, 1 - probability)
    prediction = np.where(
        probability >= .5, 'whole-cell', 'cell-attached')
    result['classifier_p_whole_cell'] = probability
    result['classifier_confidence'] = confidence
    result['classifier_prediction'] = prediction
    result['classifier_family'] = np.where(
        confidence >= float(min_confidence), prediction, '')
    result['classifier_source'] = source
    return result
