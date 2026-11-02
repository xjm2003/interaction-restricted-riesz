"""Cross-fitted real-outcome contrasts in an already selected PHQ-9 cohort.

The loader must supply exactly one independently sampled patient per row and
pre-treatment covariates. This module cannot establish causal identification,
repair outcome-dependent cohort selection, or turn recorded exposure into a
well-defined intervention. No patient-level predictions are returned or saved.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from cal_propensity import fit_cal_models


ARMS = ('no_active_treatment', 'antidepressant_only', 'psychotherapy_only', 'combined')
CONTRASTS = ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))
CONTRAST_NAMES = tuple(ARMS[a] + '_minus_' + ARMS[b] for a, b in CONTRASTS)
CONTRAST_MATRIX = np.array([np.eye(4)[a] - np.eye(4)[b] for a, b in CONTRASTS])
PROCEDURES = ('raw_full', 'raw_restricted', 'projected_full', 'projected_restricted', 'raw_plugin')
PAIRS = {
    'correction_projected': ('projected_restricted', 'projected_full'),
    'correction_raw': ('raw_restricted', 'raw_full'),
    'projection_full': ('projected_full', 'raw_full'),
    'projection_restricted': ('projected_restricted', 'raw_restricted'),
    'overall': ('projected_restricted', 'raw_full'),
}
STATUS = 'EXPLORATORY_SELECTED_COHORT'
K = np.array([1., -1., -1., 1.])


def _probabilities(p, positive=True):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] != 4 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError('Expected a finite nonempty probability matrix with four columns.')
    if np.any(p < 0) or (positive and np.any(p <= 0)) or not np.allclose(p.sum(1), 1., atol=1e-9, rtol=0):
        raise ValueError('Probabilities must sum to one and satisfy the requested positivity condition.')
    return p


def stabilized_probabilities(p, floor):
    """Explicit uniform-mixture sensitivity: p_f = f + (1 - 4f) p.

    This is NOT simple clipping and is not an estimate of the original propensity
    when f is fixed above zero. No probability transformation is used at f=0.
    """
    p = _probabilities(p, positive=False)
    floor = float(floor)
    if not np.isfinite(floor) or not 0 <= floor < .25:
        raise ValueError('Each probability floor must be finite and in [0, 0.25).')
    result = p if floor == 0 else floor + (1. - 4. * floor) * p
    return _probabilities(result)


def matched_scores(p_train, p_test, mu_train, mu_test, actions, outcomes,
                   contrasts=CONTRAST_MATRIX):
    """CONST projection using outer-training predictions only, one landmark.

    Training outcome observations are absent from this API. Outcome and Riesz
    projection coordinates are determined from the training X predictions.
    """
    pt, pe = _probabilities(p_train), _probabilities(p_test)
    mt, me = np.asarray(mu_train, float), np.asarray(mu_test, float)
    c = np.asarray(contrasts, float)
    a, y = np.asarray(actions), np.asarray(outcomes, float)
    if mt.shape != pt.shape or me.shape != pe.shape or not np.isfinite(mt).all() or not np.isfinite(me).all():
        raise ValueError('Outcome predictions must be finite and aligned with probabilities.')
    if c.ndim != 2 or c.shape[1] != 4 or not np.isfinite(c).all():
        raise ValueError('Contrasts must be a finite queries-by-four matrix.')
    if a.shape != (len(pe),) or y.shape != (len(pe),) or not np.isfinite(y).all():
        raise ValueError('Held-out actions and outcomes must be aligned.')
    if not np.isfinite(a).all() or np.any(a != np.floor(a)) or np.any((a < 0) | (a > 3)):
        raise ValueError('Actions must be integer arm indices 0..3.')
    a = a.astype(int)
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        jt = 1. / np.sum(1. / pt, axis=1)
        je = 1. / np.sum(1. / pe, axis=1)
        if jt.sum() <= 0:
            raise FloatingPointError('Training projection weights underflowed.')
        delta = np.sum(jt * (mt @ K)) / np.sum(jt)
        ell_t = np.einsum('na,qa->nq', K / pt, c)
        ell_e = np.einsum('na,qa->nq', K / pe, c)
        gamma = np.sum(jt[:, None] * ell_t, axis=0) / np.sum(jt)
        projected = me - je[:, None] * (K / pe) * (me @ K - delta)[:, None]
        full = c[None, :, :] / pe[:, None, :]
        restricted = full - (je[:, None] * (ell_e - gamma)[...])[:, :, None] * (K / pe)[:, None, :]
        row = np.arange(len(pe))[:, None]
        query = np.arange(len(c))[None, :]
        def score(m, alpha):
            return m @ c.T + alpha[row, query, a[:, None]] * (y - m[np.arange(len(pe)), a])[:, None]
        scores = dict(raw_full=score(me, full), raw_restricted=score(me, restricted),
                      projected_full=score(projected, full), projected_restricted=score(projected, restricted),
                      raw_plugin=me @ c.T)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise FloatingPointError('Nonfinite score; check overlap and probability sensitivity.')
        paired_algebra = (full - restricted)[row, query, a[:, None]] * (y - projected[np.arange(len(pe)), a])[:, None]
        diagnostics = dict(
            outcome_restriction_max_error=float(np.max(np.abs(projected @ K - delta))),
            representer_restriction_max_error=float(np.max(np.abs(np.sum(restricted * K, axis=2) - gamma))),
            paired_algebra_max_error=float(np.max(np.abs(scores['projected_full'] - scores['projected_restricted'] - paired_algebra))),
            outcome_projection_max_change=float(np.max(np.abs(projected-me))),
            full_representer_max_abs=float(np.max(np.abs(full))),
            restricted_representer_max_abs=float(np.max(np.abs(restricted))),
        )
    return scores, diagnostics


def _clean_x(x):
    if not isinstance(x, pd.DataFrame) or x.empty or x.columns.duplicated().any():
        raise ValueError('X must be a nonempty DataFrame with unique baseline-column names.')
    out = x.copy()
    numeric, categorical = [], []
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors='raise').astype(float)
            if np.isinf(out[col].to_numpy()).any():
                raise ValueError('Baseline numeric values must not contain infinity.')
            numeric.append(col)
        else:
            out[col] = out[col].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            categorical.append(col)
    return out, numeric, categorical


def _preprocessor(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


def _outcome_design(x, actions):
    return np.column_stack([x, np.eye(4)[np.asarray(actions, int)]])


def _predict_four(model, x):
    return np.column_stack([model.predict(_outcome_design(x, np.full(len(x), a))) for a in range(4)])


def _interval(values, plugin=False):
    values = np.asarray(values, float)
    estimate = float(values.mean())
    if plugin:
        return dict(estimate=estimate, se=None, ci_lower=None, ci_upper=None,
                    family_ci_lower=None, family_ci_upper=None,
                    interval_interpretation='NONE: plug-in uncertainty requires nuisance-fit uncertainty')
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    z, zf = float(norm.ppf(.975)), float(norm.ppf(1-.05/(2*len(CONTRASTS))))
    return dict(estimate=estimate, se=se, ci_lower=estimate-z*se, ci_upper=estimate+z*se,
                family_ci_lower=estimate-zf*se, family_ci_upper=estimate+zf*se,
                interval_interpretation='NOMINAL patient-score Wald; identification and nuisance-rate assumptions unverified')


def _setting_fields(setting):
    if setting == 'inner_cv_selected':
        return dict(probability_setting='inner_cv_selected', probability_floor=None,
                    propensity_interpretation='training-only CAL selection; see fold-specific choices')
    return dict(probability_setting=f'fixed_floor_{setting:g}', probability_floor=float(setting),
                propensity_interpretation='unmodified' if setting == 0 else 'uniform_mixture_sensitivity')


def _method_label(source, method):
    sources = dict(cal_sparse='CAL-sparse', cal_linear='CAL-linear', logit='Logit', hgb='HGB')
    procedures = dict(projected_restricted='matched outcome projection + restricted Riesz',
                      raw_full='full AIPW', projected_full='projected outcome + full AIPW',
                      raw_restricted='raw outcome + restricted Riesz', raw_plugin='raw plug-in')
    return sources[source] + ' | ' + procedures[method]


def _overlap_rows(p, a, source, setting):
    rows = []
    for arm in range(4):
        v = p[:, arm]
        take = a == arm
        w = 1. / v[take]
        rows.append(dict(probability_source=source, **_setting_fields(setting), arm=ARMS[arm],
            n_patients=len(a), observed_arm_n=int(take.sum()),
            probability_min=float(v.min()), probability_q01=float(np.quantile(v,.01)),
            probability_q05=float(np.quantile(v,.05)), probability_median=float(np.median(v)),
            probability_below_001=int(np.sum(v<.01)), probability_below_005=int(np.sum(v<.05)),
            observed_probability_min=float(v[take].min()), inverse_weight_max=float(w.max()),
            inverse_weight_ess=float((w / w.max()).sum()**2 / np.sum((w / w.max())**2)),
            interpretation='OOF treatment-probability diagnostic; no selection correction'))
    return rows


def run_estimation(X, actions, outcomes, seed=20261022, folds=5,
                   propensity_sources=('cal_sparse', 'cal_linear', 'logit', 'hgb'), probability_floors=(0., .01, .02),
                   threads=2, progress=print):
    """Fit fixed-specification nuisances once per fold; report aggregate results.

    Every result applies only to the provided selected cohort. Above-zero
    probability floors are explicit sensitivity analyses. They generally lose
    the exact original-propensity estimating-equation property. The restricted
    procedures additionally rely on their structural restriction; their nominal
    intervals do not repair its misspecification.
    """
    x, numeric, categorical = _clean_x(X)
    a, y = np.asarray(actions), np.asarray(outcomes, float)
    if a.shape != (len(x),) or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError('X, treatment and real outcome must be row-aligned and outcome-complete.')
    if not np.isfinite(a).all() or np.any(a != np.floor(a)) or np.any((a < 0) | (a > 3)):
        raise ValueError('Treatment must be encoded as integers 0..3.')
    a = a.astype(int)
    if not isinstance(folds, (int, np.integer)) or folds < 2 or threads < 1:
        raise ValueError('At least two folds and one computational thread are required.')
    counts = np.bincount(a, minlength=4)
    if np.any(counts < folds):
        raise ValueError('Each of the four arms needs at least as many patients as cross-fitting folds.')
    sources = tuple(dict.fromkeys(propensity_sources))
    floors = tuple(dict.fromkeys(float(v) for v in probability_floors))
    if not sources or set(sources) - {'logit', 'hgb', 'cal_sparse', 'cal_linear'}:
        raise ValueError('Use at least one source from cal_sparse, cal_linear, logit, hgb.')
    if not floors or any(not np.isfinite(f) or not 0 <= f < .25 for f in floors):
        raise ValueError('Use nonempty distinct probability floors in [0, .25).')
    n = len(x)
    cal_families = tuple(s for s in sources if s.startswith('cal_'))
    if cal_families and np.any(counts - np.ceil(counts / folds).astype(int) < 2):
        raise ValueError('CAL requires at least two training patients per arm in every outer fold.')
    settings = {s: (('inner_cv_selected',) + floors if s.startswith('cal_') else floors) for s in sources}
    cache = {(s,f,l): {p:np.full((n,len(CONTRASTS)), np.nan) for p in PROCEDURES}
             for s in sources for f in settings[s] for l in ('linear','flexible')}
    probabilities = {(s,f):np.full((n,4),np.nan) for s in sources for f in settings[s]}
    invalid = set()
    diagnostics = []
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with threadpool_limits(limits=threads):
        for fold, (train, test) in enumerate(split.split(x, a), 1):
            progress(f'Fold {fold}/{folds}: fit baseline preprocessing and both outcome learners')
            pre = _preprocessor(numeric, categorical)
            xt = np.asarray(pre.fit_transform(x.iloc[train]), float)
            xe = np.asarray(pre.transform(x.iloc[test]), float)
            design = _outcome_design(xt, a[train])
            outcome_models = {
                'linear': Ridge(alpha=10.),
                'flexible': HistGradientBoostingRegressor(learning_rate=.08,max_iter=160,
                    max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5.,
                    early_stopping=False,random_state=seed+fold),
            }
            outcome_predictions = {}
            for learner, model in outcome_models.items():
                model.fit(design,y[train])
                outcome_predictions[learner] = (_predict_four(model,xt),_predict_four(model,xe))
            cal_models = {}
            if cal_families:
                cal_models, cal_diagnostics = fit_cal_models(
                    x.iloc[train], a[train], cal_families,
                    lambda: _preprocessor(numeric, categorical), seed + 172 + fold,
                    progress=lambda message: progress(f'Fold {fold}/{folds}: {message}'))
                diagnostics.append(dict(fold=fold, event='CAL_TUNING', **cal_diagnostics))
            for source in sources:
                progress(f'Fold {fold}/{folds}: fit {source} propensity; calculate matched procedures')
                caught = []
                if source in cal_models:
                    model = cal_models[source]
                    pt, pe = model.predict_proba(x.iloc[train]), model.predict_proba(x.iloc[test])
                    selected_pt = model.predict_proba(x.iloc[train], selected=True)
                    selected_pe = model.predict_proba(x.iloc[test], selected=True)
                    diagnostics.append(dict(fold=fold, probability_source=source,
                        event='CAL_SELECTED', choice=model.choice,
                        primary_probability_setting='inner_cv_selected',
                        selected_floor=float(model.choice['floor']),
                        empirical_intercept_candidate_selected=bool(model.choice['pilot'])))
                else:
                    model = (LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,tol=1e-6)
                             if source=='logit' else HistGradientBoostingClassifier(
                                 learning_rate=.05,max_iter=150,max_leaf_nodes=15,
                                 min_samples_leaf=20,l2_regularization=5.,early_stopping=False,
                                 random_state=seed+fold))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always',ConvergenceWarning)
                        model.fit(xt,a[train])
                    if not np.array_equal(model.classes_,np.arange(4)):
                        raise ValueError('A training fold did not fit all four treatment classes.')
                    pt, pe = model.predict_proba(xt), model.predict_proba(xe)
                _probabilities(pt,positive=False); _probabilities(pe,positive=False)
                diagnostics.append(dict(fold=fold,probability_source=source,status='FIT_COMPLETE',
                    training_n=len(train),evaluation_n=len(test),encoded_covariates=xt.shape[1],
                    convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught),
                    raw_training_probability_min=float(pt.min()),raw_evaluation_probability_min=float(pe.min()),
                    preprocessing_training_only=True))
                for setting in settings[source]:
                    try:
                        if setting == 'inner_cv_selected':
                            ptf, pef = _probabilities(selected_pt), _probabilities(selected_pe)
                        else:
                            ptf, pef = stabilized_probabilities(pt,setting),stabilized_probabilities(pe,setting)
                        probabilities[source,setting][test] = pef
                        for learner,(mt,me) in outcome_predictions.items():
                            scores, info = matched_scores(ptf,pef,mt,me,a[test],y[test])
                            for method, value in scores.items():
                                cache[source,setting,learner][method][test] = value
                            diagnostics.append(dict(fold=fold,probability_source=source,
                                **_setting_fields(setting),learner=learner,status='SCORES_COMPLETE',**info))
                    except (ValueError,ArithmeticError,FloatingPointError) as exc:
                        if setting == 'inner_cv_selected':
                            raise RuntimeError(f'{source}: selected CAL score failed; no substitute result.') from exc
                        invalid.add((source,setting))
                        diagnostics.append(dict(fold=fold,probability_source=source,
                            **_setting_fields(setting),status='SOURCE_FLOOR_NOT_REPORTED',
                            reason=type(exc).__name__+': '+str(exc)))
    result = dict(estimates=[],paired_differences=[],overlap=[],fit_diagnostics=diagnostics)
    for source in sources:
        for setting in settings[source]:
            if (source,setting) in invalid:
                continue
            p = _probabilities(probabilities[source,setting])
            result['overlap'].extend(_overlap_rows(p,a,source,setting))
            for learner in ('linear','flexible'):
                scores = cache[source,setting,learner]
                if any(not np.isfinite(z).all() for z in scores.values()):
                    raise RuntimeError('Incomplete out-of-fold score grid; no partial-cohort estimate is allowed.')
                common = dict(probability_source=source,**_setting_fields(setting),learner=learner,
                    n_patients=n,folds=folds,seed=int(seed),status=STATUS,
                    scope='selected observed-outcome cohort; causal interpretation not established',
                    outcome='real_phq9_60_120d',unit='PHQ-9 points',
                    family_interpretation='six contrasts within this source/floor/learner/procedure only')
                for method in PROCEDURES:
                    for q,(left,right) in enumerate(CONTRASTS):
                        result['estimates'].append(dict(**common,method=method,contrast=CONTRAST_NAMES[q],
                            method_label=_method_label(source,method),
                            is_primary_method=(source=='cal_sparse' and setting=='inner_cv_selected'
                                               and method=='projected_restricted'),
                            left_arm=ARMS[left],right_arm=ARMS[right],
                            **_interval(scores[method][:,q],plugin=method=='raw_plugin')))
                for pair,(left,right) in PAIRS.items():
                    difference=scores[left]-scores[right]
                    for q in range(len(CONTRASTS)):
                        info=_interval(difference[:,q])
                        result['paired_differences'].append(dict(**common,pair=pair,
                            left_method=left,right_method=right,contrast=CONTRAST_NAMES[q],
                            estimate_difference=info['estimate'],se_difference=info['se'],
                            ci_lower=info['ci_lower'],ci_upper=info['ci_upper'],
                            interpretation='Nominal paired score difference; not a validated Hausman test'))
    if not result['estimates']:
        raise ValueError('No source/floor combination produced a complete valid score grid.')
    return result
