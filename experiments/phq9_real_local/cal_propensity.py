"""Observed-data CAL adapter; the archived convex solver is used unchanged.

The original CAL loss, ridge/L1 grids, reference arm, unpenalized intercept,
KKT tolerance and action-log-loss selector are retained. Real data have no
known marginal pilot: the intercept-only candidate is estimated separately
inside each inner training fold, then refitted on the outer training fold.
Neither outcomes nor outer evaluation rows are arguments to fit_cal_models.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from vendor import logit_solver

RIDGES = (.003, .03, .3)
L1_VALUES = (0., .01, .05)
SELECTION_FLOORS = (.01, .02)
FAMILIES = ('cal_linear', 'cal_sparse')


@dataclass
class LinearMap:
    input_dimension: int
    kept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('Finite nonempty encoded CAL features required.')
        mean, scale = x.mean(0), x.std(0)
        tolerance = 64. * np.finfo(float).eps * np.maximum(1., np.max(np.abs(x), axis=0))
        keep = np.flatnonzero((np.ptp(x, axis=0) > tolerance) & (scale > tolerance))
        return cls(x.shape[1], keep, mean[keep], scale[keep])

    def transform(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.input_dimension or not np.isfinite(x).all():
            raise ValueError('CAL feature dimensions or values changed.')
        return np.column_stack([np.ones(len(x)), (x[:, self.kept] - self.mean) / self.scale])


@dataclass
class CalProbability:
    preprocessor: object
    geometry: LinearMap
    beta: object
    pbar: np.ndarray
    choice: dict

    def predict_proba(self, x, selected=False):
        if self.choice['pilot']:
            p = np.broadcast_to(self.pbar, (len(x), 4)).copy()
        else:
            phi = self.geometry.transform(self.preprocessor.transform(x))
            p = softmax(np.column_stack([np.zeros(len(x)), phi @ self.beta]), axis=1)
            if selected:
                floor = self.choice['floor']
                p = floor + (1. - 4. * floor) * p
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-10, rtol=0):
            raise FloatingPointError('Invalid CAL probabilities.')
        return p


def _action_ce(p, a):
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise FloatingPointError('Nonpositive probability in CAL validation.')
    return float(-np.log(p[np.arange(len(a)), a]).mean())


def _geometry(x, preprocessor_factory):
    pre = preprocessor_factory()
    encoded = np.asarray(pre.fit_transform(x), float)
    geometry = LinearMap.fit(encoded)
    return pre, geometry, geometry.transform(encoded)


def _fit_path(phi, a, pairs, phase):
    paths, logs = {}, []
    initial = None
    for ridge, l1 in sorted(pairs, key=lambda k: (-k[0], -k[1])):
        try:
            beta, certificate = logit_solver.fit_coefficients(phi, a, 'cal', ridge, l1, initial)
            initial = beta
            paths[ridge, l1] = beta
            logs.append(dict(certificate, event='CAL_SOLVER', phase=phase))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            paths[ridge, l1] = None
            logs.append(dict(event='CAL_SOLVER', phase=phase, ridge=ridge, l1=l1,
                             status='INVALID', reason=type(exc).__name__ + ': ' + str(exc)))
    return paths, logs


def fit_cal_models(x, actions, families, preprocessor_factory, seed, progress=None):
    """Two-fold tuning inside one OUTER training fold; outcomes are not read.

    CAL-sparse is a candidate family including L1=0. The selected L1/pilot
    choice is reported explicitly; positive sparsity is never assumed.
    """
    families = tuple(dict.fromkeys(families))
    if not families or set(families) - set(FAMILIES):
        raise ValueError('Unknown CAL family.')
    a = np.asarray(actions)
    if a.shape != (len(x),) or not np.isfinite(a).all() or np.any(a != np.floor(a)) or not np.isin(a, range(4)).all():
        raise ValueError('CAL needs aligned observed treatment indices 0..3.')
    a = a.astype(int)
    if np.bincount(a, minlength=4).min() < 2:
        raise ValueError('CAL inner validation needs at least two outer-training patients per arm.')
    say = progress or (lambda _: None)
    l1_values = L1_VALUES if 'cal_sparse' in families else (0.,)
    pairs = {(r, l) for r in RIDGES for l in l1_values}
    splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(x, a))
    logs, fold_scores, pilot_scores = [], [], []
    for fold, (train, val) in enumerate(splits, 1):
        say(f'CAL inner fold {fold}/2: {len(pairs)} archived penalty fits; training-only preprocessing')
        pre, geometry, phi = _geometry(x.iloc[train], preprocessor_factory)
        val_phi = geometry.transform(pre.transform(x.iloc[val]))
        paths, fit_logs = _fit_path(phi, a[train], pairs, f'inner_{fold}')
        logs.extend(fit_logs)
        pbar = np.bincount(a[train], minlength=4).astype(float) / len(train)
        pilot_scores.append(_action_ce(np.broadcast_to(pbar, (len(val), 4)), a[val]))
        scores = {}
        for (ridge, l1), beta in paths.items():
            if beta is None:
                continue
            raw = softmax(np.column_stack([np.zeros(len(val)), val_phi @ beta]), axis=1)
            for floor in SELECTION_FLOORS:
                scores[ridge, l1, floor] = _action_ce(floor + (1. - 4. * floor) * raw, a[val])
        fold_scores.append(scores)
    choices, candidates = {}, []
    for family in families:
        records = [dict(family=family, candidate=0, pilot=True, ridge=0., l1=0., floor=0.,
                        valid=True, inner_fold_ce=list(pilot_scores), action_ce_cv=float(np.mean(pilot_scores)),
                        pilot_source='inner_training_empirical_arm_frequencies')]
        for ridge in RIDGES:
            for l1 in L1_VALUES if family == 'cal_sparse' else (0.,):
                for floor in SELECTION_FLOORS:
                    scores = [s.get((ridge, l1, floor)) for s in fold_scores]
                    valid = all(v is not None and np.isfinite(v) for v in scores)
                    records.append(dict(family=family, candidate=len(records), pilot=False,
                                        ridge=ridge, l1=l1, floor=floor, valid=valid,
                                        inner_fold_ce=scores,
                                        action_ce_cv=float(np.mean(scores)) if valid else None))
        best = records[0]
        for rec in records[1:]:
            if rec['valid'] and rec['action_ce_cv'] < best['action_ce_cv'] - 1e-12:
                best = rec
        choices[family] = dict(best, selector='two_inner_fold_action_cross_entropy',
                               known_simulator_probabilities_used=False)
        candidates.extend(records)
    required = {(c['ridge'], c['l1']) for c in choices.values() if not c['pilot']}
    say('CAL: refit frozen selections on the outer training fold')
    pre, geometry, phi = _geometry(x, preprocessor_factory)
    paths, fit_logs = _fit_path(phi, a, required, 'outer_refit')
    logs.extend(fit_logs)
    pbar = np.bincount(a, minlength=4).astype(float) / len(a)
    models = {}
    for family, choice in choices.items():
        beta = None if choice['pilot'] else paths[choice['ridge'], choice['l1']]
        if not choice['pilot'] and beta is None:
            raise RuntimeError(f'{family}: selected outer refit failed; no silent fallback or reselection.')
        models[family] = CalProbability(pre, geometry, beta, pbar, choice)
    return models, dict(selection=choices, candidates=candidates, solver=logs,
                        preprocessing_fitted_within_each_inner_training_fold=True,
                        selected_pilot_refit='outer_training_empirical_arm_frequencies',
                        selection_uses_outcomes=False, selection_uses_outer_evaluation_rows=False,
                        ridge_grid=list(RIDGES), l1_grid=list(l1_values),
                        selection_floor_grid=list(SELECTION_FLOORS),
                        solver_kkt_tolerance=logit_solver.KKT_TOLERANCE)
