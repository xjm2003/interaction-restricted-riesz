"""Observed-data sparse Riesz fitting, then exact fixed-population evaluation.

The fitting API receives observed artifacts, known query weights and the
benchmark's original marginal pilot. Conditional simulator probabilities are
used only to generate actions and, after freeze_callback, to evaluate risk.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
try:
    from . import sparse_model
except ImportError:
    import sparse_model

VERSION = 'semi-sparse-riesz-v1.0'
METHODS = ('pilot', 'regularized_B1', 'projected_regularized_B1',
           'independent_additive_ridge', 'independent_sparse_riesz',
           'shared_additive_ridge', 'shared_sparse_riesz')


def protocol_config(data):
    core = data['core']
    return dict(version=VERSION, methods=list(METHODS),
                observation_folds=2, sampling='original_iid_landmark_rows_with_replacement',
                original_B1_ridge_grid=list(map(float, core.BLOCK_RIDGE_GRID)),
                original_B1_relative_cutoff=float(core.FROZEN_RELATIVE_TOLERANCE),
                original_B1_gamma='original_continuous_observed_calibration_in_[0,1]',
                sparse_ridges=[.003, .03, .3], sparse_l1=[0., .01, .05],
                sparse_gammas=[0., .5, 1.], kkt_relative_tolerance=2e-7,
                inverse_projection=[1., 100.],
                projected_B1='same_selected_raw_B1_projected_after_fit_no_reselection',
                geometry='fold_fitted_nonconstant_standardized_sieve_plus_tanh_and_bounded_square_nonbinary_columns',
                geometry_final='refit_on_all_observed_X', feature_cap=None,
                training_weight='mean_query_squared_scale_normalized_within_each_step_and_action',
                validation_weight='raw_unnormalized_squared_query_scale',
                pilot='original_known_marginal_by_step_and_action',
                time_steps=int(core.N_STEPS), landmark_probability=float(core.P_LANDMARK),
                shared_training_queries='train', shared_final_queries='train_plus_selection',
                independent_final_queries='each_held_target_own_empirical_moment',
                joint_time_sharing=False, group_lasso=False, learned_low_rank=False,
                conditional_propensity_fitted=False, oracle_model_selection=False,
                primary_ratio='arithmetic_mean_MSE_over_arithmetic_mean_MSE',
                replicate_unit='paired_observation_seed_conditional_on_fixed_population_and_queries')


def population_arrays(data):
    core = data['core']; pop = data['population']
    steps = pop['step'].to_numpy(dtype=int)
    probabilities = pop[['p_star__' + a for a in core.ACTION_ORDER]].to_numpy(dtype=float)
    if (probabilities.shape != (len(steps), 4) or not np.isfinite(probabilities).all()
            or np.any(probabilities < .01 - 1e-10)
            or not np.allclose(probabilities.sum(axis=1), 1., rtol=0, atol=1e-8)):
        raise ValueError('Frozen benchmark probabilities must retain the original 0.01 floor.')
    if not np.array_equal(np.unique(steps), np.arange(core.N_STEPS)):
        raise ValueError('Population landmark support changed.')
    pbar = np.stack([probabilities[steps == t].mean(axis=0) for t in range(core.N_STEPS)])
    if not np.isfinite(pbar).all() or np.max(1 / pbar) > 100. + 1e-6:
        raise ValueError('Original marginal pilot is outside the frozen inverse range.')
    return steps, probabilities, pbar


def draw_artifacts(data, sample_size, seed):
    """Byte-equivalent original row/action/observation-fold sampling rule."""
    core = data['core']; q = data['queries']; N = len(data['x_population'])
    steps, probabilities, pbar = population_arrays(data)
    rng = np.random.default_rng(int(seed) + 100000 * int(sample_size))
    rows = rng.integers(0, N, size=int(sample_size))
    actions = core.draw_actions(probabilities[rows], rng)
    x = np.asarray(data['x_population'][rows], dtype=float)
    observed_steps = steps[rows]
    order = rng.permutation(sample_size)
    halves = (order[:sample_size // 2], order[sample_size // 2:], np.arange(sample_size))
    artifacts = [core.make_sample_artifact(rows[k], observed_steps[k], actions[k], x[k], q, pbar)
                 for k in halves]
    return artifacts, pbar


def fit_original_B1(core, queries, artifacts):
    """Extract exactly the original B1 branch, without fitting unused RR models."""
    selection = queries['indices']['selection']; d = artifacts[2]['x'].shape[1]
    rows = []
    for ridge in core.BLOCK_RIDGE_GRID:
        state_a = core.regularized_state(artifacts[0], ridge)
        state_b = core.regularized_state(artifacts[1], ridge)
        beta_ab = core.white_to_raw(state_a['targets'][selection], state_a['blocks'], d)
        beta_ba = core.white_to_raw(state_b['targets'][selection], state_b['blocks'], d)
        gamma, score = core.calibrated_score([
            core.objective_components(beta_ab, artifacts[1], selection),
            core.objective_components(beta_ba, artifacts[0], selection)])
        if not np.isfinite([gamma, score]).all():
            raise FloatingPointError('Nonfinite original B1 calibration.')
        rows.append(dict(method='regularized_B1', method_family='regularized_B1',
                         block_ridge=float(ridge), gamma=float(gamma),
                         crossfit_objective=float(score), status='valid',
                         objective_convention='correction_loss_relative_to_pilot'))
    table = pd.DataFrame(rows)
    best = table.sort_values(['crossfit_objective', 'block_ridge'],
                             ascending=[True, False], kind='mergesort').iloc[0]
    choice = dict(block_ridge=float(best.block_ridge), gamma=float(best.gamma),
                  crossfit_objective=float(best.crossfit_objective))
    state = core.regularized_state(artifacts[2], choice['block_ridge'])
    beta = core.white_to_raw(state['targets'][queries['indices']['held_out']] * choice['gamma'],
                             state['blocks'], d)
    if not np.isfinite(beta).all():
        raise FloatingPointError('Nonfinite final original B1 coefficients.')
    return beta, choice, table


def query_scale(queries, indices, rows, steps, landmark_probability):
    """N by Q by 4 signed multiplier; zero support stays exactly zero."""
    indices = np.asarray(indices, dtype=int)
    omega = queries['omega_unique'][queries['inverse_eta'][indices, None], rows[None, :]].T
    time_weight = queries['h'][indices[:, None], steps[None, :]].T
    return ((time_weight * omega / landmark_probability)[:, :, None]
            * queries['c'][indices][None, :, :])


def project_representer(alpha, scale):
    """Equivalent inverse-weight projection without division by signed/zero s."""
    return np.clip(alpha, np.minimum(scale, 100. * scale), np.maximum(scale, 100. * scale))


def evaluate_frozen(data, beta, models, indices, chunk_size=2048):
    """Literal population integration; no subtraction of large risk polynomials."""
    core = data['core']; q = data['queries']; x = data['x_population']
    steps, probabilities, pbar = population_arrays(data)
    N = len(x); Q = len(indices)
    risks = {name: np.zeros(Q, dtype=float) for name in METHODS}
    ranges = {name: dict(minimum_raw_inverse=float('inf'), maximum_raw_inverse=float('-inf'),
                         minimum_inverse=float('inf'), maximum_inverse=float('-inf'),
                         projected_cells=0, prediction_cells=0) for name in models}
    projection_max_increase = 0.
    for start in range(0, N, int(chunk_size)):
        stop = min(N, start + int(chunk_size)); rows = np.arange(start, stop)
        xb = np.asarray(x[start:stop], dtype=float); st = steps[start:stop]
        pb = probabilities[start:stop, None, :]
        scale = query_scale(q, indices, rows, st, core.P_LANDMARK)
        truth = scale / pb
        pilot = scale / pbar[st, None, :]
        correction = np.zeros_like(scale)
        for t in range(core.N_STEPS):
            mask = st == t
            if mask.any():
                correction[mask] = np.einsum('nd,qad->nqa', xb[mask], beta[:, t], optimize=True)
        raw_B1 = pilot + correction
        projected_B1 = project_representer(raw_B1, scale)
        original_error = pb * (raw_B1 - truth) ** 2
        projected_error = pb * (projected_B1 - truth) ** 2
        projection_max_increase = max(projection_max_increase,
                                      float(np.max(projected_error - original_error)))
        for name, fitted in [('pilot', pilot), ('regularized_B1', raw_B1),
                              ('projected_regularized_B1', projected_B1)]:
            risks[name] += np.sum(pb * (fitted - truth) ** 2, axis=(0, 2)) / N
        for name, model in models.items():
            raw = model.predict_raw(xb, st, indices)
            inverse = np.clip(raw, 1., 100.)
            if raw.shape != (len(xb), Q, 4) or not np.isfinite(raw).all():
                raise FloatingPointError('Invalid frozen sparse inverse prediction for ' + name)
            # Evaluate the actual projected function used during selection.
            risks[name] += np.sum(pb * (scale * inverse - truth) ** 2, axis=(0, 2)) / N
            r = ranges[name]
            r['minimum_raw_inverse'] = min(r['minimum_raw_inverse'], float(raw.min()))
            r['maximum_raw_inverse'] = max(r['maximum_raw_inverse'], float(raw.max()))
            r['minimum_inverse'] = min(r['minimum_inverse'], float(inverse.min()))
            r['maximum_inverse'] = max(r['maximum_inverse'], float(inverse.max()))
            r['projected_cells'] += int(np.count_nonzero(raw != inverse))
            r['prediction_cells'] += int(raw.size)
    for name, risk in risks.items():
        if not np.isfinite(risk).all() or np.any(risk < 0.):
            raise FloatingPointError('Invalid exact population risk for ' + name)
    np.testing.assert_array_less(risks['projected_regularized_B1'], risks['regularized_B1'] + 1e-8)
    return risks, dict(prediction_ranges=ranges,
                       maximum_pointwise_projection_risk_increase=projection_max_increase,
                       projected_B1_risk_bound_verified=True,
                       all_population_predictions_finite=True)


def run_replication(data, sample_size, seed, chunk_size=2048, progress=None, freeze_callback=None):
    started = time.perf_counter(); core = data['core']; q = data['queries']
    artifacts, pbar = draw_artifacts(data, sample_size, seed)
    if progress:
        progress('Selecting original B1 and shared/independent sparse Riesz on the same two observed folds.')
    beta, base_choice, old_table = fit_original_B1(core, q, artifacts)
    fitted = sparse_model.fit_observed(artifacts, q, pbar, core, progress)
    expected = set(METHODS[3:])
    if set(fitted['models']) != expected or set(fitted['choices']) != expected:
        raise RuntimeError('Sparse model family set changed.')
    selection_seconds = time.perf_counter() - started
    choices = [dict(method='pilot', gamma=0.), dict(method='regularized_B1', **base_choice),
               dict(method='projected_regularized_B1', **base_choice, postfit_projection=True)]
    choices.extend(dict(method=name, **spec) for name, spec in fitted['choices'].items())
    freeze = dict(version=VERSION, sample_size=int(sample_size), seed=int(seed),
                  selected_parameters=choices, checks=fitted['checks'],
                  all_model_choices_and_final_fits_complete=True,
                  population_risk_evaluation_started=False,
                  known_pilot_maximum_inverse=float(np.max(1 / pbar)))
    if freeze_callback is not None:
        freeze_callback(freeze)
    if progress:
        progress('All selected models frozen; integrating exact held-query MSE over the fixed population.')
    held = np.asarray(q['indices']['held_out'], dtype=int)
    risks, evaluation_checks = evaluate_frozen(data, beta, fitted['models'], held, chunk_size)
    records = []
    for name in METHODS:
        for local, qi in enumerate(held):
            query = data['query_library'].iloc[qi]
            records.append(dict(sample_size=int(sample_size), seed=int(seed), method=name,
                                query_id=str(query.query_id), support_family=str(query.support_family),
                                oracle_M2=float(risks[name][local]), status='ok'))
    table = pd.concat([old_table, fitted['selection_table']], ignore_index=True)
    selected = pd.DataFrame(choices); logs = fitted['fit_diagnostics'].copy()
    for frame in (table, selected, logs):
        frame.insert(0, 'seed', int(seed)); frame.insert(0, 'sample_size', int(sample_size))
    metadata = dict(version=VERSION, sample_size=int(sample_size), seed=int(seed),
                    selection_fit_seconds=float(selection_seconds),
                    total_seconds=float(time.perf_counter() - started),
                    basis_dimension=int(data['x_population'].shape[1]),
                    minimum_observed_action_landmark_count=int(artifacts[2]['counts'].min()),
                    known_pilot_maximum_inverse=float(np.max(1 / pbar)),
                    **fitted['checks'], **evaluation_checks)
    return dict(query_results=pd.DataFrame(records), selection_table=table,
                selected_parameters=selected, fit_diagnostics=logs, metadata=metadata)
