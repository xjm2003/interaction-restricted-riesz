"""Frozen, single-index CVD semi-synthetic Riesz benchmark protocol.

Only the covariate distribution comes from the cohort.  The four-arm assignment
mechanism below is a known simulator, calibrated to the audit's joint exposure
frequencies; it is not an estimate of clinical treatment propensities.  Neither
observed outcomes nor oracle quantities are used to choose features, queries,
hyperparameters, or model coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ACTION_ORDER = ["neither", "antihtn_only", "statin_only", "both"]
REFERENCE_COUNTS = np.array([70705, 27880, 10435, 18131], dtype=np.int64)
TARGET_MARGINALS = REFERENCE_COUNTS / REFERENCE_COUNTS.sum()
UNIFORM_MIXING = 0.04
QUERY_SEED = 20260915
ORACLE_EIGENVALUE_RELATIVE_CUTOFF = 1e-10
REQUIRED_Z = ("age", "sbp", "bmi", "tc", "hdlc", "eGFR")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _dual_objective(logits: np.ndarray, intercepts: np.ndarray,
                    target: np.ndarray) -> float:
    values = logits + intercepts
    maximum = values.max(axis=1)
    logsumexp = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    return float(logsumexp.mean() - target @ intercepts)


def _calibrate_intercepts(logits: np.ndarray,
                          tolerance: float = 1e-10,
                          max_iterations: int = 1000,
                          ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrate multinomial intercepts; last-arm intercept is the reference.

    Calibration takes place before adding the uniform component.  Consequently
    the final probabilities satisfy both p_a >= .01 and the target marginal,
    unlike clipping individual probabilities followed by renormalization.
    """
    softmax_target = (TARGET_MARGINALS - UNIFORM_MIXING / 4.0) / (1.0 - UNIFORM_MIXING)
    if np.any(softmax_target <= 0):
        raise ValueError("Target marginal is incompatible with the uniform mixture.")
    intercepts = np.log(softmax_target)
    intercepts -= intercepts[-1]
    for iteration in range(max_iterations + 1):
        soft = _softmax(logits + intercepts)
        probabilities = (1.0 - UNIFORM_MIXING) * soft + UNIFORM_MIXING / 4.0
        error = float(np.max(np.abs(probabilities.mean(axis=0) - TARGET_MARGINALS)))
        if error <= tolerance:
            return probabilities, intercepts, {
                "method": "multinomial_intercept_Newton_with_backtracking",
                "iterations": iteration,
                "tolerance": tolerance,
                "maximum_iterations": max_iterations,
                "maximum_absolute_marginal_error": error,
            }
        if iteration == max_iterations:
            break
        # The final intercept is fixed at zero to remove softmax nonidentifiability.
        gradient = soft.mean(axis=0)[:3] - softmax_target[:3]
        partial = soft[:, :3]
        hessian = np.diag(partial.mean(axis=0)) - partial.T @ partial / len(soft)
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Simulator intercept calibration Hessian is singular.") from exc
        objective = _dual_objective(logits, intercepts, softmax_target)
        slope = float(gradient @ direction)
        step = 1.0
        for _ in range(50):
            candidate = intercepts.copy()
            candidate[:3] -= step * direction
            candidate_objective = _dual_objective(logits, candidate, softmax_target)
            # A small absolute tolerance avoids false failures at machine precision.
            if candidate_objective <= objective - 1e-4 * step * slope + 1e-14:
                intercepts = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("Simulator intercept calibration line search failed.")
    raise RuntimeError(
        f"Simulator intercept calibration failed after {max_iterations} iterations; "
        f"maximum marginal error is {error:.3g}."
    )


def _make_queries(tilt_features: np.ndarray, query_seed: int = QUERY_SEED
                  ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if not isinstance(query_seed, (int, np.integer)) or query_seed < 0:
        raise ValueError("query_seed must be a nonnegative integer")
    rng = np.random.default_rng(query_seed)
    etas = np.vstack([np.zeros(3), rng.uniform(-0.8, 0.8, size=(27, 3))])
    permutation = rng.permutation(np.arange(1, 28))
    group_sets = {
        "train": np.concatenate([[0], permutation[:17]]),
        "selection": permutation[17:21],
        "held_out": permutation[21:],
    }
    group_split = {
        int(group): split for split, groups in group_sets.items() for group in groups
    }
    contrasts = []
    for first in range(4):
        for second in range(first + 1, 4):
            coefficients = np.zeros(4)
            coefficients[first] = -1.0
            coefficients[second] = 1.0
            contrasts.append((
                f"{ACTION_ORDER[second]}_vs_{ACTION_ORDER[first]}",
                "pairwise", coefficients,
            ))
    contrasts.extend([
        ("statin_main_effect", "main_effect", np.array([-0.5, -0.5, 0.5, 0.5])),
        ("antihtn_main_effect", "main_effect", np.array([-0.5, 0.5, -0.5, 0.5])),
    ])
    monomial_names = [
        "one", "eta0", "eta1", "eta2", "eta0_sq", "eta1_sq", "eta2_sq",
        "eta0_eta1", "eta0_eta2", "eta1_eta2",
    ]
    descriptor_columns = [
        f"z__c{arm}_{name}" for arm in range(3) for name in monomial_names
    ]
    rows = []
    for tilt_id, eta in enumerate(etas):
        e0, e1, e2 = eta
        monomials = np.array([1.0, e0, e1, e2, e0*e0, e1*e1, e2*e2,
                              e0*e1, e0*e2, e1*e2])
        for contrast_name, family, coefficients in contrasts:
            row = {
                "query_id": f"q{len(rows):03d}",
                "query_split": group_split[tilt_id],
                "tilt_id": int(tilt_id),
                "contrast": contrast_name,
                "support_family": family,
                "h_0": 1.0,
            }
            row.update({f"c__{action}": float(coefficients[arm])
                        for arm, action in enumerate(ACTION_ORDER)})
            row.update({f"eta_{index}": float(value) for index, value in enumerate(eta)})
            descriptor = (coefficients[:3, None] * monomials[None, :]).ravel()
            row.update(dict(zip(descriptor_columns, descriptor.tolist())))
            rows.append(row)
    library = pd.DataFrame(rows)
    raw_omega = np.exp(np.clip(etas @ tilt_features.T, -2.0, 2.0))
    omega = raw_omega / raw_omega.mean(axis=1, keepdims=True)
    split = library["query_split"].to_numpy()
    indices = {name: np.flatnonzero(split == name)
               for name in ("train", "selection", "held_out")}
    indices["fit_final"] = np.flatnonzero(np.isin(split, ["train", "selection"]))
    queries = {
        "h": library[["h_0"]].to_numpy(dtype=float),
        "c": library[[f"c__{action}" for action in ACTION_ORDER]].to_numpy(dtype=float),
        "z": library[descriptor_columns].to_numpy(dtype=float),
        "inverse_eta": library["tilt_id"].to_numpy(dtype=int),
        "omega_unique": omega,
        "indices": indices,
    }
    report = {
        "seed": int(query_seed),
        "query_count": len(library),
        "tilt_group_count": len(etas),
        "split_unit": "entire_tilt_group_all_eight_contrasts",
        "split_counts": {name: len(indices[name]) for name in group_sets},
        "split_tilt_ids": {name: sorted(map(int, groups))
                           for name, groups in group_sets.items()},
        "eta_generation": "zero_vector_plus_27_rng_uniform(-0.8,0.8)_vectors",
        "eta_by_tilt_id": etas.tolist(),
        "tilt_features": ["clip(z_age,-2,2)/2", "clip(z_sbp,-2,2)/2",
                          "clip(-z_eGFR,-2,2)/2"],
        "omega": "exp(clip(eta_dot_tilt,-2,2)) / finite_population_mean",
        "contrast_definition": "six_later_arm_minus_earlier_arm_pairs_plus_two_averaged_main_effects",
        "contrast_normalization": "pairwise_coefficients_are_minus1_plus1; main_effect_coefficients_are_plus_or_minus0.5",
        "descriptor": "c_first_three_tensor_[1,eta0,eta1,eta2,eta0^2,eta1^2,eta2^2,eta0*eta1,eta0*eta2,eta1*eta2]",
        "descriptor_columns": descriptor_columns,
        "descriptor_dimension": len(descriptor_columns),
        "training_descriptor_rank": int(np.linalg.matrix_rank(queries["z"][indices["train"]])),
        "held_out_has_no_training_tilt_groups": True,
        "real_outcomes_used": False,
    }
    return queries, library, report


def _oracle_objects(x: np.ndarray, probabilities: np.ndarray,
                    queries: dict[str, Any],
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Population truth for evaluation only, using exact finite-population means."""
    n, d = x.shape
    omega = queries["omega_unique"]
    inverse = queries["inverse_eta"]
    c = queries["c"]
    n_queries = len(c)
    pbar = probabilities.mean(axis=0)
    grams = np.empty((1, 4, d, d), dtype=float)
    moments = np.empty((n_queries, 1, 4, d), dtype=float)
    pilot_m2 = np.zeros(n_queries, dtype=float)
    projected_energy = np.zeros(n_queries, dtype=float)
    whitened_train_blocks = []
    gram_diagnostics = []
    train_indices = queries["indices"]["train"]
    for arm in range(4):
        p_arm = probabilities[:, arm]
        gram = x.T @ (p_arm[:, None] * x) / n
        gram = (gram + gram.T) / 2.0
        grams[0, arm] = gram
        # Compute only 28 tilt-group moments, then scale to all eight contrasts.
        group_moments = omega @ ((1.0 - p_arm / pbar[arm])[:, None] * x) / n
        moments[:, 0, arm, :] = c[:, arm, None] * group_moments[inverse]
        residual_second_moment = (1.0 - p_arm / pbar[arm])**2 / p_arm
        group_energy = (omega * omega) @ residual_second_moment / n
        pilot_m2 += c[:, arm]**2 * group_energy[inverse]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        maximum = float(max(eigenvalues[-1], 0.0))
        keep = eigenvalues > ORACLE_EIGENVALUE_RELATIVE_CUTOFF * maximum
        if maximum <= 0 or not np.any(keep):
            raise ValueError(f"Population Gram has no positive directions for arm {arm}.")
        whitening = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
        whitened = moments[:, 0, arm, :] @ whitening
        projected_energy += np.sum(whitened**2, axis=1)
        whitened_train_blocks.append(whitened[train_indices])
        gram_diagnostics.append({
            "arm": ACTION_ORDER[arm], "dimension": d,
            "oracle_retained_rank": int(keep.sum()),
            "eigenvalue_minimum": float(eigenvalues[0]),
            "eigenvalue_maximum": maximum,
            "smallest_retained_eigenvalue": float(eigenvalues[keep].min()),
        })
    raw_floor = pilot_m2 - projected_energy
    tolerance = 1e-8 * np.maximum(1.0, pilot_m2)
    if np.any(raw_floor < -tolerance):
        raise FloatingPointError("Oracle projection produced a materially negative risk floor.")
    floor = np.maximum(raw_floor, 0.0)
    singular_values = np.linalg.svd(np.concatenate(whitened_train_blocks, axis=1),
                                    compute_uv=False)
    energy = singular_values**2
    total = float(energy.sum())
    participation_rank = (total**2 / float(np.sum(energy**2))) if total > 0 else 0.0
    spectrum = {
        "use": "diagnostic_only_never_model_fitting_or_hyperparameter_selection",
        "queries": "training_split_only",
        "matrix": "training_query_oracle_correction_moments_whitened_by_population_Gram",
        "singular_values": singular_values.tolist(),
        "total_projected_energy": total,
        "effective_rank_participation_ratio": float(participation_rank),
        "top_rank_energy_fraction": {
            str(rank): float(energy[:rank].sum() / total) if total > 0 else 0.0
            for rank in (2, 4, 8, 16)
        },
        "rank_threshold_relative_to_maximum_singular_value": 1e-10,
        "numerical_rank": int(np.sum(singular_values > 1e-10 * singular_values[0]))
            if len(singular_values) and singular_values[0] > 0 else 0,
    }
    report = {
        "population_gram": "mean_X[p_a(X) * phi(X) * phi(X)^T]",
        "oracle_correction_moment": "mean_X[omega_q(X) * c_qa * (1-p_a(X)/pbar_a) * phi(X)]",
        "pilot_correction_M2": "mean_X[sum_a omega_q(X)^2 * c_qa^2 * (1-p_a(X)/pbar_a)^2 / p_a(X)]",
        "risk": "pilot_correction_M2 - 2*sum_a beta_qa^T*m_qa + sum_a beta_qa^T*G_a*beta_qa",
        "sieve_floor": "pilot_correction_M2 - sum_a m_qa^T*pinv(G_a)*m_qa",
        "oracle_gram_eigenvalue_relative_cutoff": ORACLE_EIGENVALUE_RELATIVE_CUTOFF,
        "gram_blocks": gram_diagnostics,
        "minimum_unclipped_floor": float(raw_floor.min()),
        "numerically_negative_floors_clipped_to_zero": int(np.sum(raw_floor < 0)),
        "training_oracle_correction_spectrum": spectrum,
    }
    return grams, moments, pilot_m2, floor, report


def build_protocol(x: np.ndarray, zdict: dict[str, np.ndarray], *, query_seed: int = QUERY_SEED
                   ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame,
                              np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the fixed protocol; x and standardized covariates must align by row.

    Returns population, prepared queries, query-library dataframe, population
    Grams, oracle correction moments, approximation floor, and a JSON-serializable
    report.  Population contains no patient identifiers or observed outcomes.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite, nonempty two-dimensional feature matrix with >=2 rows.")
    standardized = {}
    for name in REQUIRED_Z:
        if name not in zdict:
            raise ValueError(f"Missing required standardized simulator covariate: {name}")
        values = np.asarray(zdict[name], dtype=float)
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError(f"Covariate {name} must be finite and have shape ({len(x)},).")
        standardized[name] = values
    z = standardized
    logits = np.column_stack([
        np.zeros(len(x)),
        0.65*z["sbp"] + 0.20*z["age"] + 0.10*z["bmi"],
        0.55*z["tc"] - 0.25*z["hdlc"] + 0.20*z["age"],
        0.35*z["sbp"] + 0.35*z["tc"] + 0.25*z["age"] - 0.20*z["eGFR"]
        + 0.15*np.tanh(z["sbp"] * z["age"]),
    ])
    probabilities, intercepts, calibration = _calibrate_intercepts(logits)
    tilt_features = np.column_stack([
        np.clip(z["age"], -2.0, 2.0) / 2.0,
        np.clip(z["sbp"], -2.0, 2.0) / 2.0,
        np.clip(-z["eGFR"], -2.0, 2.0) / 2.0,
    ])
    queries, library, query_report = _make_queries(tilt_features, query_seed=query_seed)
    grams, moments, pilot_m2, floor, oracle_report = _oracle_objects(x, probabilities, queries)
    library["pilot_correction_M2"] = pilot_m2
    library["approximation_floor"] = floor
    population = pd.DataFrame({"step": np.zeros(len(x), dtype=int)})
    for arm, name in enumerate(ACTION_ORDER):
        population[f"p_star__{name}"] = probabilities[:, arm]
    quantile_levels = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    report = {
        "protocol": "cvd_single_index_semisynthetic_v1",
        "population_rows": len(x), "basis_dimension": x.shape[1],
        "N_STEPS": 1, "P_LANDMARK": 1.0, "action_order": ACTION_ORDER,
        "action_encoding": ["statin=0,antihtn=0", "statin=0,antihtn=1",
                            "statin=1,antihtn=0", "statin=1,antihtn=1"],
        "marginal_p": [probabilities.mean(axis=0).tolist()],
        "target": "finite_empirical_covariate_population_representer_squared_error",
        "representer": "alpha_q(X,a)=omega_q(X)*c_qa/p_a(X)",
        "pilot": "alpha0_q(X,a)=omega_q(X)*c_qa/pbar_a",
        "observed_treatments_used_for_sampling": False,
        "observed_outcomes_used": False,
        "causal_effect_estimation": False,
        "assignment": {
            "mechanism": "known_synthetic_softmax_assignment_with_uniform_mixture",
            "audit_reference_counts": REFERENCE_COUNTS.tolist(),
            "audit_reference_denominator": int(REFERENCE_COUNTS.sum()),
            "target_marginals": TARGET_MARGINALS.tolist(),
            "actual_marginals": probabilities.mean(axis=0).tolist(),
            "interpretation": "audit_frequencies_only_calibrate_a_simulator_not_clinical_propensity_estimates",
            "logits_before_intercepts": {
                "neither": "0",
                "antihtn_only": "0.65*z_sbp + 0.20*z_age + 0.10*z_bmi",
                "statin_only": "0.55*z_tc - 0.25*z_hdlc + 0.20*z_age",
                "both": "0.35*z_sbp + 0.35*z_tc + 0.25*z_age - 0.20*z_eGFR + 0.15*tanh(z_sbp*z_age)",
            },
            "calibrated_intercepts": intercepts.tolist(),
            "intercept_reference_arm": "both",
            "uniform_mixing": UNIFORM_MIXING,
            "guaranteed_minimum_probability": UNIFORM_MIXING / 4,
            "probability_minimum": float(probabilities.min()),
            "probability_maximum": float(probabilities.max()),
            "quantile_levels": quantile_levels,
            "probability_quantiles_by_arm": {
                name: np.quantile(probabilities[:, arm], quantile_levels).tolist()
                for arm, name in enumerate(ACTION_ORDER)
            },
            "calibration": calibration,
        },
        "queries": query_report,
        "oracle": oracle_report,
    }
    return population, queries, library, grams, moments, floor, report
