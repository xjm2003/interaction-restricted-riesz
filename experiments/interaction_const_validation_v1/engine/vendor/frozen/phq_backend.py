#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_run_qaoct_ridge_shrinkage_formal.py

Formal QAOCT Direct-RR representer study with:
1. one action draw per replication;
2. observation cross-fitting;
3. blockwise ridge-regularized empirical inverse;
4. cross-fitted global amplitude calibration gamma in [0,1];
5. rank-zero pilot fallback;
6. separate tuning for Direct-RR, global descriptor, and regularized B1.

This is the confirmatory representer-risk run. The protocol, grids, sample sizes, seeds, estimands, and decision margins are frozen before execution.

For each empirical block with eigendecomposition G = V diag(lambda) V^T,
directions below the frozen 3e-6 relative threshold are removed, then

    W_kappa = V_keep diag(
        (lambda_keep + kappa * lambda_max)^(-1/2)
    ).

The empirical target is y_hat = r_hat W_kappa and a raw coefficient prediction
is beta_hat = y_pred W_kappa^T.

For any candidate direction beta, the validation-fold Riesz objective is

    J_val(gamma beta)
      = gamma^2 beta^T G_val beta
        - 2 gamma beta^T r_val.

A single gamma is pooled across the two observation-fold directions and the
32 selection queries:

    gamma_hat = clip(
        sum beta^T r_val / sum beta^T G_val beta,
        0, 1
    ).

This separates direction learning from finite-sample amplitude stabilization.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COL = "PATID"
STEP_COL = "step"

ACTION_ORDER = [
    "no_active_treatment",
    "antidepressant_only",
    "psychotherapy_only",
    "combined",
]
N_STEPS = 4
N_ACTIONS = 4
P_LANDMARK = 0.25

FORMAL_SEEDS = list(range(10001, 10101))
DEFAULT_SAMPLE_SIZES = [12_000, 27_038, 54_076, 108_152]

FROZEN_RELATIVE_TOLERANCE = 3e-6
MIN_BLOCK_RANK = 8

BLOCK_RIDGE_GRID = [
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
]
DESCRIPTOR_RIDGE_GRID = [0.0, 1e-2, 1e-1, 1.0, 10.0]
RANK_GRID = [0, 4, 8, 12, 16, 24, 32, 40]
GAMMA_LOWER = 0.0
GAMMA_UPPER = 1.0

# Frozen confirmatory margins, selected after development and before formal
# seeds were assigned.
PILOT_SUPERIORITY_MARGIN = 1.0
SAME_DICTIONARY_EQUIVALENCE_MARGIN = 1.02
REGULARIZED_B1_NONINFERIORITY_MARGIN = 1.05


def parse_args() -> argparse.Namespace:
    home = Path.home()
    root = home / "qaoct_semisynthetic_v1"
    parser = argparse.ArgumentParser(
        description="Run the frozen formal QAOCT Direct-RR representer study."
    )
    parser.add_argument(
        "--history-bank",
        type=Path,
        default=root / "history_bank_v1_2" / "history_bank_v1_2.csv.gz",
    )
    parser.add_argument(
        "--propensity-scores",
        type=Path,
        default=root / "propensity_generator_v1" / "propensity_scores_oof_v1.csv.gz",
    )
    parser.add_argument(
        "--query-library",
        type=Path,
        default=root / "query_library_v1" / "query_library_v1.csv",
    )
    parser.add_argument(
        "--history-basis",
        type=Path,
        default=root / "sieve_v1_1" / "history_basis_X_v1_1.npy",
    )
    parser.add_argument(
        "--history-basis-row-index",
        type=Path,
        default=root / "sieve_v1_1" / "history_basis_row_index_v1_1.csv.gz",
    )
    parser.add_argument(
        "--population-grams",
        type=Path,
        default=root / "sieve_v1_1" / "sieve_block_grams_v1_1.npz",
    )
    parser.add_argument(
        "--oracle-moments",
        type=Path,
        default=root / "sieve_v1_1" / "oracle_correction_moments_v1_1.npy",
    )
    parser.add_argument(
        "--selected-projection",
        type=Path,
        default=root / "sieve_tolerance_v1" / "selected_projection_by_query.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ridge_shrinkage_formal_v1",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_SIZES,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=FORMAL_SEEDS,
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def validate_query_library(query_library: pd.DataFrame) -> list[str]:
    required = {
        "query_id",
        "query_split",
        "support_family",
        "pilot_correction_M2",
        *[f"h_{t}" for t in range(N_STEPS)],
        *[f"c__{a}" for a in ACTION_ORDER],
        *[f"eta_{j}" for j in range(3)],
    }
    missing = sorted(required - set(query_library.columns))
    if missing:
        raise ValueError(f"Query library missing columns: {missing}")
    if query_library["query_id"].duplicated().any():
        raise ValueError("Duplicate query IDs.")
    counts = query_library["query_split"].value_counts().to_dict()
    if counts != {"train": 144, "held_out": 48, "selection": 32}:
        raise ValueError(f"Unexpected query split counts: {counts}")

    descriptor_columns = [
        column
        for column in query_library.columns
        if column.startswith("z__")
    ]
    if not descriptor_columns:
        raise ValueError("No z__ descriptor columns.")
    return descriptor_columns


def align_population(
    history: pd.DataFrame,
    propensity: pd.DataFrame,
    row_index: pd.DataFrame,
    x: np.ndarray,
) -> pd.DataFrame:
    for frame in [history, propensity, row_index]:
        frame[ID_COL] = frame[ID_COL].astype(str)
        frame[STEP_COL] = pd.to_numeric(
            frame[STEP_COL], errors="raise"
        ).astype(int)

    history_columns = [
        ID_COL,
        STEP_COL,
        "baseline_phq9",
        "tvx_has_post_phq_before_step",
        "tvx_post_phq_last_change_from_baseline",
        "prior_antidepressant_exposure_count",
        "prior_psychotherapy_exposure_count",
    ]
    probability_columns = [
        f"p_star__{action}" for action in ACTION_ORDER
    ]

    for name, frame, columns in [
        ("history", history, history_columns),
        (
            "propensity",
            propensity,
            [ID_COL, STEP_COL, *probability_columns],
        ),
        (
            "row index",
            row_index,
            [ID_COL, STEP_COL, "row_number"],
        ),
    ]:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if frame.duplicated([ID_COL, STEP_COL]).any():
            raise ValueError(f"{name} has duplicate patient-step rows.")

    population = row_index[
        [ID_COL, STEP_COL, "row_number"]
    ].merge(
        history[history_columns],
        on=[ID_COL, STEP_COL],
        how="left",
        validate="one_to_one",
    )
    population = population.merge(
        propensity[[ID_COL, STEP_COL, *probability_columns]],
        on=[ID_COL, STEP_COL],
        how="left",
        validate="one_to_one",
    )
    population = population.sort_values(
        "row_number",
        kind="mergesort",
    ).reset_index(drop=True)

    if not np.array_equal(
        population["row_number"].to_numpy(dtype=int),
        np.arange(len(population)),
    ):
        raise ValueError("Basis row index is not contiguous.")
    if len(population) != len(x):
        raise ValueError("Population and basis row counts differ.")

    probabilities = population[
        probability_columns
    ].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite propensities.")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError("Propensities do not sum to one.")
    if probabilities.min() < 0.01 - 1e-10:
        raise ValueError("Propensity floor is violated.")
    if not np.isfinite(x).all():
        raise ValueError("History basis contains nonfinite values.")

    return population


def build_tilt_features(population: pd.DataFrame) -> np.ndarray:
    severity = (
        pd.to_numeric(
            population["baseline_phq9"], errors="coerce"
        )
        .fillna(15.0)
        .to_numpy(dtype=float)
    )
    severity_z = np.clip(
        (severity - 15.0) / 5.0, -2.0, 2.0
    )

    has_recent = (
        pd.to_numeric(
            population["tvx_has_post_phq_before_step"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    recent_change = (
        pd.to_numeric(
            population[
                "tvx_post_phq_last_change_from_baseline"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    recent_z = (
        np.clip(recent_change / 5.0, -2.0, 2.0)
        * (has_recent > 0)
    )

    prior_ad = (
        pd.to_numeric(
            population[
                "prior_antidepressant_exposure_count"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prior_psy = (
        pd.to_numeric(
            population[
                "prior_psychotherapy_exposure_count"
            ],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    burden = np.clip(
        (prior_ad + prior_psy) / 2.0, 0.0, 2.0
    )

    features = np.column_stack(
        [severity_z, recent_z, burden]
    )
    if not np.isfinite(features).all():
        raise ValueError("Tilt features contain nonfinite values.")
    return features


def normalized_omega(
    steps: np.ndarray,
    tilt_features: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    raw = np.exp(
        np.clip(tilt_features @ eta, -2.0, 2.0)
    )
    omega = np.zeros(len(raw), dtype=float)
    for step in range(N_STEPS):
        mask = steps == step
        omega[mask] = raw[mask] / raw[mask].mean()
    return omega


def prepare_queries(
    query_library: pd.DataFrame,
    descriptor_columns: list[str],
    population_steps: np.ndarray,
    tilt_features: np.ndarray,
) -> dict[str, Any]:
    h = query_library[
        [f"h_{step}" for step in range(N_STEPS)]
    ].to_numpy(dtype=float)
    c = query_library[
        [f"c__{action}" for action in ACTION_ORDER]
    ].to_numpy(dtype=float)
    eta = query_library[
        [f"eta_{index}" for index in range(3)]
    ].to_numpy(dtype=float)
    z = query_library[
        descriptor_columns
    ].to_numpy(dtype=float)

    unique_eta, inverse_eta = np.unique(
        eta, axis=0, return_inverse=True
    )
    omega_unique = np.vstack(
        [
            normalized_omega(
                population_steps,
                tilt_features,
                value,
            )
            for value in unique_eta
        ]
    )

    split = query_library["query_split"].to_numpy()
    indices = {
        "train": np.where(split == "train")[0],
        "selection": np.where(
            split == "selection"
        )[0],
        "held_out": np.where(
            split == "held_out"
        )[0],
        "fit_final": np.where(
            np.isin(split, ["train", "selection"])
        )[0],
    }
    return {
        "h": h,
        "c": c,
        "z": z,
        "inverse_eta": inverse_eta,
        "omega_unique": omega_unique,
        "indices": indices,
    }


def draw_actions(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    uniforms = rng.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=1)
    actions = (
        uniforms[:, None] > cumulative
    ).sum(axis=1)
    return np.minimum(
        actions, N_ACTIONS - 1
    ).astype(int)


def make_sample_artifact(
    population_rows: np.ndarray,
    steps: np.ndarray,
    actions: np.ndarray,
    x: np.ndarray,
    queries: dict[str, Any],
    marginal_p: np.ndarray,
) -> dict[str, Any]:
    n = len(population_rows)
    d = x.shape[1]
    n_queries = len(queries["h"])

    omega_sample = queries["omega_unique"][
        queries["inverse_eta"][:, None],
        population_rows[None, :],
    ]
    h_sample = queries["h"][
        np.arange(n_queries)[:, None],
        steps[None, :],
    ]
    base = (
        h_sample / P_LANDMARK * omega_sample
    )

    raw_moments = np.zeros(
        (n_queries, N_STEPS, N_ACTIONS, d),
        dtype=np.float64,
    )
    grams = np.zeros(
        (N_STEPS, N_ACTIONS, d, d),
        dtype=np.float64,
    )
    eigvals = np.zeros(
        (N_STEPS, N_ACTIONS, d),
        dtype=np.float64,
    )
    eigvecs = np.zeros(
        (N_STEPS, N_ACTIONS, d, d),
        dtype=np.float64,
    )
    counts = np.zeros(
        (N_STEPS, N_ACTIONS),
        dtype=int,
    )

    for step in range(N_STEPS):
        step_mask = steps == step
        x_step = x[step_mask]
        actions_step = actions[step_mask]

        for action_index in range(N_ACTIONS):
            block_mask = (
                step_mask
                & (actions == action_index)
            )
            counts[step, action_index] = int(
                block_mask.sum()
            )
            if counts[step, action_index] == 0:
                raise RuntimeError(
                    f"Empty block step={step}, "
                    f"action={ACTION_ORDER[action_index]}"
                )

            x_block = x[block_mask]
            gram = (x_block.T @ x_block) / n
            gram = (gram + gram.T) / 2.0
            grams[step, action_index] = gram

            values, vectors = np.linalg.eigh(gram)
            values = np.maximum(values, 0.0)
            order = np.argsort(values)[::-1]
            eigvals[step, action_index] = values[order]
            eigvecs[step, action_index] = vectors[:, order]

            pbar = marginal_p[step, action_index]
            pilot_factor = (
                1.0
                - (
                    actions_step == action_index
                ).astype(float)
                / pbar
            )
            weights = (
                base[:, step_mask]
                * queries["c"][
                    :, action_index
                ][:, None]
                * pilot_factor[None, :]
            )
            raw_moments[
                :, step, action_index, :
            ] = (weights @ x_step) / n

    return {
        "rows": population_rows,
        "steps": steps,
        "actions": actions,
        "x": x,
        "n": n,
        "raw_moments": raw_moments,
        "grams": grams,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "counts": counts,
    }


def regularized_state(
    artifact: dict[str, Any],
    block_ridge: float,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    target_parts: list[np.ndarray] = []
    offset = 0
    ranks = []
    effective_conditions = []

    for step in range(N_STEPS):
        for action_index, action_name in enumerate(
            ACTION_ORDER
        ):
            values = artifact["eigvals"][
                step, action_index
            ]
            vectors = artifact["eigvecs"][
                step, action_index
            ]
            maximum = float(values[0])
            keep = values > (
                FROZEN_RELATIVE_TOLERANCE
                * maximum
            )
            rank = int(keep.sum())
            if rank < MIN_BLOCK_RANK:
                raise RuntimeError(
                    f"Retained rank {rank} below minimum."
                )

            kept_values = values[keep]
            kept_vectors = vectors[:, keep]
            regularized_values = (
                kept_values
                + block_ridge * maximum
            )
            whitener = (
                kept_vectors
                / np.sqrt(
                    regularized_values
                )[None, :]
            )

            moment = artifact["raw_moments"][
                :, step, action_index, :
            ]
            target_parts.append(
                moment @ whitener
            )
            blocks.append(
                {
                    "step": step,
                    "action_index": action_index,
                    "action": action_name,
                    "whitener": whitener,
                    "offset_start": offset,
                    "offset_end": offset + rank,
                    "rank": rank,
                }
            )
            offset += rank
            ranks.append(rank)
            effective_conditions.append(
                float(
                    regularized_values.max()
                    / regularized_values.min()
                )
            )

    return {
        "blocks": blocks,
        "targets": np.concatenate(
            target_parts, axis=1
        ),
        "dimension": offset,
        "minimum_rank": int(min(ranks)),
        "maximum_rank": int(max(ranks)),
        "maximum_effective_condition": float(
            max(effective_conditions)
        ),
    }


def descriptor_designs(
    z: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_fit = z[fit_indices]
    mean = z_fit.mean(axis=0)
    sd = z_fit.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    x_fit = np.column_stack(
        [
            np.ones(len(fit_indices)),
            (z_fit - mean) / sd,
        ]
    )
    x_predict = np.column_stack(
        [
            np.ones(len(predict_indices)),
            (z[predict_indices] - mean)
            / sd,
        ]
    )
    return x_fit, x_predict


def full_descriptor_coefficient(
    design_fit: np.ndarray,
    targets_fit: np.ndarray,
    ridge: float,
) -> np.ndarray:
    penalty = np.eye(
        design_fit.shape[1]
    )
    penalty[0, 0] = 0.0
    normal = (
        design_fit.T @ design_fit
        + ridge * penalty
    )
    return (
        np.linalg.pinv(
            normal, rcond=1e-10
        )
        @ design_fit.T
        @ targets_fit
    )


def white_to_raw(
    white: np.ndarray,
    blocks: list[dict[str, Any]],
    raw_dimension: int,
) -> np.ndarray:
    beta = np.zeros(
        (
            len(white),
            N_STEPS,
            N_ACTIONS,
            raw_dimension,
        ),
        dtype=np.float64,
    )
    for block in blocks:
        beta[
            :,
            block["step"],
            block["action_index"],
            :,
        ] = (
            white[
                :,
                block["offset_start"]:
                block["offset_end"],
            ]
            @ block["whitener"].T
        )
    return beta


def objective_components(
    beta: np.ndarray,
    validation: dict[str, Any],
    query_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    moments = validation[
        "raw_moments"
    ][query_indices]
    cross = np.sum(
        beta * moments,
        axis=(1, 2, 3),
    )

    quadratic = np.zeros(
        len(beta), dtype=float
    )
    for step in range(N_STEPS):
        for action in range(N_ACTIONS):
            block_beta = beta[
                :, step, action, :
            ]
            quadratic += np.einsum(
                "qd,df,qf->q",
                block_beta,
                validation["grams"][
                    step, action
                ],
                block_beta,
                optimize=True,
            )
    return quadratic, cross


def calibrated_score(
    components: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[float, float]:
    total_quadratic = float(
        sum(
            quadratic.sum()
            for quadratic, _ in components
        )
    )
    total_cross = float(
        sum(
            cross.sum()
            for _, cross in components
        )
    )

    if total_quadratic <= 0:
        gamma = 0.0
    else:
        gamma = float(
            np.clip(
                total_cross
                / total_quadratic,
                GAMMA_LOWER,
                GAMMA_UPPER,
            )
        )

    objectives = np.concatenate(
        [
            gamma * gamma * quadratic
            - 2.0 * gamma * cross
            for quadratic, cross
            in components
        ]
    )
    return gamma, float(
        objectives.mean()
    )


def prediction_from_coefficient(
    design_predict: np.ndarray,
    coefficient: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if rank is None:
        return (
            design_predict @ coefficient,
            np.zeros(
                (
                    coefficient.shape[1],
                    0,
                )
            ),
        )

    if rank == 0:
        return (
            np.zeros(
                (
                    len(design_predict),
                    coefficient.shape[1],
                )
            ),
            np.zeros(
                (
                    coefficient.shape[1],
                    0,
                )
            ),
        )

    # Dictionary is built from fitted training values.
    raise RuntimeError(
        "Ranked prediction requires fitted_train."
    )


def select_models(
    queries: dict[str, Any],
    artifact_a: dict[str, Any],
    artifact_b: dict[str, Any],
    raw_dimension: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    train = queries["indices"]["train"]
    selection = queries[
        "indices"
    ]["selection"]
    z = queries["z"]

    design_train, design_selection = (
        descriptor_designs(
            z, train, selection
        )
    )

    rows: list[dict[str, Any]] = []

    # Rank-zero pilot fallback.
    rows.append(
        {
            "method_family": "direct_RR",
            "block_ridge": 0.0,
            "descriptor_ridge": 0.0,
            "rank": 0,
            "gamma": 0.0,
            "crossfit_objective": 0.0,
        }
    )

    states_a = {}
    states_b = {}
    for block_ridge in BLOCK_RIDGE_GRID:
        state_a = regularized_state(
            artifact_a, block_ridge
        )
        state_b = regularized_state(
            artifact_b, block_ridge
        )
        states_a[block_ridge] = state_a
        states_b[block_ridge] = state_b

        # Target-specific regularized B1.
        beta_ab = white_to_raw(
            state_a["targets"][selection],
            state_a["blocks"],
            raw_dimension,
        )
        beta_ba = white_to_raw(
            state_b["targets"][selection],
            state_b["blocks"],
            raw_dimension,
        )
        gamma, score = calibrated_score(
            [
                objective_components(
                    beta_ab,
                    artifact_b,
                    selection,
                ),
                objective_components(
                    beta_ba,
                    artifact_a,
                    selection,
                ),
            ]
        )
        rows.append(
            {
                "method_family": "regularized_B1",
                "block_ridge": block_ridge,
                "descriptor_ridge": 0.0,
                "rank": -1,
                "gamma": gamma,
                "crossfit_objective": score,
            }
        )

        for descriptor_ridge in (
            DESCRIPTOR_RIDGE_GRID
        ):
            coefficient_a = (
                full_descriptor_coefficient(
                    design_train,
                    state_a["targets"][train],
                    descriptor_ridge,
                )
            )
            coefficient_b = (
                full_descriptor_coefficient(
                    design_train,
                    state_b["targets"][train],
                    descriptor_ridge,
                )
            )

            fitted_a = (
                design_train
                @ coefficient_a
            )
            fitted_b = (
                design_train
                @ coefficient_b
            )
            prediction_full_a = (
                design_selection
                @ coefficient_a
            )
            prediction_full_b = (
                design_selection
                @ coefficient_b
            )

            # Global descriptor candidate.
            beta_ab = white_to_raw(
                prediction_full_a,
                state_a["blocks"],
                raw_dimension,
            )
            beta_ba = white_to_raw(
                prediction_full_b,
                state_b["blocks"],
                raw_dimension,
            )
            gamma, score = calibrated_score(
                [
                    objective_components(
                        beta_ab,
                        artifact_b,
                        selection,
                    ),
                    objective_components(
                        beta_ba,
                        artifact_a,
                        selection,
                    ),
                ]
            )
            rows.append(
                {
                    "method_family": (
                        "global_descriptor"
                    ),
                    "block_ridge": block_ridge,
                    "descriptor_ridge": (
                        descriptor_ridge
                    ),
                    "rank": -1,
                    "gamma": gamma,
                    "crossfit_objective": score,
                }
            )

            _, _, vt_a = np.linalg.svd(
                fitted_a,
                full_matrices=False,
            )
            _, _, vt_b = np.linalg.svd(
                fitted_b,
                full_matrices=False,
            )

            maximum_rank = min(
                vt_a.shape[0],
                vt_b.shape[0],
                int(
                    np.linalg.matrix_rank(
                        design_train
                    )
                ),
            )

            for rank in RANK_GRID:
                if rank == 0:
                    continue
                if rank > maximum_rank:
                    continue

                dictionary_a = vt_a[:rank].T
                dictionary_b = vt_b[:rank].T
                prediction_a = (
                    prediction_full_a
                    @ dictionary_a
                    @ dictionary_a.T
                )
                prediction_b = (
                    prediction_full_b
                    @ dictionary_b
                    @ dictionary_b.T
                )

                beta_ab = white_to_raw(
                    prediction_a,
                    state_a["blocks"],
                    raw_dimension,
                )
                beta_ba = white_to_raw(
                    prediction_b,
                    state_b["blocks"],
                    raw_dimension,
                )
                gamma, score = calibrated_score(
                    [
                        objective_components(
                            beta_ab,
                            artifact_b,
                            selection,
                        ),
                        objective_components(
                            beta_ba,
                            artifact_a,
                            selection,
                        ),
                    ]
                )
                rows.append(
                    {
                        "method_family": (
                            "direct_RR"
                        ),
                        "block_ridge": (
                            block_ridge
                        ),
                        "descriptor_ridge": (
                            descriptor_ridge
                        ),
                        "rank": rank,
                        "gamma": gamma,
                        "crossfit_objective": score,
                    }
                )

    table = pd.DataFrame(rows)

    selected = {}
    for family in [
        "direct_RR",
        "global_descriptor",
        "regularized_B1",
    ]:
        family_table = table[
            table["method_family"].eq(
                family
            )
        ]
        best = family_table.sort_values(
            [
                "crossfit_objective",
                "rank",
                "block_ridge",
                "descriptor_ridge",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
            kind="mergesort",
        ).iloc[0]
        selected[family] = {
            "block_ridge": float(
                best["block_ridge"]
            ),
            "descriptor_ridge": float(
                best["descriptor_ridge"]
            ),
            "rank": int(best["rank"]),
            "gamma": float(best["gamma"]),
            "crossfit_objective": float(
                best[
                    "crossfit_objective"
                ]
            ),
        }

    return selected, table


def fit_final_descriptor(
    queries: dict[str, Any],
    state: dict[str, Any],
    specification: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    fit_indices = queries[
        "indices"
    ]["fit_final"]
    held_indices = queries[
        "indices"
    ]["held_out"]

    design_fit, design_held = (
        descriptor_designs(
            queries["z"],
            fit_indices,
            held_indices,
        )
    )
    coefficient = (
        full_descriptor_coefficient(
            design_fit,
            state["targets"][
                fit_indices
            ],
            specification[
                "descriptor_ridge"
            ],
        )
    )
    prediction_full = (
        design_held @ coefficient
    )

    if specification["rank"] < 0:
        return (
            prediction_full
            * specification["gamma"],
            np.zeros(
                (
                    prediction_full.shape[1],
                    0,
                )
            ),
        )
    if specification["rank"] == 0:
        return (
            np.zeros_like(
                prediction_full
            ),
            np.zeros(
                (
                    prediction_full.shape[1],
                    0,
                )
            ),
        )

    fitted = (
        design_fit @ coefficient
    )
    _, _, vt = np.linalg.svd(
        fitted, full_matrices=False
    )
    rank = min(
        specification["rank"],
        vt.shape[0],
        int(
            np.linalg.matrix_rank(
                design_fit
            )
        ),
    )
    dictionary = vt[:rank].T
    prediction = (
        prediction_full
        @ dictionary
        @ dictionary.T
        * specification["gamma"]
    )
    return prediction, dictionary


def oracle_risk(
    beta: np.ndarray,
    total_m2: np.ndarray,
    oracle_moments: np.ndarray,
    population_grams: np.ndarray,
) -> np.ndarray:
    cross = np.sum(
        beta * oracle_moments,
        axis=(1, 2, 3),
    )
    quadratic = np.zeros(
        len(beta), dtype=float
    )
    for step in range(N_STEPS):
        for action in range(N_ACTIONS):
            b = beta[
                :, step, action, :
            ]
            quadratic += np.einsum(
                "qd,df,qf->q",
                b,
                population_grams[
                    step, action
                ],
                b,
                optimize=True,
            )
    risk = (
        total_m2
        - 2.0 * cross
        + quadratic
    )
    risk[
        (risk < 0)
        & (risk > -1e-8)
    ] = 0.0
    if np.any(risk < -1e-8):
        raise RuntimeError(
            "Materially negative oracle risk."
        )
    return risk


def summarize_replication(
    query_results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows = []
    for group_type, column in [
        ("overall", None),
        ("support_family", "support_family"),
    ]:
        groups = (
            [("all", query_results)]
            if column is None
            else query_results.groupby(
                column, sort=True
            )
        )
        for group_name, group in groups:
            for method, method_group in (
                group.groupby(
                    "method", sort=True
                )
            ):
                rows.append(
                    {
                        "group_type": group_type,
                        "group_name": str(
                            group_name
                        ),
                        "method": method,
                        "n_queries": int(
                            len(method_group)
                        ),
                        "mean_oracle_M2": float(
                            method_group[
                                "oracle_M2"
                            ].mean()
                        ),
                        "median_oracle_M2": float(
                            method_group[
                                "oracle_M2"
                            ].median()
                        ),
                    }
                )
    summary = pd.DataFrame(rows)
    pivot = summary.pivot_table(
        index=[
            "group_type",
            "group_name",
        ],
        columns="method",
        values="mean_oracle_M2",
    ).reset_index()

    comparisons = []
    for _, row in pivot.iterrows():
        direct = float(
            row["direct_RR"]
        )
        for baseline in [
            "pilot",
            "regularized_B1",
            "same_dictionary_B1",
            "global_descriptor",
            "unregularized_B1",
        ]:
            comparisons.append(
                {
                    "group_type": row[
                        "group_type"
                    ],
                    "group_name": row[
                        "group_name"
                    ],
                    "comparison": (
                        f"direct_RR_vs_"
                        f"{baseline}"
                    ),
                    "risk_ratio": (
                        direct
                        / float(
                            row[baseline]
                        )
                    ),
                }
            )
    return (
        summary,
        pd.DataFrame(comparisons),
    )


def run_replication(
    seed: int,
    sample_size: int,
    population: pd.DataFrame,
    x_population: np.ndarray,
    queries: dict[str, Any],
    query_library: pd.DataFrame,
    population_grams: np.ndarray,
    oracle_moments: np.ndarray,
    floor: np.ndarray,
) -> dict[str, Any]:
    started = time.time()
    rng = np.random.default_rng(
        seed + 100_000 * sample_size
    )

    n_population = len(population)
    rows = rng.integers(
        0,
        n_population,
        size=sample_size,
        endpoint=False,
    )

    population_steps = population[
        STEP_COL
    ].to_numpy(dtype=int)
    population_probabilities = (
        population[
            [
                f"p_star__{action}"
                for action in ACTION_ORDER
            ]
        ].to_numpy(dtype=float)
    )
    steps = population_steps[rows]
    probabilities = (
        population_probabilities[rows]
    )
    actions = draw_actions(
        probabilities, rng
    )
    x_sample = x_population[
        rows
    ].astype(
        np.float64,
        copy=False,
    )

    marginal_p = np.zeros(
        (N_STEPS, N_ACTIONS),
        dtype=float,
    )
    for step in range(N_STEPS):
        mask = population_steps == step
        marginal_p[step] = (
            population_probabilities[
                mask
            ].mean(axis=0)
        )

    permutation = rng.permutation(
        sample_size
    )
    midpoint = sample_size // 2
    positions_a = permutation[:midpoint]
    positions_b = permutation[midpoint:]

    artifact_a = make_sample_artifact(
        population_rows=rows[
            positions_a
        ],
        steps=steps[positions_a],
        actions=actions[
            positions_a
        ],
        x=x_sample[positions_a],
        queries=queries,
        marginal_p=marginal_p,
    )
    artifact_b = make_sample_artifact(
        population_rows=rows[
            positions_b
        ],
        steps=steps[positions_b],
        actions=actions[
            positions_b
        ],
        x=x_sample[positions_b],
        queries=queries,
        marginal_p=marginal_p,
    )
    artifact_full = make_sample_artifact(
        population_rows=rows,
        steps=steps,
        actions=actions,
        x=x_sample,
        queries=queries,
        marginal_p=marginal_p,
    )

    selected, selection_table = (
        select_models(
            queries=queries,
            artifact_a=artifact_a,
            artifact_b=artifact_b,
            raw_dimension=x_sample.shape[1],
        )
    )

    held = queries["indices"][
        "held_out"
    ]
    total_m2 = query_library[
        "pilot_correction_M2"
    ].to_numpy(dtype=float)[held]
    oracle_held = oracle_moments[
        held
    ]

    # Direct-RR final.
    direct_state = regularized_state(
        artifact_full,
        selected["direct_RR"][
            "block_ridge"
        ],
    )
    direct_white, dictionary = (
        fit_final_descriptor(
            queries,
            direct_state,
            selected["direct_RR"],
        )
    )
    direct_beta = white_to_raw(
        direct_white,
        direct_state["blocks"],
        x_sample.shape[1],
    )

    # Global descriptor final.
    global_state = regularized_state(
        artifact_full,
        selected[
            "global_descriptor"
        ]["block_ridge"],
    )
    global_white, _ = (
        fit_final_descriptor(
            queries,
            global_state,
            selected[
                "global_descriptor"
            ],
        )
    )
    global_beta = white_to_raw(
        global_white,
        global_state["blocks"],
        x_sample.shape[1],
    )

    # Target-specific regularized B1.
    b1_state = regularized_state(
        artifact_full,
        selected[
            "regularized_B1"
        ]["block_ridge"],
    )
    b1_white = (
        b1_state["targets"][held]
        * selected[
            "regularized_B1"
        ]["gamma"]
    )
    b1_beta = white_to_raw(
        b1_white,
        b1_state["blocks"],
        x_sample.shape[1],
    )

    # Same Direct-RR dictionary, target-specific coordinates,
    # and the same cross-fitted amplitude calibration.
    if dictionary.shape[1] == 0:
        same_white = np.zeros_like(
            direct_state["targets"][
                held
            ]
        )
    else:
        same_white = (
            direct_state["targets"][
                held
            ]
            @ dictionary
            @ dictionary.T
            * selected[
                "direct_RR"
            ]["gamma"]
        )
    same_beta = white_to_raw(
        same_white,
        direct_state["blocks"],
        x_sample.shape[1],
    )

    # Existing unregularized B1 for continuity with v1/v2.
    unregularized_state = (
        regularized_state(
            artifact_full, 0.0
        )
    )
    unregularized_beta = (
        white_to_raw(
            unregularized_state[
                "targets"
            ][held],
            unregularized_state[
                "blocks"
            ],
            x_sample.shape[1],
        )
    )

    zero_beta = np.zeros_like(
        direct_beta
    )
    beta_by_method = {
        "pilot": zero_beta,
        "direct_RR": direct_beta,
        "global_descriptor": global_beta,
        "regularized_B1": b1_beta,
        "same_dictionary_B1": same_beta,
        "unregularized_B1": (
            unregularized_beta
        ),
    }

    result_rows = []
    for method, beta in (
        beta_by_method.items()
    ):
        risk = oracle_risk(
            beta,
            total_m2,
            oracle_held,
            population_grams,
        )
        for local, query_index in (
            enumerate(held)
        ):
            query = query_library.iloc[
                query_index
            ]
            result_rows.append(
                {
                    "seed": seed,
                    "sample_size": (
                        sample_size
                    ),
                    "query_id": query[
                        "query_id"
                    ],
                    "support_family": (
                        query[
                            "support_family"
                        ]
                    ),
                    "method": method,
                    "oracle_M2": float(
                        risk[local]
                    ),
                    "pilot_M2": float(
                        total_m2[local]
                    ),
                    "sieve_floor": float(
                        floor[query_index]
                    ),
                }
            )

    query_results = pd.DataFrame(
        result_rows
    )
    summary, comparisons = (
        summarize_replication(
            query_results
        )
    )
    summary.insert(0, "seed", seed)
    summary.insert(
        1, "sample_size", sample_size
    )
    comparisons.insert(0, "seed", seed)
    comparisons.insert(
        1, "sample_size", sample_size
    )

    metadata = {
        "seed": seed,
        "sample_size": sample_size,
        "direct_rank": selected[
            "direct_RR"
        ]["rank"],
        "direct_block_ridge": selected[
            "direct_RR"
        ]["block_ridge"],
        "direct_descriptor_ridge": selected[
            "direct_RR"
        ]["descriptor_ridge"],
        "direct_gamma": selected[
            "direct_RR"
        ]["gamma"],
        "direct_crossfit_objective": selected[
            "direct_RR"
        ]["crossfit_objective"],
        "global_block_ridge": selected[
            "global_descriptor"
        ]["block_ridge"],
        "global_descriptor_ridge": selected[
            "global_descriptor"
        ]["descriptor_ridge"],
        "global_gamma": selected[
            "global_descriptor"
        ]["gamma"],
        "b1_block_ridge": selected[
            "regularized_B1"
        ]["block_ridge"],
        "b1_gamma": selected[
            "regularized_B1"
        ]["gamma"],
        "dictionary_rank": int(
            dictionary.shape[1]
        ),
        "direct_whitened_dimension": int(
            direct_state["dimension"]
        ),
        "direct_max_effective_condition": (
            direct_state[
                "maximum_effective_condition"
            ]
        ),
        "minimum_action_block_count": int(
            artifact_full[
                "counts"
            ].min()
        ),
        "seconds": float(
            time.time() - started
        ),
    }

    return {
        "query_results": query_results,
        "summary": summary,
        "comparisons": comparisons,
        "metadata": metadata,
        "selection_table": (
            selection_table.assign(
                seed=seed,
                sample_size=sample_size,
            )
        ),
    }


def aggregate_results(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_aggregate = (
        summaries.groupby(
            ["sample_size", "group_type", "group_name", "method"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_seed_mean_M2=("mean_oracle_M2", "mean"),
            median_seed_mean_M2=("mean_oracle_M2", "median"),
            sd_seed_mean_M2=("mean_oracle_M2", "std"),
            minimum_seed_mean_M2=("mean_oracle_M2", "min"),
            maximum_seed_mean_M2=("mean_oracle_M2", "max"),
        )
    )

    rows = []
    for keys, group in comparisons.groupby(
        ["sample_size", "group_type", "group_name", "comparison"],
        sort=True,
    ):
        ratios = group["risk_ratio"].to_numpy(dtype=float)
        if np.any(ratios <= 0) or not np.isfinite(ratios).all():
            raise RuntimeError(f"Invalid risk ratios for formal inference: {keys}")
        log_ratios = np.log(ratios)
        n = len(log_ratios)
        mean_log = float(log_ratios.mean())
        se_log = float(log_ratios.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        z = 1.959963984540054
        rows.append(
            {
                "sample_size": int(keys[0]),
                "group_type": keys[1],
                "group_name": keys[2],
                "comparison": keys[3],
                "n_seeds": int(n),
                "geometric_mean_risk_ratio": float(np.exp(mean_log)),
                "geometric_mean_ratio_CI95_lower": float(np.exp(mean_log - z * se_log)),
                "geometric_mean_ratio_CI95_upper": float(np.exp(mean_log + z * se_log)),
                "median_risk_ratio": float(np.median(ratios)),
                "minimum_risk_ratio": float(ratios.min()),
                "maximum_risk_ratio": float(ratios.max()),
                "win_count_ratio_lt_1": int(np.sum(ratios < 1.0)),
                "win_fraction_ratio_lt_1": float(np.mean(ratios < 1.0)),
            }
        )
    return method_aggregate, pd.DataFrame(rows)


def formal_decision_table(comparison_aggregate: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("overall_superiority_to_pilot", "overall", "all", "direct_RR_vs_pilot", PILOT_SUPERIORITY_MARGIN),
        ("overall_equivalence_bound_same_dictionary_B1", "overall", "all", "direct_RR_vs_same_dictionary_B1", SAME_DICTIONARY_EQUIVALENCE_MARGIN),
        ("overall_noninferiority_regularized_B1", "overall", "all", "direct_RR_vs_regularized_B1", REGULARIZED_B1_NONINFERIORITY_MARGIN),
        ("rare_support_superiority_to_pilot", "support_family", "rare", "direct_RR_vs_pilot", PILOT_SUPERIORITY_MARGIN),
    ]
    records=[]
    for sample_size in sorted(comparison_aggregate["sample_size"].unique()):
        for criterion, group_type, group_name, comparison, margin in specifications:
            row=comparison_aggregate[
                comparison_aggregate["sample_size"].eq(sample_size)
                & comparison_aggregate["group_type"].eq(group_type)
                & comparison_aggregate["group_name"].eq(group_name)
                & comparison_aggregate["comparison"].eq(comparison)
            ]
            if len(row)!=1:
                raise RuntimeError(f"Could not resolve formal row: {sample_size}, {criterion}")
            row=row.iloc[0]
            upper=float(row["geometric_mean_ratio_CI95_upper"])
            records.append({
                "sample_size": int(sample_size),
                "criterion": criterion,
                "comparison": comparison,
                "group_type": group_type,
                "group_name": group_name,
                "frozen_margin": float(margin),
                "geometric_mean_risk_ratio": float(row["geometric_mean_risk_ratio"]),
                "CI95_lower": float(row["geometric_mean_ratio_CI95_lower"]),
                "CI95_upper": upper,
                "pass": bool(upper < margin),
            })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()

    for variable in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[variable] = str(
            args.threads
        )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    print("=" * 80)
    print(
        "QAOCT RIDGE-SHRINKAGE "
        "DIRECT-RR FORMAL V1"
    )
    print("=" * 80)
    print(
        f"sample sizes: {args.sample_sizes}"
    )
    print(f"seeds:        {args.seeds}")
    print(
        "relative tol: "
        f"{FROZEN_RELATIVE_TOLERANCE:.3e}"
    )
    print(
        f"output dir:   {args.output_dir}"
    )

    required_paths = [
        args.history_bank,
        args.propensity_scores,
        args.query_library,
        args.history_basis,
        args.history_basis_row_index,
        args.population_grams,
        args.oracle_moments,
        args.selected_projection,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    history = pd.read_csv(
        args.history_bank,
        compression="infer",
        low_memory=False,
        dtype={ID_COL: str},
    )
    propensity = pd.read_csv(
        args.propensity_scores,
        compression="infer",
        low_memory=False,
        dtype={ID_COL: str},
    )
    query_library = pd.read_csv(
        args.query_library
    )
    row_index = pd.read_csv(
        args.history_basis_row_index,
        compression="infer",
        dtype={ID_COL: str},
    )
    projection = pd.read_csv(
        args.selected_projection
    )

    descriptor_columns = (
        validate_query_library(
            query_library
        )
    )
    x = np.load(
        args.history_basis,
        mmap_mode="r",
    )
    population = align_population(
        history,
        propensity,
        row_index,
        x,
    )

    gram_archive = np.load(
        args.population_grams
    )
    population_grams = (
        gram_archive["grams"].astype(
            np.float64
        )
    )
    oracle_moments = np.load(
        args.oracle_moments,
        mmap_mode="r",
    )

    projection = query_library[
        ["query_id"]
    ].merge(
        projection[
            [
                "query_id",
                "approximation_floor",
            ]
        ],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if projection[
        "approximation_floor"
    ].isna().any():
        raise ValueError(
            "Projection-floor alignment failed."
        )
    floor = projection[
        "approximation_floor"
    ].to_numpy(dtype=float)

    population_steps = population[
        STEP_COL
    ].to_numpy(dtype=int)
    tilt_features = (
        build_tilt_features(
            population
        )
    )
    queries = prepare_queries(
        query_library,
        descriptor_columns,
        population_steps,
        tilt_features,
    )

    all_query_results = []
    all_summaries = []
    all_comparisons = []
    all_metadata = []
    all_selection = []

    for sample_size in (
        args.sample_sizes
    ):
        for seed in args.seeds:
            result = run_replication(
                seed=int(seed),
                sample_size=int(
                    sample_size
                ),
                population=population,
                x_population=x,
                queries=queries,
                query_library=(
                    query_library
                ),
                population_grams=(
                    population_grams
                ),
                oracle_moments=(
                    oracle_moments
                ),
                floor=floor,
            )
            all_query_results.append(
                result[
                    "query_results"
                ]
            )
            all_summaries.append(
                result["summary"]
            )
            all_comparisons.append(
                result["comparisons"]
            )
            all_metadata.append(
                result["metadata"]
            )
            all_selection.append(
                result[
                    "selection_table"
                ]
            )

            overall = result[
                "comparisons"
            ]
            overall = overall[
                overall[
                    "group_type"
                ].eq("overall")
                & overall[
                    "group_name"
                ].eq("all")
            ]
            ratios = dict(
                zip(
                    overall[
                        "comparison"
                    ],
                    overall[
                        "risk_ratio"
                    ],
                )
            )
            meta = result["metadata"]
            print(
                f"[n={sample_size} "
                f"seed={seed}] "
                f"rank={meta['direct_rank']} "
                f"kappa={meta['direct_block_ridge']:.3g} "
                f"ridge={meta['direct_descriptor_ridge']:.3g} "
                f"gamma={meta['direct_gamma']:.4f} "
                f"direct/pilot="
                f"{ratios['direct_RR_vs_pilot']:.4f} "
                f"direct/regB1="
                f"{ratios['direct_RR_vs_regularized_B1']:.4f} "
                f"direct/same="
                f"{ratios['direct_RR_vs_same_dictionary_B1']:.4f} "
                f"sec={meta['seconds']:.2f}"
            )

    query_results = pd.concat(
        all_query_results,
        ignore_index=True,
    )
    summaries = pd.concat(
        all_summaries,
        ignore_index=True,
    )
    comparisons = pd.concat(
        all_comparisons,
        ignore_index=True,
    )
    metadata = pd.DataFrame(
        all_metadata
    )
    selection = pd.concat(
        all_selection,
        ignore_index=True,
    )

    (
        method_aggregate,
        comparison_aggregate,
    ) = aggregate_results(
        summaries,
        comparisons,
    )
    decision_table = formal_decision_table(comparison_aggregate)

    checks = {
        "all_cells_completed": bool(
            len(metadata)
            == len(args.sample_sizes)
            * len(args.seeds)
        ),
        "all_direct_ranks_in_grid": bool(
            metadata[
                "direct_rank"
            ].isin(RANK_GRID).all()
        ),
        "all_gammas_in_0_1": bool(
            metadata[
                [
                    "direct_gamma",
                    "global_gamma",
                    "b1_gamma",
                ]
            ].apply(
                lambda column: column.between(
                    0.0, 1.0
                ).all()
            ).all()
        ),
        "all_outputs_finite": bool(
            np.isfinite(
                summaries[
                    [
                        "mean_oracle_M2",
                        "median_oracle_M2",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "one_action_draw_per_replication": True,
    }
    checks["all_formal_cells_have_100_seeds"] = bool(
        comparison_aggregate["n_seeds"].eq(len(FORMAL_SEEDS)).all()
    )
    checks["formal_decisions_are_complete"] = bool(
        len(decision_table) == len(args.sample_sizes) * 4
    )
    status = (
        "RIDGE_SHRINKAGE_FORMAL_TECHNICALLY_COMPLETE"
        if all(checks.values())
        else "RIDGE_SHRINKAGE_FORMAL_REVIEW_REQUIRED"
    )

    query_results.to_csv(
        args.output_dir
        / "formal_query_risks.csv.gz",
        index=False,
        compression="gzip",
    )
    summaries.to_csv(
        args.output_dir
        / "formal_seed_method_summary.csv",
        index=False,
    )
    comparisons.to_csv(
        args.output_dir
        / "formal_seed_comparisons.csv",
        index=False,
    )
    metadata.to_csv(
        args.output_dir
        / "formal_seed_metadata.csv",
        index=False,
    )
    selection.to_csv(
        args.output_dir
        / "formal_selection_grid.csv.gz",
        index=False,
        compression="gzip",
    )
    method_aggregate.to_csv(
        args.output_dir
        / "formal_method_aggregate.csv",
        index=False,
    )
    comparison_aggregate.to_csv(
        args.output_dir
        / "formal_comparison_aggregate.csv",
        index=False,
    )
    decision_table.to_csv(
        args.output_dir
        / "formal_decision_table.csv",
        index=False,
    )

    audit = {
        "protocol": "qaoct_ridge_shrinkage_formal_v1",
        "status": status,
        "development_only": False,
        "formal_claims_from_these_seeds": True,
        "technical_checks": checks,
        "sample_sizes": [
            int(value)
            for value in args.sample_sizes
        ],
        "seeds": [
            int(value)
            for value in args.seeds
        ],
        "frozen_relative_tolerance": (
            FROZEN_RELATIVE_TOLERANCE
        ),
        "block_ridge_grid": (
            BLOCK_RIDGE_GRID
        ),
        "descriptor_ridge_grid": (
            DESCRIPTOR_RIDGE_GRID
        ),
        "rank_grid": RANK_GRID,
        "gamma_bounds": [
            GAMMA_LOWER,
            GAMMA_UPPER,
        ],
        "selection_protocol": (
            "two-fold observation cross-fitting "
            "with one action draw per replication; "
            "block ridge, descriptor ridge, rank, "
            "and pooled nonnegative amplitude gamma "
            "selected by held-out empirical Riesz "
            "objective on the 32 selection queries"
        ),
        "formal_margins": {
            "pilot_superiority": PILOT_SUPERIORITY_MARGIN,
            "same_dictionary_equivalence_upper_bound": SAME_DICTIONARY_EQUIVALENCE_MARGIN,
            "regularized_B1_noninferiority_upper_bound": REGULARIZED_B1_NONINFERIORITY_MARGIN,
        },
        "formal_decision_summary": {
            "n_criteria": int(len(decision_table)),
            "n_passed": int(decision_table["pass"].sum()),
            "all_passed": bool(decision_table["pass"].all()),
        },
        "notes": [
            "This is the frozen confirmatory representer-risk run.",
            "Rank zero remains the pilot fallback.",
            "Direct-RR versus global descriptor is descriptive, not confirmatory.",
            "The benchmark treats landmark rows as i.i.d. sampling units.",
        ],
    }
    (
        args.output_dir
        / "ridge_shrinkage_formal_v1_audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("-" * 80)
    print("SEED METADATA")
    print(
        metadata[
            [
                "sample_size",
                "seed",
                "direct_rank",
                "direct_block_ridge",
                "direct_descriptor_ridge",
                "direct_gamma",
                "global_gamma",
                "b1_gamma",
                "minimum_action_block_count",
                "seconds",
            ]
        ].to_string(index=False)
    )
    print("-" * 80)
    print("COMPARISON AGGREGATE")
    print(
        comparison_aggregate.to_string(
            index=False
        )
    )
    print("-" * 80)
    print("FORMAL DECISION TABLE")
    print(decision_table.to_string(index=False))
    print("-" * 80)
    print(f"STATUS: {status}")

    if status != "RIDGE_SHRINKAGE_FORMAL_TECHNICALLY_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
