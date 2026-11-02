"""CVD input adapter preserving the frozen single-index semi-synthetic DGP.

Only baseline covariates are model inputs. The original reader validates source
identifiers and dates, then discards them. Its optional observed-action audit is
not retained here. Simulator truth stays in memory for action generation and
post-freeze risk evaluation; the old protocol's oracle projections are discarded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np

from frozen import cvd_backend, cvd_data, cvd_protocol


DEFAULT_INPUT = None
INPUT_ENV_VAR = "CVD_DATA_PATH"
DEFAULT_QUERY_SEED = 2026091651
VERSION = "cvd-semi-adapter-v1.0"


def configure_core():
    """Configure the unmodified backend exactly as the original CVD grid."""
    core = cvd_backend
    core.N_STEPS = 1
    core.N_ACTIONS = 4
    core.P_LANDMARK = 1.0
    core.ACTION_ORDER = list(cvd_protocol.ACTION_ORDER)
    core.BLOCK_RIDGE_GRID = [0.0, 0.001, 0.01, 0.1, 1.0]
    core.DESCRIPTOR_RIDGE_GRID = [0.01, 1.0, 10.0]
    core.RANK_GRID = [0, 1, 2, 4, 8, 12, 16, 24]
    return core


def _build(frame, *, query_seed, source_metadata):
    x, zdict, basis_report = cvd_data.build_basis(frame)
    # Keep the original simulator unchanged. It also builds historical oracle
    # diagnostics: none enter selection, the adapter output or its metadata.
    population, queries, library, grams, moments, floor, protocol = (
        cvd_protocol.build_protocol(x, zdict, query_seed=query_seed)
    )
    del grams, moments, floor
    library = library.drop(columns=["pilot_correction_M2", "approximation_floor"])
    p = population[["p_star__" + a for a in cvd_protocol.ACTION_ORDER]].to_numpy()
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.0):
        raise ValueError("CVD simulator produced invalid action probabilities.")
    metadata: dict[str, Any] = {
        **source_metadata,
        "adapter_version": VERSION,
        "dataset": "cvd",
        "population_rows": int(len(population)),
        "population_patients": int(len(population)),
        "basis_dimension": int(x.shape[1]),
        "basis_feature_names": list(basis_report["feature_names"]),
        "basis_dropped_feature_names": list(basis_report["dropped_features"]),
        "query_seed": int(query_seed),
        "query_split_counts": {
            name: int(len(queries["indices"][name]))
            for name in ("train", "selection", "held_out")
        },
        "query_split_unit": "entire tilt group; all eight contrasts remain together",
        "disjoint_tilt_groups": True,
        "n_steps": 1,
        "action_order": list(cvd_protocol.ACTION_ORDER),
        "baseline_grid": {
            "block_ridge": [0.0, 0.001, 0.01, 0.1, 1.0],
            "descriptor_ridge": [0.01, 1.0, 10.0],
            "rank": [0, 1, 2, 4, 8, 12, 16, 24],
            "relative_eigenvalue_cutoff": float(cvd_backend.FROZEN_RELATIVE_TOLERANCE),
            "minimum_block_rank": int(cvd_backend.MIN_BLOCK_RANK),
            "gamma_bounds": [float(cvd_backend.GAMMA_LOWER), float(cvd_backend.GAMMA_UPPER)],
            "p_landmark": 1.0,
        },
        "basis_fit_scope": "entire fixed empirical covariate population",
        "basis_preprocessing": (
            "original median imputation, 0.5/99.5 percentile clipping, population "
            "scaling, numeric squares and pairwise interactions, categorical indicators"
        ),
        "protocol": "original cvd_single_index_semisynthetic_v1",
        "simulator_marginal_targets": cvd_protocol.TARGET_MARGINALS.tolist(),
        "simulator_uniform_mixing": float(cvd_protocol.UNIFORM_MIXING),
        "simulator_intercepts_exported": False,
        "basis_preprocessing_coefficients_exported": False,
        "real_outcomes_used": False,
        "real_actions_used_in_fitting_or_DGP": False,
        "original_loader_action_audit_discarded": True,
        "patient_identifiers_or_dates_retained": False,
        "patient_records_exported": False,
        "oracle_projections_available_to_fitting": False,
        "baseline_timing": "candidate baseline fields; confirm measurement timing in the source data dictionary",
        "target": "finite-population held-query representer MSE; not clinical effect or new-patient prediction",
    }
    del protocol, basis_report
    return {
        "core": configure_core(),
        "population": population,
        "x_population": x,
        "queries": queries,
        "query_library": library,
        "input_metadata": metadata,
    }


def resolve_input_path(input_path=None):
    """Use an explicit input first, otherwise the caller's CVD data location."""
    value = input_path if input_path is not None else os.environ.get(INPUT_ENV_VAR)
    if value is None or not str(value).strip():
        raise ValueError("CVD input is required: pass --input PATH or set CVD_DATA_PATH.")
    return Path(value).expanduser()


def load_real(input_path=None, member=None, query_seed=DEFAULT_QUERY_SEED):
    """Load the full validated cohort from CSV, gzip, ZIP or extracted directory."""
    frame, report = cvd_data.load_cvd(resolve_input_path(input_path), member=member)
    # Avoid persisting arbitrary source headers, action audit, dates or numeric
    # preprocessing coefficients. Existing input-quality counts are aggregate.
    metadata = {
        "data_kind": "real_baseline_covariates_with_simulated_actions",
        "mock": False,
        "source_file_name": Path(report["input_path"]).name,
        "source_zip_member": report["selected_zip_member"],
        "full_input_rows": int(report["full_input_rows"]),
        "single_index_per_patient": bool(report["single_index_per_patient"]),
        "source_subsampling": "none",
        "retained_baseline_columns": list(report["retained_baseline_columns"]),
        "absent_optional_columns": list(report["absent_optional_columns"]),
        "missing_counts": report["missing_n_selected_population"],
        "numeric_quality_counts": report["numeric_quality_full_input"],
        "categorical_quality_counts": report["categorical_quality_full_input"],
    }
    return _build(frame, query_seed=query_seed, source_metadata=metadata)


def load_mock(patients=1000, seed=2026091701, query_seed=DEFAULT_QUERY_SEED):
    """Fictional CVD covariates for software checks, with the original CVD DGP."""
    fictional = cvd_data.make_mock_population(n=patients, seed=seed)
    frame = fictional[list(cvd_data.BASELINE_COLUMNS)].copy()
    return _build(frame, query_seed=query_seed, source_metadata={
        "data_kind": "fictional_covariates_for_software_validation",
        "mock": True,
        "population_seed": int(seed),
        "source_subsampling": "none",
        "source_file_name": None,
        "source_zip_member": None,
        "retained_baseline_columns": list(frame.columns),
        "full_input_rows": int(len(frame)),
        "single_index_per_patient": True,
        "clinical_source_read": False,
    })
