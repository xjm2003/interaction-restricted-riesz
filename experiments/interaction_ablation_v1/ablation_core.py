"""Matched 2 x 2 scores using the unchanged archived projection engine.

This module fits no nuisance models and receives no population truth. Query
multipliers already contain their frozen normalizations and scientific time
weights. Arm order is (00, 10, 01, 11). No implicit clipping or smoothing occurs.
"""
from __future__ import annotations

import numpy as np


PROCEDURES = (
    "raw_full", "raw_restricted", "projected_full",
    "projected_restricted", "raw_plugin",
)
PAIRS = {
    "correction_projected": ("projected_restricted", "projected_full"),
    "correction_raw": ("raw_restricted", "raw_full"),
    "projection_full": ("projected_full", "raw_full"),
    "projection_restricted": ("projected_restricted", "raw_restricted"),
    "overall": ("projected_restricted", "raw_full"),
}


def _finite(value, ndim, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected a nonempty finite {ndim}-D array.")
    return array


def _n_steps(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("n_steps must be a positive integer.")
    return int(value)


def _indices(value, length, upper, name):
    array = _finite(value, 1, name)
    if array.shape != (length,) or np.any(array < 0) or np.any(array >= upper):
        raise ValueError(f"{name}: misaligned or outside the frozen index range.")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name}: indices must be integer-valued.")
    return array.astype(np.int64)


def _probabilities(value, name):
    array = _finite(value, 2, name)
    if array.shape[1] != 4 or np.any(array <= 0) or not np.allclose(
        array.sum(axis=1), 1.0, atol=1e-9, rtol=0.0
    ):
        raise ValueError(f"{name}: four positive probabilities summing to one are required.")
    return array


def _maximum(value):
    result = float(np.max(np.abs(value)))
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite diagnostic.")
    return result


def matched_scores(engine, p_train, p_test, mu_train, mu_test,
                   scales_train, scales_test, steps_train, steps_test,
                   actions, y, n_steps):
    """Return all five evaluation scores and patient-free projection diagnostics.

    Projection coordinates are fitted only from outer-training predictions and
    query multipliers, separately at each landmark. ``y`` and ``actions`` are
    evaluation observations; neither is passed into the projection fits.
    """
    n_steps = _n_steps(n_steps)
    p_train = _probabilities(p_train, "p_train")
    p_test = _probabilities(p_test, "p_test")
    mu_train = _finite(mu_train, 2, "mu_train")
    mu_test = _finite(mu_test, 2, "mu_test")
    scales_train = _finite(scales_train, 3, "scales_train")
    scales_test = _finite(scales_test, 3, "scales_test")
    n_train, n_test = len(p_train), len(p_test)
    if mu_train.shape != p_train.shape or mu_test.shape != p_test.shape:
        raise ValueError("Outcome predictions and probabilities must have the same (rows, 4) shape.")
    if scales_train.shape[0] != n_train or scales_test.shape[0] != n_test:
        raise ValueError("Query multipliers are not row-aligned.")
    if scales_train.shape[2] != 4 or scales_test.shape[2] != 4:
        raise ValueError("Query multipliers must use the four frozen treatment arms.")
    if scales_train.shape[1] != scales_test.shape[1]:
        raise ValueError("Training and evaluation query counts differ.")
    steps_train = _indices(steps_train, n_train, n_steps, "steps_train")
    steps_test = _indices(steps_test, n_test, n_steps, "steps_test")
    actions = _indices(actions, n_test, 4, "actions")
    y = _finite(y, 1, "y")
    if y.shape != (n_test,):
        raise ValueError("Evaluation outcomes are not row-aligned.")
    counts = np.bincount(steps_train, minlength=n_steps)
    if np.any(counts < 2):
        raise ValueError("Each landmark needs at least two outer-training rows.")

    phi_train = np.ones((n_train, 1), dtype=float)
    phi_test = np.ones((n_test, 1), dtype=float)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        projector = engine.WeightedProjector.fit(
            phi_train, steps_train, p_train, "CONST", n_steps
        )
        tau_train = mu_train @ engine.K
        ell_train = engine.query_ell(scales_train, p_train)
        outcome_coordinates = projector.fit_coordinates(tau_train)
        query_coordinates = projector.fit_coordinates(ell_train)
        delta_test = projector.predict(outcome_coordinates, phi_test, steps_test)[:, 0]
        gamma_test = projector.predict(query_coordinates, phi_test, steps_test)
        mu_projected = engine.project_outcome(mu_test, p_test, delta_test)
        alpha_full = scales_test / p_test[:, None, :]
        alpha_restricted = engine.project_representer(scales_test, p_test, gamma_test)
        scores = {
            "raw_full": engine.orthogonal_scores(
                scales_test, alpha_full, mu_test, actions, y),
            "raw_restricted": engine.orthogonal_scores(
                scales_test, alpha_restricted, mu_test, actions, y),
            "projected_full": engine.orthogonal_scores(
                scales_test, alpha_full, mu_projected, actions, y),
            "projected_restricted": engine.orthogonal_scores(
                scales_test, alpha_restricted, mu_projected, actions, y),
            "raw_plugin": np.sum(scales_test * mu_test[:, None, :], axis=2),
        }
        for name, value in scores.items():
            if value.shape != scales_test.shape[:2] or not np.isfinite(value).all():
                raise FloatingPointError(f"Invalid evaluation scores for {name}.")
        row = np.arange(n_test)[:, None]
        query = np.arange(scales_test.shape[1])[None, :]
        selected_action = actions[:, None]
        observed_difference = (alpha_full - alpha_restricted)[row, query, selected_action]
        projected_residual = y - mu_projected[np.arange(n_test), actions]
        algebraic_difference = observed_difference * projected_residual[:, None]
        diagnostics = {
            "structure": "CONST",
            "arm_order": ["00", "10", "01", "11"],
            "n_steps": n_steps,
            "training_rows": n_train,
            "evaluation_rows": n_test,
            "queries": int(scales_test.shape[1]),
            "training_only_projection": True,
            "per_landmark_projection": projector.diagnostics,
            "outcome_training_projection": projector.training_residual_summary(
                outcome_coordinates, tau_train),
            "representer_training_projection": projector.training_residual_summary(
                query_coordinates, ell_train),
            "max_outcome_restriction_residual": _maximum(mu_projected @ engine.K - delta_test),
            "max_representer_restriction_residual": _maximum(
                np.sum(alpha_restricted * engine.K[None, None, :], axis=2) - gamma_test),
            "max_paired_difference_identity_residual": _maximum(
                scores["projected_full"] - scores["projected_restricted"] - algebraic_difference),
            "max_abs_outcome_projection_change": _maximum(mu_projected - mu_test),
            "max_abs_representer_projection_change": _maximum(alpha_restricted - alpha_full),
            "max_abs_full_representer": _maximum(alpha_full),
            "max_abs_restricted_representer": _maximum(alpha_restricted),
            "evaluation_min_probability": float(p_test.min()),
        }
    return scores, diagnostics


def _profile_layout(clusters, steps, n_rows, n_steps):
    labels = np.asarray(clusters)
    if labels.shape != (n_rows,):
        raise ValueError("Cluster IDs must be a one-dimensional row-aligned array.")
    if labels.dtype.kind in "biufc" and not np.isfinite(labels).all():
        raise ValueError("Cluster IDs must be finite.")
    if labels.dtype.kind == "O":
        for value in labels:
            if value is None or (isinstance(value, (float, complex, np.number)) and not np.isfinite(value)):
                raise ValueError("Cluster IDs cannot be missing or nonfinite.")
    try:
        unique, inverse = np.unique(labels, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cluster IDs must have a consistent sortable type.") from exc
    steps = _indices(steps, n_rows, n_steps, "steps")
    layout = np.zeros((len(unique), n_steps), dtype=np.int64)
    np.add.at(layout, (inverse, steps), 1)
    if not np.all(layout == 1):
        raise ValueError("Each sampled profile must contain exactly one row at every frozen landmark.")
    return labels


def summarize_scores(engine, scores, clusters, steps, n_steps):
    """Save only independent-unit moments, including paired differences.

    A PHQ-9 unit is the mean of its four landmark scores. Squaring occurs only
    AFTER this averaging. Fold summaries can be added before computing the
    global estimate/SE; equal fold sizes are not required. Pair signs follow
    ``PAIRS`` (first procedure minus second).
    """
    n_steps = _n_steps(n_steps)
    if set(scores) != set(PROCEDURES):
        raise ValueError("All five matched procedures are required, with no unknown procedure names.")
    arrays = {name: _finite(scores[name], 2, name) for name in PROCEDURES}
    first = arrays[PROCEDURES[0]]
    if any(array.shape != first.shape for array in arrays.values()):
        raise ValueError("All procedures must use identical rows and queries.")
    clusters = _profile_layout(clusters, steps, len(first), n_steps)

    def moments(array):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            total, squares, units = engine.profile_summaries(array, clusters)
        if not np.isfinite(total).all() or not np.isfinite(squares).all():
            raise FloatingPointError("Nonfinite independent-unit score moments.")
        return {"sum_score": total.tolist(), "sumsq_score": squares.tolist(), "units": int(units)}

    return {
        "stats": {name: moments(arrays[name]) for name in PROCEDURES},
        "paired_stats": {name: moments(arrays[left] - arrays[right])
                         for name, (left, right) in PAIRS.items()},
    }
