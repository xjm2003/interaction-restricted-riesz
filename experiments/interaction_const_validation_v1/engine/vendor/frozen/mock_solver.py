"""Observed-data convex inverse-probability and multinomial estimators.

No simulator/oracle quantities are accepted here. The marginal probabilities
are the benchmark's already-known pilot. A fitted object stores coefficients
and convergence metadata, never the source observations.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp


RIDGES = (0.001, 0.01, 0.1, 1.0)
STATIONARITY_TOLERANCE = 5e-7
NEWTON_TOLERANCE = 1e-9


def _inputs(x, steps):
    x = np.asarray(x, dtype=float)
    steps = np.asarray(steps)
    if x.ndim != 2 or x.shape[1] != 12 or len(x) != len(steps):
        raise ValueError("Expected x=(n,12), with intercept followed by 11 features.")
    if not np.isfinite(x).all() or not np.allclose(x[:, 0], 1.0):
        raise ValueError("x must be finite and its first column must be one.")
    if steps.ndim != 1 or not np.isin(steps, np.arange(4)).all():
        raise ValueError("steps must take values 0,1,2,3.")
    return x, steps.astype(int)


def _standardize_fit(a):
    mean = np.mean(a, axis=0)
    scale = np.std(a, axis=0)
    # Constant columns are retained as exact zeros after centering.
    scale = np.where(scale > 0, scale, 1.0)
    return mean, scale


def fit_feature_map(x, steps, degree):
    """Fit normalization on this observation fit fold only.

    Degree 1: intercept, standardized 11 x features and three time indicators.
    Degree 2 adds all 66 upper-triangular products of standardized x features,
    each centered and scaled using this same fold. There are no time products.
    """
    x, steps = _inputs(x, steps)
    if degree not in (1, 2) or not len(x):
        raise ValueError("degree must be 1 or 2, with nonempty fitting data.")
    xm, xs = _standardize_fit(x[:, 1:])
    tm, ts = _standardize_fit((steps[:, None] == np.arange(1, 4)).astype(float))
    geom = {"degree": int(degree), "x_mean": xm, "x_scale": xs,
            "time_mean": tm, "time_scale": ts, "input_dimension": 12}
    if degree == 2:
        ii, jj = np.triu_indices(11)
        z = (x[:, 1:] - xm) / xs
        qm, qs = _standardize_fit(z[:, ii] * z[:, jj])
        geom.update(quadratic_i=ii, quadratic_j=jj, quadratic_mean=qm,
                    quadratic_scale=qs)
    return geom


def transform_features(x, steps, geometry):
    x, steps = _inputs(x, steps)
    z = (x[:, 1:] - geometry["x_mean"]) / geometry["x_scale"]
    tt = ((steps[:, None] == np.arange(1, 4)).astype(float)
          - geometry["time_mean"]) / geometry["time_scale"]
    columns = [np.ones((len(x), 1)), z, tt]
    if geometry["degree"] == 2:
        products = z[:, geometry["quadratic_i"]] * z[:, geometry["quadratic_j"]]
        columns.append((products - geometry["quadratic_mean"])
                       / geometry["quadratic_scale"])
    elif geometry["degree"] != 1:
        raise ValueError("Invalid stored feature degree.")
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite transformed features.")
    return result


def _design(Phi, actions=None):
    Phi = np.asarray(Phi, dtype=float)
    if (Phi.ndim != 2 or not len(Phi) or not np.isfinite(Phi).all()
            or not np.allclose(Phi[:, 0], 1.0)):
        raise ValueError("Finite, nonempty design with leading intercept required.")
    if actions is not None:
        actions = np.asarray(actions)
        if actions.shape != (len(Phi),) or not np.isin(actions, np.arange(4)).all():
            raise ValueError("actions must be an n vector of 0,1,2,3.")
        actions = actions.astype(int)
    return Phi, actions


def _pbar(pbar_byrow, n, action=None):
    p = np.asarray(pbar_byrow, dtype=float)
    if p.shape == (4,):
        p = np.broadcast_to(p, (n, 4))
    if action is not None and p.shape == (n,):
        if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
            raise ValueError("All marginal pilot probabilities must lie in (0,1).")
        return p
    if (p.shape != (n, 4) or not np.isfinite(p).all()
            or np.any((p <= 0) | (p >= 1))
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10)):
        raise ValueError("pbar_byrow must have shape (n,4) and valid positive rows.")
    return p if action is None else p[:, action]


def _calibration_values(beta, Phi, active_weight, absent_weight, offset, ridge,
                        hessian=False):
    """Exact normalized objective/gradient; overflow trials are rejected."""
    f = offset + Phi @ beta
    active = active_weight > 0
    with np.errstate(over="ignore", invalid="ignore"):
        # The exponential is needed only for observed action-a records.
        e = np.exp(f[active])
        ae = active_weight[active] * e
        loss = float(np.sum(ae) - absent_weight @ f
                     + 0.5 * ridge * (beta[1:] @ beta[1:]))
    if not np.isfinite(loss) or not np.isfinite(ae).all():
        return (np.inf, None, None) if hessian else (np.inf, None)
    residual = -absent_weight.copy()
    residual[active] += ae
    grad = Phi.T @ residual
    grad[1:] += ridge * beta[1:]
    if not np.isfinite(grad).all():
        return (np.inf, None, None) if hessian else (np.inf, None)
    if not hessian:
        return loss, grad
    H = Phi[active].T @ (ae[:, None] * Phi[active])
    H.flat[::len(beta) + 1] += np.r_[0.0, np.repeat(ridge, len(beta) - 1)]
    return loss, grad, H


def calibration_objective(beta, Phi, actions, weights, pbar_byrow, action, ridge,
                          hessian=False):
    """Public analytic objective for independent mathematical verification."""
    Phi, actions = _design(Phi, actions)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(Phi),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite nonnegative n vector.")
    if not weights.sum() > 0:
        raise ValueError("Positive total query weight is required.")
    p = _pbar(pbar_byrow, len(Phi), action)
    wn = weights / weights.sum()
    return _calibration_values(np.asarray(beta, dtype=float), Phi,
                               wn * (actions == action), wn * (actions != action),
                               np.log1p(-p) - np.log(p), ridge, hessian)


def _newton(beta, values, max_iter=100):
    """Damped Newton with Armijo rejection of nonfinite exact-objective trials."""
    beta = np.asarray(beta, dtype=float).copy()
    objective_path = []
    backtracks = 0
    for iteration in range(max_iter + 1):
        loss, grad, H = values(beta, True)
        if not np.isfinite(loss) or grad is None or not np.isfinite(H).all():
            raise FloatingPointError("Nonfinite current Newton objective or Hessian.")
        objective_path.append(float(loss))
        gnorm = float(np.max(np.abs(grad)))
        if gnorm <= NEWTON_TOLERANCE:
            break
        if iteration == max_iter:
            break
        direction = solve(H, -grad, assume_a="pos", check_finite=False)
        directional = float(grad @ direction)
        if not np.isfinite(directional) or directional >= 0:
            raise RuntimeError("Newton direction does not decrease the exact objective.")
        rate = 1.0
        accepted = False
        for _ in range(80):
            trial = beta + rate * direction
            trial_loss, trial_grad = values(trial, False)
            # The second clause avoids roundoff stalling at an already tiny
            # Newton decrement; it still demands a strictly smaller gradient.
            armijo = trial_loss <= loss + 1e-4 * rate * directional
            roundoff = (trial_loss <= loss + 2e-14 * (1 + abs(loss))
                        and trial_grad is not None
                        and np.max(np.abs(trial_grad)) < gnorm
                        and -directional * rate < 1e-12 * (1 + abs(loss)))
            if np.isfinite(trial_loss) and (armijo or roundoff):
                beta = trial
                accepted = True
                break
            rate *= 0.5
            backtracks += 1
        if not accepted:
            raise RuntimeError("Exact-objective Newton line search failed.")
    loss, grad, _ = values(beta, True)
    certificate = {"objective": float(loss), "gradient_inf": float(np.max(np.abs(grad))),
                   "iterations": int(iteration), "backtracks": int(backtracks),
                   "objective_initial": objective_path[0],
                   "maximum_objective_increase": float(max([0.0] + list(np.diff(objective_path)))),
                   "stationarity_tolerance": STATIONARITY_TOLERANCE,
                   "converged": bool(np.max(np.abs(grad)) <= STATIONARITY_TOLERANCE),
                   "solver": "damped_newton_exact_objective"}
    if not certificate["converged"]:
        raise RuntimeError(f"Stationarity check failed: {certificate}")
    return beta, certificate


def fit_calibrated(Phi, actions, weights, pbar_byrow, action, ridge):
    Phi, actions = _design(Phi, actions)
    if action not in range(4) or ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("Valid action and positive finite ridge required.")
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (len(Phi),) or not np.isfinite(weights).all()
            or np.any(weights < 0)):
        raise ValueError("Finite nonnegative weights required.")
    p = _pbar(pbar_byrow, len(Phi), action)
    if weights.sum() == 0:
        # This term is absent from the query risk. The normalized objective
        # would be 0/0, so do not optimize it or claim identified coefficients.
        # We retain the pilot by convention. 'converged' below only records
        # that no numerical optimization is pending for this inactive term.
        cert = {"inactive": True, "fit_status": "inactive_no_identification",
                "objective": 0.0, "gradient_inf": 0.0, "iterations": 0,
                "backtracks": 0, "objective_initial": 0.0,
                "maximum_objective_increase": 0.0,
                "stationarity_tolerance": STATIONARITY_TOLERANCE,
                "converged": True, "unique_solution_claim": False,
                "solver": "no_fit_pilot_convention", "dimension": int(Phi.shape[1]),
                "weighted_action_fraction": 0.0, "n_positive_weight": 0}
        return {"kind": "calibrated_inverse", "beta": np.zeros(Phi.shape[1]),
                "action": int(action), "ridge": float(ridge), "certificate": cert}
    wn = weights / weights.sum()
    aw, bw = wn * (actions == action), wn * (actions != action)
    if not aw.sum() > 0 or not bw.sum() > 0:
        raise ValueError("Both weighted action and complement support are required.")
    offset = np.log1p(-p) - np.log(p)
    def values(beta, hessian):
        return _calibration_values(beta, Phi, aw, bw, offset, ridge, hessian)
    beta, cert = _newton(np.zeros(Phi.shape[1]), values)
    cert.update(weighted_action_fraction=float(aw.sum()),
                n_positive_weight=int(np.sum(weights > 0)),
                dimension=int(Phi.shape[1]), inactive=False,
                fit_status="stationary_observed_objective")
    return {"kind": "calibrated_inverse", "beta": beta, "action": int(action),
            "ridge": float(ridge), "certificate": cert}


def predict_inverse(model, Phi, pbar_byrow):
    Phi, _ = _design(Phi)
    p = _pbar(pbar_byrow, len(Phi), model["action"])
    f = np.log1p(-p) - np.log(p) + Phi @ model["beta"]
    with np.errstate(over="ignore"):
        w = 1.0 + np.exp(f)
    if not np.isfinite(w).all():
        raise FloatingPointError("Inverse prediction overflow; no clipping applied.")
    return w


def _multinomial_values(beta_flat, Phi, actions, offset, ridge, hessian=False):
    beta = np.asarray(beta_flat).reshape(Phi.shape[1], 3)
    logits = np.column_stack([np.zeros(len(Phi)), offset + Phi @ beta])
    log_norm = logsumexp(logits, axis=1)
    probabilities = np.exp(logits - log_norm[:, None])
    loss = float(np.mean(log_norm - logits[np.arange(len(Phi)), actions])
                 + 0.5 * ridge * np.sum(beta[1:] ** 2))
    residual = probabilities[:, 1:].copy()
    nonbase = actions > 0
    residual[np.flatnonzero(nonbase), actions[nonbase] - 1] -= 1
    grad = Phi.T @ residual / len(Phi)
    grad[1:] += ridge * beta[1:]
    if not hessian:
        return loss, grad.ravel()
    d = Phi.shape[1]
    H = np.empty((d, 3, d, 3))
    for a in range(3):
        for b in range(a, 3):
            wa = probabilities[:, a + 1] * ((a == b) - probabilities[:, b + 1])
            block = Phi.T @ (wa[:, None] * Phi) / len(Phi)
            H[:, a, :, b] = block
            H[:, b, :, a] = block.T
    H = H.reshape(3 * d, 3 * d)
    H.flat[::3 * d + 1] += np.repeat(np.r_[0.0, np.full(d - 1, ridge)], 3)
    return loss, grad.ravel(), H


def multinomial_objective(beta_flat, Phi, actions, pbar_byrow, ridge, hessian=False):
    """Public reference-class softmax cross-entropy objective and derivatives."""
    Phi, actions = _design(Phi, actions)
    p = _pbar(pbar_byrow, len(Phi))
    offset = np.log(p[:, 1:]) - np.log(p[:, [0]])
    return _multinomial_values(beta_flat, Phi, actions, offset, ridge, hessian)


def fit_multinomial(Phi, actions, pbar_byrow, ridge):
    Phi, actions = _design(Phi, actions)
    if ridge <= 0 or not np.isfinite(ridge) or len(np.unique(actions)) != 4:
        raise ValueError("Positive ridge and support for all four actions required.")
    p = _pbar(pbar_byrow, len(Phi))
    offset = np.log(p[:, 1:]) - np.log(p[:, [0]])
    def values(beta, hessian=False):
        return _multinomial_values(beta, Phi, actions, offset, ridge, hessian)
    result = minimize(values, np.zeros(Phi.shape[1] * 3), method="L-BFGS-B", jac=True,
                      options={"maxiter": 1000, "maxls": 60, "gtol": 1e-9,
                               "ftol": 1e-14, "maxcor": 20})
    loss, grad = values(result.x)
    cert = {"objective": float(loss), "gradient_inf": float(np.max(np.abs(grad))),
            "iterations": int(result.nit), "solver": "L-BFGS-B_exact_objective",
            "optimizer_message": str(result.message), "optimizer_success": bool(result.success),
            "stationarity_tolerance": STATIONARITY_TOLERANCE,
            "converged": bool(np.max(np.abs(grad)) <= STATIONARITY_TOLERANCE),
            "dimension": int(Phi.shape[1])}
    coefficients = result.x
    if not cert["converged"]:
        coefficients, refined = _newton(coefficients, values, max_iter=30)
        refined["initial_lbfgs_iterations"] = int(result.nit)
        refined["dimension"] = int(Phi.shape[1])
        refined["solver"] = "L-BFGS-B_then_damped_Newton_same_objective"
        cert = refined
    return {"kind": "multinomial_ce", "beta": coefficients.reshape(Phi.shape[1], 3),
            "ridge": float(ridge), "certificate": cert}


def predict_probabilities(model, Phi, pbar_byrow, floor=0.0):
    """Softmax followed by q_a=floor+(1-4*floor)*p_a, without fit-time changes."""
    Phi, _ = _design(Phi)
    p = _pbar(pbar_byrow, len(Phi))
    if not 0 <= floor < 0.25:
        raise ValueError("floor must lie in [0,0.25).")
    logits = np.column_stack([np.zeros(len(Phi)),
                             np.log(p[:, 1:]) - np.log(p[:, [0]]) + Phi @ model["beta"]])
    probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, None])
    if floor:
        probabilities = floor + (1.0 - 4.0 * floor) * probabilities
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise FloatingPointError("Nonfinite or unnormalized probabilities.")
    return probabilities
