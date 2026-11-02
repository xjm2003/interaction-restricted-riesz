"""Minimal cell-space interaction projection; no propensity/outcome truth is used.

Action order: [00,10,01,11]. Main response functions stay unrestricted.
The only fitted new matrices have dimension <= 4, separately per landmark.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

K = np.array([1., -1., -1., 1.])
STRUCTURES = ('CONST', 'LOWDIM', 'NO_INT')
RCOND = 1e-10  # fixed support convention, not a selected tuning parameter


def _array(x, ndim, name):
    x = np.asarray(x, dtype=float)
    if x.ndim != ndim or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError(f'{name}: expected nonempty finite {ndim}-D array.')
    return x


def check_probabilities(p):
    p = _array(p, 2, 'probabilities')
    if p.shape[1] != 4 or np.any(p <= 0) or not np.allclose(p.sum(1), 1., atol=1e-9, rtol=0):
        raise ValueError('Expected four strictly positive probabilities summing to one; no implicit clipping.')
    return p


def check_steps(steps, n, n_steps):
    raw = np.asarray(steps)
    if raw.shape != (n,) or not np.isfinite(raw).all() or not np.all(raw == raw.astype(int)):
        raise ValueError('Invalid landmark indices.')
    st = raw.astype(int)
    if np.any((st < 0) | (st >= n_steps)):
        raise ValueError('Landmark index outside the frozen design.')
    return st


def interaction_features(phi, structure):
    phi = _array(phi, 2, 'modifier features')
    if not np.allclose(phi[:, 0], 1., atol=1e-12, rtol=0):
        raise ValueError('Modifier features must begin with an intercept.')
    if structure == 'CONST':
        return np.ones((len(phi), 1))
    if structure == 'LOWDIM':
        return phi
    if structure == 'NO_INT':
        return np.zeros((len(phi), 0))
    raise ValueError('Unknown frozen interaction structure: ' + str(structure))


def weight_j(p):
    p = check_probabilities(p)
    return 1. / np.sum(1. / p, axis=1)


def query_ell(scales, p):
    p = check_probabilities(p)
    s = _array(scales, 3, 'query multipliers')
    if s.shape[0] != len(p) or s.shape[2] != 4:
        raise ValueError('Misaligned query multipliers.')
    return np.sum(s * (K[None, :] / p)[:, None, :], axis=2)


@dataclass
class WeightedProjector:
    """Train-X-only j-weighted projection; coefficients differ by landmark.

    Training predictions are outer-training refit predictions, NOT in-training-row
    OOF predictions. Outer held rows never enter this fit.
    """
    features: np.ndarray
    steps: np.ndarray
    weights: np.ndarray
    structure: str
    n_steps: int
    inverses: dict
    diagnostics: list
    input_dim: int

    @classmethod
    def fit(cls, phi, steps, probabilities, structure, n_steps, rcond=RCOND):
        phi = _array(phi, 2, 'modifier features')
        p = check_probabilities(probabilities)
        if len(phi) != len(p) or n_steps < 1 or not (0 < rcond < 1):
            raise ValueError('Invalid projection dimensions/cutoff.')
        st = check_steps(steps, len(phi), n_steps)
        f = interaction_features(phi, structure)
        w = weight_j(p)
        inverses, info = {}, []
        for t in range(n_steps):
            take = st == t
            if take.sum() < 2:
                raise ValueError(f'Landmark {t} has insufficient outer-training observations.')
            if f.shape[1] == 0:
                inv = np.zeros((0, 0))
                rank, cond, orth = 0, 1., 0.
            else:
                z = f[take]
                gram = (z.T * w[take]) @ z / take.sum()
                gram = (gram + gram.T) / 2
                eig, vec = np.linalg.eigh(gram)
                if eig[-1] <= 0 or eig[0] < -1e-9 * max(1., eig[-1]):
                    raise ArithmeticError('Invalid weighted projection Gram.')
                keep = eig > rcond * eig[-1]
                inv = (vec[:, keep] / eig[keep]) @ vec[:, keep].T
                rank = int(keep.sum())
                cond = float(eig[-1] / eig[keep][0])
                orth = float(np.max(np.abs(gram @ inv @ gram - gram)))
            inverses[t] = inv
            info.append(dict(landmark=t, structure=structure, train_rows=int(take.sum()),
                dimension=f.shape[1], retained_rank=rank, retained_condition=cond,
                gram_generalized_inverse_residual=orth, fixed_rcond=float(rcond),
                propensity_min=float(p[take].min()), no_time_pooling=True))
        return cls(f.copy(), st.copy(), w.copy(), structure, int(n_steps), inverses,
                   info, phi.shape[1])

    def fit_coordinates(self, targets):
        """Project scalar/vector TRAINING predictions or X-only query functions.

        No original Y, true conditional means, true propensities, or true effects
        are arguments. The caller freezes both kinds of coordinates before test.
        """
        z = np.asarray(targets, float)
        if z.ndim == 1:
            z = z[:, None]
        z = _array(z, 2, 'projection targets')
        if len(z) != len(self.features):
            raise ValueError('Projection targets must be aligned with outer-training rows.')
        k = self.features.shape[1]
        coeff = np.zeros((self.n_steps, k, z.shape[1]))
        for t in range(self.n_steps):
            take = self.steps == t
            rhs = (self.features[take].T * self.weights[take]) @ z[take] / take.sum()
            coeff[t] = self.inverses[t] @ rhs
        if not np.isfinite(coeff).all():
            raise FloatingPointError('Nonfinite projection coefficients.')
        return coeff

    def predict(self, coordinates, phi, steps):
        phi = _array(phi, 2, 'prediction modifiers')
        if phi.shape[1] != self.input_dim:
            raise ValueError('Modifier definition changed after fitting.')
        f = interaction_features(phi, self.structure)
        st = check_steps(steps, len(f), self.n_steps)
        c = np.asarray(coordinates, float)
        if c.ndim != 3 or c.shape[:2] != (self.n_steps, f.shape[1]) or not np.isfinite(c).all():
            raise ValueError('Invalid frozen coordinates.')
        out = np.empty((len(f), c.shape[2]))
        for t in np.unique(st):
            take = st == t
            out[take] = f[take] @ c[t]
        return out

    def training_residual_summary(self, coordinates, targets):
        """Only aggregate diagnostics; no predictions or coefficients are saved."""
        z = np.asarray(targets, float)
        if z.ndim == 1:
            z = z[:, None]
        fitted = np.empty_like(z)
        max_normal = 0.
        for t in range(self.n_steps):
            take = self.steps == t
            fitted[take] = self.features[take] @ coordinates[t]
            moment = (self.features[take].T * self.weights[take]) @ (z[take] - fitted[take]) / take.sum()
            if moment.size:
                max_normal = max(max_normal, float(np.max(np.abs(moment))))
        residual = z - fitted
        return dict(weighted_residual_mse=float(np.mean(self.weights[:, None] * residual**2)),
                    max_abs_prediction_change=float(np.max(np.abs(residual))),
                    max_normal_moment=max_normal)


def project_outcome(base_mu, probabilities, target_interaction):
    """Closest four-cell prediction in the supplied probability metric.

    Does not merely replace the 11 cell. Never refits/changes the base learner.
    """
    p = check_probabilities(probabilities)
    mu = _array(base_mu, 2, 'base outcome predictions')
    tau = np.asarray(target_interaction, float).reshape(-1)
    if mu.shape != p.shape or tau.shape != (len(p),) or not np.isfinite(tau).all():
        raise ValueError('Misaligned projected outcome inputs.')
    d = (mu @ K - tau) * weight_j(p)
    result = mu - (K[None, :] / p) * d[:, None]
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite projected outcome.')
    return result


def project_representer(scales, probabilities, target_interaction):
    """Exact cell projection given learned p and frozen low-dim query coordinates.

    Main-effect directions are NOT fitted using a large empirical series Gram.
    Only the interaction function is restricted.
    """
    p = check_probabilities(probabilities)
    s = _array(scales, 3, 'query multipliers')
    ell = query_ell(s, p)
    g = _array(target_interaction, 2, 'query projected interaction')
    if g.shape != ell.shape:
        raise ValueError('Query coordinate count changed.')
    alpha = s / p[:, None, :] - (K[None, :] / p)[:, None, :] * (
        weight_j(p)[:, None] * (ell - g))[:, :, None]
    if not np.isfinite(alpha).all():
        raise FloatingPointError('Nonfinite projected representer.')
    return alpha
