"""Observed-only additive/sparse Riesz fits for the PHQ-9 and CVD protocols.

Each step/arm correction minimizes mean-within-step[h*(I*u**2 - 2*u)]
plus ridge*||beta||2**2 + 2*l1*||beta||1, with mean-within-step(h)=1.
Both penalties include the intercept. Elementwise L1 is not a rank penalty.
Selection uses the original, unnormalized query scales and projected weights.
The three solver functions below are copied exactly from confirm v1 linear.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

RIDGES = (.003, .03, .3)
L1_VALUES = (0., .01, .05)
GAMMAS = (0., .5, 1.)
KKT_TOLERANCE = 2e-7
FAMILIES = ('independent_additive_ridge', 'independent_sparse_riesz',
            'shared_additive_ridge', 'shared_sparse_riesz')

def _kkt(matrix, rhs, beta, l1):
    grad = matrix @ beta - rhs
    active = beta != 0.
    residual = np.maximum(np.abs(grad) - l1, 0.)
    residual[active] = np.abs(grad[active] + l1 * np.sign(beta[active]))
    return float(np.max(residual))

def _active_polish(matrix, rhs, beta, l1, tolerance):
    """Solve one coefficient-sign region exactly, admitting KKT violators.

    After the warm start, each unconstrained active solve is followed only
    until its first zero crossing.  Once the active gradient is zero, add the
    most violating inactive coordinate.  Every accepted direction decreases
    the same convex objective; termination still requires the full KKT test.
    """
    beta = beta.copy()
    active = beta != 0.
    signs = np.sign(beta)
    for iteration in range(1, 4 * len(beta) + 21):
        ix = np.flatnonzero(active)
        if len(ix):
            proposal = np.linalg.solve(matrix[np.ix_(ix, ix)], rhs[ix] - l1 * signs[ix])
            crossing = proposal * signs[ix] <= 0.
            if crossing.any():
                direction = proposal - beta[ix]
                candidates = np.flatnonzero(crossing)
                denom = direction[candidates]
                fractions = np.divide(-beta[ix[candidates]], denom,
                                      out=np.ones(len(candidates)), where=denom != 0.)
                step = float(np.clip(np.min(fractions), 0., 1.))
                beta[ix] += step * direction
                hit = ix[candidates[fractions <= step + 1e-12]]
                beta[hit] = 0.; active[hit] = False; signs[hit] = 0.
                continue
            beta[ix] = proposal
        grad = matrix @ beta - rhs
        if _kkt(matrix, rhs, beta, l1) <= tolerance:
            return beta, iteration
        violations = np.maximum(np.abs(grad) - l1, 0.)
        violations[active] = 0.
        j = int(np.argmax(violations))
        if violations[j] <= tolerance:
            # Numerical residual inside an active solve remains explicit.
            return beta, iteration
        active[j] = True
        signs[j] = -np.sign(grad[j])
    return beta, iteration

def solve_quadratic(gram, rhs, ridge, l1=0., initial=None, max_sweeps=3000):
    """Cyclic coordinate descent for a coercive quadratic plus elementwise L1."""
    matrix = (np.asarray(gram) + np.asarray(gram).T) / 2.
    matrix = matrix + float(ridge) * np.eye(len(rhs))
    rhs = np.asarray(rhs, dtype=float)
    if ridge <= 0 or l1 < 0 or not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError('Finite quadratic, positive ridge and nonnegative l1 required.')
    tolerance = KKT_TOLERANCE * max(1., float(np.max(np.abs(rhs))))
    if l1 == 0.:
        beta = np.linalg.solve(matrix, rhs)
        error = _kkt(matrix, rhs, beta, l1)
        return beta, dict(kkt_max=error, kkt_tolerance=tolerance,
                          converged=bool(error <= tolerance), sweeps=0)
    beta = np.zeros(len(rhs)) if initial is None else np.asarray(initial).copy()
    residual = matrix @ beta - rhs
    diagonal = np.diag(matrix)
    error = np.inf
    active_steps = 0
    for sweep in range(1, int(max_sweeps) + 1):
        for j in range(len(beta)):
            proposal = beta[j] - residual[j] / diagonal[j]
            updated = np.sign(proposal) * max(abs(proposal) - l1 / diagonal[j], 0.)
            delta = updated - beta[j]
            if delta:
                beta[j] = updated
                residual += delta * matrix[:, j]
        if sweep <= 5 or sweep % 5 == 0:
            # Recompute instead of trusting accumulated coordinate-roundoff.
            residual = matrix @ beta - rhs
            error = _kkt(matrix, rhs, beta, l1)
            if error <= tolerance:
                break
            if sweep == 50 and max_sweeps >= 50:
                beta, active_steps = _active_polish(matrix, rhs, beta, l1, tolerance)
                residual = matrix @ beta - rhs
                error = _kkt(matrix, rhs, beta, l1)
                if error <= tolerance:
                    break
    error = _kkt(matrix, rhs, beta, l1)
    return beta, dict(kkt_max=error, kkt_tolerance=tolerance,
                      converged=bool(error <= tolerance), sweeps=int(sweep),
                      active_steps=int(active_steps))


def _x(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Expected a finite nonempty sieve matrix.')
    return value


def _scaling(X):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    magnitude = np.maximum(1., np.max(np.abs(X), axis=0))
    tolerance = 64. * np.finfo(float).eps * magnitude
    constant = (np.ptp(X, axis=0) <= tolerance) | (scale <= tolerance)
    return mean, np.where(constant, 1., scale), constant, tolerance


@dataclass
class AdditiveMap:
    """Generic train-fitted sieve expansion; no raw-variable ordering assumed."""
    input_dimension: int
    kept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    continuous: np.ndarray
    nonlinear_kept: np.ndarray
    nonlinear_mean: np.ndarray
    nonlinear_scale: np.ndarray
    metadata: dict

    @classmethod
    def fit(cls, X):
        X = _x(X)
        mean, scale, constant, tolerance = _scaling(X)
        kept = np.flatnonzero(~constant)
        # Any observed two-level direction is left linear. This also recognizes
        # binaries already standardized by the archived sieve construction.
        lo, hi = X.min(axis=0), X.max(axis=0)
        binary = np.all((np.abs(X - lo) <= tolerance) | (np.abs(X - hi) <= tolerance), axis=0)
        continuous = np.flatnonzero(~binary[kept])
        z = (X[:, kept] - mean[kept]) / scale[kept]
        if len(continuous):
            cont = z[:, continuous]
            bounded = cont / np.hypot(1., cont)
            nonlinear = np.column_stack([np.tanh(cont), bounded * bounded])
            nm, ns, nc, _ = _scaling(nonlinear)
            nk = np.flatnonzero(~nc)
        else:
            nm = ns = np.array([], dtype=float)
            nk = np.array([], dtype=int)
        metadata = dict(input_dimension=int(X.shape[1]),
                        constant_input_directions_removed=int(constant.sum()),
                        retained_linear_directions=int(len(kept)),
                        observed_binary_directions=int(binary[kept].sum()),
                        expanded_continuous_directions=int(len(continuous)),
                        retained_nonlinear_directions=int(len(nk)),
                        dimension=int(1 + len(kept) + len(nk)))
        return cls(X.shape[1], kept, mean[kept], scale[kept], continuous,
                   nk, nm[nk], ns[nk], metadata)

    def transform(self, X):
        X = _x(X)
        if X.shape[1] != self.input_dimension:
            raise ValueError('Sieve input dimension changed after feature-map fitting.')
        z = (X[:, self.kept] - self.mean) / self.scale
        columns = [np.ones((len(X), 1)), z]
        if len(self.nonlinear_kept):
            cont = z[:, self.continuous]
            bounded = cont / np.hypot(1., cont)
            nonlinear = np.column_stack([np.tanh(cont), bounded * bounded])
            columns.append((nonlinear[:, self.nonlinear_kept] - self.nonlinear_mean) / self.nonlinear_scale)
        transformed = np.column_stack(columns)
        if not np.isfinite(transformed).all():
            raise FloatingPointError('Nonfinite additive features.')
        return transformed


def _validate(artifacts, queries, pbar, core):
    pbar = np.asarray(pbar, dtype=float)
    if pbar.shape != (core.N_STEPS, 4) or not np.isfinite(pbar).all() or np.any(pbar <= 0):
        raise ValueError('Expected positive finite step-by-action pilot probabilities.')
    for artifact in artifacts:
        X = _x(artifact['x']); n = len(X)
        if n != int(artifact['n']):
            raise ValueError('Artifact sample size mismatch.')
        for name, upper in [('steps', core.N_STEPS), ('actions', 4)]:
            arr = np.asarray(artifact[name])
            if arr.shape != (n,) or not np.isin(arr, np.arange(upper)).all():
                raise ValueError('Invalid artifact ' + name)
        if not np.isin(np.arange(core.N_STEPS), artifact['steps']).all():
            raise ValueError('Every observation fold must contain every landmark step.')
    train = np.asarray(queries['indices']['train'], dtype=int)
    selection = np.asarray(queries['indices']['selection'], dtype=int)
    held = np.asarray(queries['indices']['held_out'], dtype=int)
    if not len(train) or not len(selection) or not len(held):
        raise ValueError('Nonempty train, selection and held-out query sets required.')
    if np.intersect1d(np.r_[train, selection], held).size:
        raise ValueError('Held-out queries overlap fitting or selection queries.')
    return pbar


def _scales_squared(artifact, queries, indices, landmark_probability):
    indices = np.asarray(indices, dtype=int)
    steps = np.asarray(artifact['steps'], dtype=int)
    omega = queries['omega_unique'][queries['inverse_eta'][indices, None],
                                      np.asarray(artifact['rows'], dtype=int)[None, :]].T
    base = omega * queries['h'][indices[:, None], steps[None, :]].T / landmark_probability
    return base[:, :, None] ** 2 * queries['c'][indices][None, :, :] ** 2


def observed_loss(inverse, artifact, queries, indices, landmark_probability):
    """Exact unnormalized observed Riesz loss averaged over samples/queries."""
    u = np.asarray(inverse, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if u.shape != (artifact['n'], len(indices), 4) or not np.isfinite(u).all():
        raise ValueError('Invalid inverse predictions for observed scoring.')
    weights = _scales_squared(artifact, queries, indices, landmark_probability)
    ii = np.arange(artifact['n'])[:, None]
    qq = np.arange(len(indices))[None, :]
    aa = np.asarray(artifact['actions'], dtype=int)[:, None]
    return float(np.mean(weights[ii, qq, aa] * u[ii, qq, aa] ** 2
                         - 2. * np.sum(weights * u, axis=2)))


def _blocks(artifact, queries, indices, core, shared):
    """Yield step-local h before normalizing; zero-scale blocks are absent."""
    indices = np.asarray(indices, dtype=int)
    steps = np.asarray(artifact['steps'], dtype=int)
    eta_ids = np.asarray(queries['inverse_eta'], dtype=int)
    for t in range(core.N_STEPS):
        mask = steps == t
        rows = np.asarray(artifact['rows'], dtype=int)[mask]
        if shared:
            eta_unique, inverse = np.unique(eta_ids[indices], return_inverse=True)
            omega2 = queries['omega_unique'][eta_unique[:, None], rows[None, :]] ** 2
            for a in range(4):
                factors = (queries['h'][indices, t] * queries['c'][indices, a] / core.P_LANDMARK) ** 2
                if not np.any(factors > 0):
                    continue
                eta_weight = np.bincount(inverse, weights=factors, minlength=len(eta_unique)) / len(indices)
                yield (t, a), mask, eta_weight @ omega2
        else:
            for eta in np.unique(eta_ids[indices]):
                group = indices[eta_ids[indices] == eta]
                # Proportional nonzero query-scale constants disappear only
                # because this protocol normalizes each step/arm separately.
                omega2 = queries['omega_unique'][eta, rows] ** 2
                for a in range(4):
                    active = (queries['h'][group, t] != 0) & (queries['c'][group, a] != 0)
                    if np.any(active):
                        yield (int(eta), t, a), mask, omega2


def _fit_paths(artifact, queries, indices, pbar, core, geometry, shared, pairs, phase):
    Phi = geometry.transform(artifact['x'])
    pairs = sorted(set((float(r), float(l)) for r, l in pairs))
    paths = {pair: dict(coefficients={}, certificates=[], failures=[], valid=True) for pair in pairs}
    diagnostics = []
    for block, mask, raw_weights in _blocks(artifact, queries, indices, core, shared):
        t, a = block[-2:]
        X = Phi[mask]
        A = np.asarray(artifact['actions'])[mask]
        weights = raw_weights / raw_weights.mean()
        observed = weights * (A == a)
        gram = X.T @ (observed[:, None] * X) / len(X)
        rhs = X.T @ (weights - observed / pbar[t, a]) / len(X)
        eig_failure = ''
        try:
            values, vectors = np.linalg.eigh((gram + gram.T) / 2.)
            transformed = vectors.T @ rhs
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            eig_failure = str(error)
        for ridge in sorted(set(pair[0] for pair in pairs)):
            initial = None
            if not eig_failure:
                initial = vectors @ (transformed / (np.maximum(values, 0.) + ridge))
            for l1 in sorted(pair[1] for pair in pairs if pair[0] == ridge):
                entry = paths[(ridge, l1)]
                reason = eig_failure
                certificate = None
                try:
                    if reason:
                        raise FloatingPointError(reason)
                    if l1 == 0.:
                        beta = initial
                        error = _kkt(gram + ridge * np.eye(len(rhs)), rhs, beta, 0.)
                        tolerance = KKT_TOLERANCE * max(1., float(np.max(np.abs(rhs))))
                        certificate = dict(kkt_max=error, kkt_tolerance=tolerance,
                                           converged=bool(error <= tolerance), sweeps=0, active_steps=0)
                    else:
                        beta, certificate = solve_quadratic(gram, rhs, ridge, l1, initial)
                    if not certificate['converged'] or not np.isfinite(beta).all():
                        reason = 'Quadratic/L1 block failed its full KKT certificate.'
                    else:
                        entry['coefficients'][block] = beta
                        initial = beta
                except (np.linalg.LinAlgError, FloatingPointError) as error:
                    reason = f'{type(error).__name__}: {error}'
                if reason:
                    entry['valid'] = False
                    entry['failures'].append(dict(block=list(block), reason=reason))
                if certificate is not None:
                    entry['certificates'].append(certificate)
                diagnostics.append(dict(phase=phase, sharing='shared' if shared else 'independent',
                    step=int(t), action=int(a), eta_index=None if shared else int(block[0]),
                    ridge=ridge, l1=l1, status='invalid_numerical' if reason else 'valid',
                    converged=bool(certificate is not None and certificate['converged'] and not reason),
                    failure_reason=reason, within_step_n=int(len(X)),
                    observed_action_n=int(np.sum(A == a)), dimension=int(X.shape[1]),
                    weight_mean_before_normalization=float(raw_weights.mean()),
                    kkt_max=None if certificate is None else certificate['kkt_max'],
                    kkt_tolerance=None if certificate is None else certificate['kkt_tolerance'],
                    sweeps=None if certificate is None else certificate['sweeps'],
                    active_steps=None if certificate is None else certificate.get('active_steps', 0)))
    return paths, diagnostics


@dataclass
class SparseRieszModel:
    geometry: AdditiveMap | None
    pbar: np.ndarray
    queries: dict
    coefficients: dict
    shared: bool
    target_indices: np.ndarray
    choice: dict

    def _predict_phi(self, Phi, steps, query_indices):
        steps = np.asarray(steps, dtype=int)
        query_indices = np.asarray(query_indices, dtype=int)
        if not self.shared and not np.isin(query_indices, self.target_indices).all():
            raise ValueError('Independent model requested an unfitted target query.')
        pilot = 1. / self.pbar[steps]
        result = np.broadcast_to(pilot[:, None, :], (len(steps), len(query_indices), 4)).copy()
        if self.choice.get('pilot') or self.choice['gamma'] == 0.:
            return result
        gamma = float(self.choice['gamma'])
        eta_ids = self.queries['inverse_eta'][query_indices]
        for t in np.unique(steps):
            mask = steps == t
            for a in range(4):
                active = (self.queries['h'][query_indices, t] != 0) & (self.queries['c'][query_indices, a] != 0)
                if not np.any(active):
                    continue
                if self.shared:
                    beta = self.coefficients.get((int(t), a))
                    if beta is not None:
                        result[np.ix_(mask, active, [a])] += (gamma * (Phi[mask] @ beta))[:, None, None]
                else:
                    for eta in np.unique(eta_ids[active]):
                        qmask = active & (eta_ids == eta)
                        beta = self.coefficients.get((int(eta), int(t), a))
                        if beta is None:
                            raise RuntimeError('Missing fitted independent step/arm coefficient.')
                        result[np.ix_(mask, qmask, [a])] += (gamma * (Phi[mask] @ beta))[:, None, None]
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite unprojected Riesz prediction.')
        return result

    def predict_raw(self, X, steps, query_indices):
        X = _x(X)
        steps = np.asarray(steps)
        if steps.shape != (len(X),) or not np.isin(steps, np.arange(len(self.pbar))).all():
            raise ValueError('Invalid prediction landmarks.')
        Phi = None if self.geometry is None else self.geometry.transform(X)
        return self._predict_phi(Phi, steps, query_indices)

    def predict(self, X, steps, query_indices):
        return np.clip(self.predict_raw(X, steps, query_indices), 1., 100.)


def _pilot_choice(family, loss):
    return dict(method_family=family, pilot=True, ridge=0., l1=0., gamma=0.,
                validation_loss=float(loss), status='valid', failure_reason='')


def _is_shared(family):
    return family.startswith('shared_')


def _is_sparse(family):
    return 'sparse' in family


def fit_observed(artifacts, queries, pbar, core, progress=None):
    """Two-fold observed selection followed by one frozen final refit.

    There is no simulator-probability, oracle-risk or outcome argument.
    Independent held targets supply their own observed weighted moments.
    """
    if len(artifacts) != 3:
        raise ValueError('Expected two observation folds and one full artifact.')
    pbar = _validate(artifacts, queries, pbar, core)
    say = progress or (lambda message: None)
    train = np.asarray(queries['indices']['train'], dtype=int)
    selection = np.asarray(queries['indices']['selection'], dtype=int)
    held = np.asarray(queries['indices']['held_out'], dtype=int)
    fit_final = np.asarray(queries['indices'].get('fit_final', np.r_[train, selection]), dtype=int)
    if np.intersect1d(fit_final, held).size:
        raise ValueError('Final shared fitting queries contain held-out targets.')
    pairs = [(r, l) for r in RIDGES for l in L1_VALUES]
    folds, diagnostics, dimensions = [], [], []
    pilot_losses = []
    for fold in range(2):
        fit, val = artifacts[fold], artifacts[1 - fold]
        geometry = AdditiveMap.fit(fit['x'])
        dimensions.append(dict(phase=f'validation_fit_{fold}', **geometry.metadata))
        say(f'Riesz fold {fold + 1}/2: {fit["n"]} draws; additive dimension {geometry.metadata["dimension"]}; separate {core.N_STEPS} steps.')
        item = dict(geometry=geometry, validation_phi=geometry.transform(val['x']))
        for shared in (False, True):
            paths, logs = _fit_paths(fit, queries, train if shared else selection,
                                    pbar, core, geometry, shared, pairs, f'validation_fit_{fold}')
            item[shared] = paths
            diagnostics.extend(logs)
        folds.append(item)
        pilot = np.broadcast_to((1. / pbar[val['steps']])[:, None, :], (val['n'], len(selection), 4))
        pilot_losses.append(observed_loss(np.clip(pilot, 1., 100.), val, queries, selection, core.P_LANDMARK))
    pilot_loss = float(np.mean(pilot_losses))
    rows, choices, score_cache = [], {}, {}
    for family in FAMILIES:
        candidates = [_pilot_choice(family, pilot_loss)]
        shared = _is_shared(family)
        for ridge in RIDGES:
            for l1 in L1_VALUES if _is_sparse(family) else (0.,):
                for gamma in GAMMAS:
                    cache_key = (shared, ridge, l1, gamma)
                    if cache_key not in score_cache:
                        losses, failures = [], []
                        for fold in range(2):
                            entry = folds[fold][shared][(ridge, l1)]
                            if not entry['valid']:
                                failures.extend(f'fold={fold}; block={f["block"]}; {f["reason"]}' for f in entry['failures'])
                                continue
                            val = artifacts[1 - fold]
                            choice = dict(pilot=False, ridge=ridge, l1=l1, gamma=gamma)
                            model = SparseRieszModel(folds[fold]['geometry'], pbar, queries,
                                entry['coefficients'], shared, selection, choice)
                            try:
                                raw = model._predict_phi(folds[fold]['validation_phi'], val['steps'], selection)
                                score = observed_loss(np.clip(raw, 1., 100.), val, queries, selection, core.P_LANDMARK)
                                if not np.isfinite(score):
                                    raise FloatingPointError('Nonfinite observed validation loss.')
                                losses.append(score)
                            except FloatingPointError as error:
                                failures.append(f'fold={fold}; {type(error).__name__}: {error}')
                        score_cache[cache_key] = (float(np.mean(losses)) if not failures else None, '; '.join(failures))
                    score, reason = score_cache[cache_key]
                    candidates.append(dict(method_family=family, pilot=False, ridge=float(ridge),
                        l1=float(l1), gamma=float(gamma), validation_loss=score,
                        status='invalid_numerical' if reason else 'valid', failure_reason=reason))
        rows.extend(candidates)
        valid = [c for c in candidates if c['status'] == 'valid']
        # Exact ties prefer the explicit pilot, then lower gamma/stronger penalty.
        choices[family] = dict(min(valid, key=lambda c: (c['validation_loss'], not c['pilot'],
                                                         c['gamma'], -c['ridge'], -c['l1'])))
    say('All four sparse/additive Riesz choices frozen; refitting on all observed draws.')
    full = artifacts[2]
    final_geometry = None
    if any(not c['pilot'] for c in choices.values()):
        final_geometry = AdditiveMap.fit(full['x'])
        dimensions.append(dict(phase='final', **final_geometry.metadata))
    final_paths = {}
    for shared in (False, True):
        required = {(c['ridge'], c['l1']) for family, c in choices.items()
                    if _is_shared(family) == shared and not c['pilot']}
        if required:
            path, logs = _fit_paths(full, queries, fit_final if shared else held,
                                   pbar, core, final_geometry, shared, required, 'final')
            diagnostics.extend(logs)
            final_paths[shared] = path
    models = {}
    final_certificates_by_method = {}
    for family, choice in choices.items():
        shared = _is_shared(family)
        entry = None if choice['pilot'] else final_paths[shared][(choice['ridge'], choice['l1'])]
        if entry is not None and not entry['valid']:
            first = entry['failures'][0]
            raise FloatingPointError(f'Final {family} block {first["block"]} failed: {first["reason"]}')
        models[family] = SparseRieszModel(None if choice['pilot'] else final_geometry, pbar, queries,
            {} if entry is None else entry['coefficients'], shared, held, dict(choice))
        certs = [] if entry is None else entry['certificates']
        final_certificates_by_method[family] = dict(pilot=bool(choice['pilot']),
            solver_blocks=int(len(certs)), all_kkt_certified=bool(all(c['converged'] for c in certs)),
            max_kkt_ratio=float(max((c['kkt_max'] / c['kkt_tolerance'] for c in certs), default=0.)))
    final_logs = [row for row in diagnostics if row['phase'] == 'final']
    final_ratios = [row['kkt_max'] / row['kkt_tolerance'] for row in final_logs
                    if row['kkt_max'] is not None and row['kkt_tolerance'] is not None]
    final_unique_blocks = {(row['sharing'], row['eta_index'], row['step'], row['action']) for row in final_logs}
    inactive_target_blocks = int(np.sum((queries['h'][held, :, None] * queries['c'][held, None, :]) == 0.))
    return dict(models=models, choices=choices, selection_table=pd.DataFrame(rows),
                fit_diagnostics=pd.DataFrame(diagnostics), checks=dict(
                    choices_frozen_before_oracle=True, explicit_pilot_fallback=True,
                    separate_step_action_functions=True, no_time_pooling=True,
                    normalized_training_weights_within_step_arm=True,
                    unnormalized_validation_query_scales=True,
                    candidate_counts={family: sum(row['method_family'] == family for row in rows) for family in FAMILIES},
                    invalid_candidate_records=sum(row['status'] != 'valid' for row in rows),
                    feature_dimensions=dimensions,
                    final_solver_blocks=int(len(final_logs)),
                    final_distinct_step_action_fit_blocks=int(len(final_unique_blocks)),
                    final_all_kkt_certified=bool(all(row['converged'] for row in final_logs)),
                    final_max_kkt_ratio=float(max(final_ratios, default=0.)),
                    final_certificates_by_method=final_certificates_by_method,
                    held_query_zero_scale_step_arm_blocks=inactive_target_blocks,
                    projection_range=[1., 100.], sparse_penalty='elementwise_all_correction_coefficients'))
