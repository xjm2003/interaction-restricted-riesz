"""Artificial longitudinal covariates for implementation checks, not real PHQ-9."""
import numpy as np
import pandas as pd
import qaoct_backend as backend


def make_mock(patients=1000, seed=2026091401):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(patients, 7))
    steps = np.tile(np.arange(4), patients)
    repeated = np.repeat(latent, 4, axis=0)
    state = repeated + .25 * rng.normal(size=repeated.shape)
    severity = np.clip(15 + 5 * repeated[:, 0], 0, 27)
    history = pd.DataFrame({"PATID": np.repeat([f"mock_{i}" for i in range(patients)], 4),
                            "step": steps, "row_number": np.arange(len(steps)),
                            "baseline_phq9": severity,
                            "tvx_has_post_phq_before_step": (steps > 0).astype(int),
                            "tvx_post_phq_last_change_from_baseline": 3 * state[:, 1],
                            "prior_antidepressant_exposure_count": np.maximum(steps * (state[:, 2] > 0), 0),
                            "prior_psychotherapy_exposure_count": np.maximum(steps * (state[:, 3] > .5), 0)})
    x = np.column_stack([np.ones(len(steps)), state, np.tanh(state[:, :4])])
    # Fixed representation, constructed without the target inverse probability.
    logits = np.column_stack([np.zeros(len(steps)), .4 * state[:, 0] - .3 * state[:, 1],
                              -.7 + .5 * state[:, 2], -1 + .4 * state[:, 0] * state[:, 2]])
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = .08 + .68 * prob / prob.sum(axis=1, keepdims=True)
    for j, action in enumerate(backend.ACTION_ORDER):
        history[f"p_star__{action}"] = prob[:, j]
    contrasts = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                          [-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [.25, .25, .25, .25]])
    eta_bank = rng.uniform(-.45, .45, size=(28, 3))
    qrows = []
    splits = ["train"] * 144 + ["selection"] * 32 + ["held_out"] * 48
    for i, split in enumerate(splits):
        h = np.eye(4)[rng.integers(4)] if i % 3 else rng.dirichlet(np.ones(4))
        c = contrasts[rng.integers(len(contrasts))]
        eta = eta_bank[i % len(eta_bank)]
        z = np.einsum("t,a,k->tak", h, c, np.r_[1.0, eta]).ravel()
        row = {"query_id": f"mock_query_{i:03d}", "query_split": split,
               "support_family": ["common", "rare_01", "rare_11", "policy"][i % 4],
               "pilot_correction_M2": 0.0}
        row.update({f"h_{j}": h[j] for j in range(4)})
        row.update({f"c__{a}": c[j] for j, a in enumerate(backend.ACTION_ORDER)})
        row.update({f"eta_{j}": eta[j] for j in range(3)})
        row.update({f"z__{j}": value for j, value in enumerate(z)})
        qrows.append(row)
    library = pd.DataFrame(qrows)
    descriptors = backend.validate_query_library(library)
    queries = backend.prepare_queries(library, descriptors, steps, backend.build_tilt_features(history))
    grams = np.zeros((4, 4, x.shape[1], x.shape[1]))
    moments = np.zeros((len(library), 4, 4, x.shape[1]))
    total = np.zeros(len(library))
    for t in range(4):
        mask = steps == t
        rows = np.flatnonzero(mask)
        xx, pp = x[mask], prob[mask]
        omega = queries["omega_unique"][queries["inverse_eta"][:, None], rows[None, :]]
        for a in range(4):
            grams[t, a] = xx.T @ (pp[:, a, None] * xx) / len(x)
            truth = (queries["h"][:, t, None] / .25 * omega * queries["c"][:, a, None] *
                     (1 / pp[:, a] - 1 / pp[:, a].mean())[None, :])
            moments[:, t, a] = (truth * pp[:, a]) @ xx / len(x)
            total += np.sum(truth ** 2 * pp[:, a], axis=1) / len(x)
    floor = total.copy()
    for t in range(4):
        for a in range(4):
            floor -= np.einsum("qd,df,qf->q", moments[:, t, a],
                               np.linalg.pinv(grams[t, a], rcond=1e-10), moments[:, t, a])
    library["pilot_correction_M2"] = total
    return dict(population=history, x_population=x, queries=queries, query_library=library,
                population_grams=grams, oracle_moments=moments, floor=np.maximum(floor, 0))
