"""Cross-fitted scores, grouped by independent sampled profiles."""
import numpy as np
from outcomes import orthogonal_scores

Z975 = 1.959963984540054


def profile_summaries(scores, clusters):
    scores = np.asarray(scores, float)
    groups, inv = np.unique(clusters, return_inverse=True)
    count = np.bincount(inv)
    if scores.ndim != 2 or len(scores) != len(inv) or len(groups) < 1:
        raise ValueError('Invalid independent profile scores.')
    if not np.isfinite(scores).all() or not np.all(count == count[0]):
        raise ValueError('Nonfinite scores or incomplete profiles.')
    v = np.zeros((len(groups), scores.shape[1]))
    np.add.at(v, inv, scores)
    v /= count[:, None]
    return v.sum(0), np.sum(v * v, axis=0), len(groups)


def combine_profile_summaries(parts):
    n = sum(int(p[2]) for p in parts)
    if n < 2:
        raise ValueError('Too few independent units.')
    sm = sum(p[0] for p in parts)
    ss = sum(p[1] for p in parts)
    var = np.maximum((ss - sm * sm / n) / (n - 1), 0.)
    return sm / n, np.sqrt(var / n), n
