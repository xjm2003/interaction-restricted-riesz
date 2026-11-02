"""Outcome learners are identical across all Riesz estimators in a replicate."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


def observed_features(x,steps,actions,n_steps):
    return np.column_stack([x,np.eye(n_steps)[steps],np.eye(4)[actions]])


def fit_outcomes(x,steps,actions,y,n_steps,seed):
    # One fixed specification each; no true mu enters either fit.
    designs=observed_features(x,steps,actions,n_steps)
    scaler=StandardScaler().fit(designs)
    linear=Ridge(alpha=10.).fit(scaler.transform(designs),y)
    flexible=HistGradientBoostingRegressor(learning_rate=.08,max_iter=160,max_leaf_nodes=15,
        min_samples_leaf=30,l2_regularization=5.,early_stopping=False,random_state=seed).fit(designs,y)
    return {'linear':(scaler,linear),'flexible':(None,flexible)}


def predict_outcome(model,x,steps,n_steps):
    scaler,learner=model
    preds=[]
    for a in range(4):
        design=observed_features(x,steps,np.full(len(x),a,int),n_steps)
        if scaler is not None: design=scaler.transform(design)
        preds.append(learner.predict(design))
    return np.column_stack(preds)


def orthogonal_scores(scale,alpha,mu_hat,actions,y):
    plug=np.sum(scale*mu_hat[:,None,:],axis=2)
    observed_alpha=alpha[np.arange(len(alpha))[:,None],np.arange(alpha.shape[1])[None,:],actions[:,None]]
    residual=y-mu_hat[np.arange(len(y)),actions]
    return plug+observed_alpha*residual[:,None]


def cluster_estimate(scores,clusters):
    """Equal-size sampled profiles: average row scores within independent units."""
    labels,inverse=np.unique(clusters,return_inverse=True)
    counts=np.bincount(inverse)
    if len(labels)<2 or not np.all(counts==counts[0]):
        raise ValueError('Expected at least two independent, equal-size sampling units.')
    totals=np.zeros((len(labels),scores.shape[1])); np.add.at(totals,inverse,scores)
    unit_scores=totals/counts[:,None]
    theta=unit_scores.mean(0)
    se=np.sqrt(unit_scores.var(0,ddof=1)/len(unit_scores))
    return theta,se
