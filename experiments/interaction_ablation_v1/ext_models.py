"""Observable-data external weight learners. No simulator truth is accepted.

RieszBoost is a local Python implementation of Lee & Schuler, arXiv:2501.04871v2,
Algorithm 2, generalized to a known four-arm linear functional. It is not the
authors' software and does NOT replace Riesz loss by treatment classification.
"""
from __future__ import annotations
from dataclasses import dataclass
import inspect
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning


class FitFailure(RuntimeError):
    """A specified learner could not be fitted; never use a silent fallback."""


def _finite_matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite feature matrix.')
    return x


def check_actions(a, n):
    a = np.asarray(a)
    if a.shape != (n,) or not np.isin(a, np.arange(4)).all():
        raise ValueError('Actions must be integers in archived order [00,10,01,11].')
    return a.astype(int)


def check_support(a):
    if not np.array_equal(np.unique(a), np.arange(4)):
        raise FitFailure('A training landmark lacks an observed arm; no fallback fit.')


def smooth_probabilities(p, floor):
    p = np.asarray(p, float)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError('Invalid four-arm probability matrix.')
    if not (0 < floor < 0.25) or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-8):
        raise ValueError('Invalid probabilities or smoothing floor.')
    return floor + (1 - 4*floor)*p


def log_loss(p, a):
    a = check_actions(a, len(p))
    if np.any(p <= 0) or not np.isfinite(p).all():
        raise FitFailure('Nonpositive/nonfinite treatment probabilities.')
    return float(-np.log(p[np.arange(len(a)), a]).mean())


def _logit_fit(x, a, C, spec, seed):
    check_support(a)
    scaler = StandardScaler().fit(x)
    kw = dict(C=float(C), solver='lbfgs', max_iter=int(spec['max_iter']),
              tol=float(spec['tol']), random_state=int(seed))
    # lbfgs with >=3 classes is multinomial; older sklearn also accepts an
    # explicit keyword. Avoid deprecated/removed keywords in newer versions.
    if 'multi_class' in inspect.signature(LogisticRegression).parameters:
        kw['multi_class'] = 'multinomial'
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always', ConvergenceWarning)
        fit = LogisticRegression(**kw).fit(scaler.transform(x), a)
    if any(issubclass(w.category, ConvergenceWarning) for w in seen):
        raise FitFailure('Multinomial logistic fit did not converge within the fixed iteration cap.')
    if not np.array_equal(fit.classes_, np.arange(4)):
        raise FitFailure('Multinomial fit did not retain all four action classes.')
    return scaler, fit


def _hgb_new(spec, lr, leaves, iterations, seed, warm=False):
    return HistGradientBoostingClassifier(
        learning_rate=float(lr), max_leaf_nodes=int(leaves), max_iter=int(iterations),
        min_samples_leaf=int(spec['min_samples_leaf']),
        l2_regularization=float(spec['l2_regularization']), early_stopping=False,
        warm_start=warm, random_state=int(seed))


@dataclass
class ProbabilityLearner:
    entries: dict

    def predict(self, x, steps):
        x = _finite_matrix(x); steps = np.asarray(steps, int)
        if steps.shape != (len(x),): raise ValueError('Step alignment error.')
        out = np.empty((len(x), 4))
        for t in np.unique(steps):
            if int(t) not in self.entries: raise ValueError('Unknown landmark.')
            sel = steps == t
            scaler, fit, floor = self.entries[int(t)]
            xx = scaler.transform(x[sel]) if scaler is not None else x[sel]
            out[sel] = smooth_probabilities(fit.predict_proba(xx), floor)
        return out


def fit_probability(kind, x, steps, actions, inner_folds, plan, seed):
    """Tune per landmark using inner holdouts of the OUTER training set only."""
    x = _finite_matrix(x); steps = np.asarray(steps, int)
    a = check_actions(actions, len(x)); inner_folds = [np.asarray(f, int) for f in inner_folds]
    _validate_folds(inner_folds, len(x))
    entries, log = {}, []
    floors = plan['probability_floor_grid']
    for t in np.unique(steps):
        ix = np.flatnonzero(steps == t)
        check_support(a[ix])
        candidate_losses = {}
        candidate_errors = {}
        if kind == 'logit':
            keys = [(float(C), float(fl)) for C in plan['logit']['C'] for fl in floors]
        elif kind == 'hgb':
            sp = plan['hgb']
            keys = [(float(lr), int(leaves), int(m), float(fl))
                    for lr in sp['learning_rate'] for leaves in sp['max_leaf_nodes']
                    for m in sp['iterations'] for fl in floors]
        else: raise ValueError('Unknown probability learner.')
        for key in keys: candidate_losses[key] = []; candidate_errors[key] = []
        for f, hold in enumerate(inner_folds):
            val = hold[steps[hold] == t]
            tr = np.setdiff1d(ix, val)
            if len(val) == 0: raise FitFailure('Empty inner validation landmark.')
            check_support(a[tr])
            if kind == 'logit':
                for C in plan['logit']['C']:
                    affected = [(float(C), float(fl)) for fl in floors]
                    try:
                        sc, model = _logit_fit(x[tr], a[tr], C, plan['logit'], seed+37*f+int(t))
                        raw = model.predict_proba(sc.transform(x[val]))
                        for fl in floors:
                            candidate_losses[float(C),float(fl)].append((len(val),log_loss(smooth_probabilities(raw, fl), a[val])))
                    except (FitFailure, ValueError, FloatingPointError) as e:
                        for key in affected: candidate_errors[key].append(type(e).__name__+': '+str(e))
            else:
                for lr in sp['learning_rate']:
                    for leaves in sp['max_leaf_nodes']:
                        model = _hgb_new(sp, lr, leaves, min(sp['iterations']), seed+37*f+int(t), True)
                        failed = None
                        for m in sorted(sp['iterations']):
                            affected = [(float(lr),int(leaves),int(m),float(fl)) for fl in floors]
                            try:
                                if failed: raise FitFailure(failed)
                                model.set_params(max_iter=int(m))
                                model.fit(x[tr], a[tr])
                                if not np.array_equal(model.classes_, np.arange(4)):
                                    raise FitFailure('Boosting probability fit lacks a class.')
                                raw = model.predict_proba(x[val])
                                for fl in floors:
                                    candidate_losses[float(lr),int(leaves),int(m),float(fl)].append((len(val),log_loss(smooth_probabilities(raw,fl),a[val])))
                            except (FitFailure, ValueError, FloatingPointError) as e:
                                failed = type(e).__name__+': '+str(e)
                                for key in affected: candidate_errors[key].append(failed)
        scored = []
        for key in keys:
            v = candidate_losses[key]
            valid = len(v)==len(inner_folds) and not candidate_errors[key]
            loss = float(sum(n*s for n,s in v)/sum(n for n,s in v)) if valid else None
            record = dict(landmark=int(t),kind=kind,params=list(key),valid=valid,
                          validation_loss=loss,reason='; '.join(candidate_errors[key]))
            log.append(record)
            if valid and np.isfinite(loss): scored.append((loss,keys.index(key),key))
        if not scored: raise FitFailure('No valid '+kind+' candidate at landmark '+str(int(t)))
        best = min(scored)[2]
        for row in log:
            if row['landmark']==int(t): row['selected'] = row['params']==list(best)
        # A selected refit failure is NOT followed by trying the next candidate.
        if kind == 'logit':
            C,fl=best; sc,model=_logit_fit(x[ix],a[ix],C,plan['logit'],seed+int(t))
        else:
            lr,leaves,m,fl=best; sc=None
            model=_hgb_new(sp,lr,leaves,m,seed+int(t));model.fit(x[ix],a[ix])
        entries[int(t)] = (sc,model,fl)
    return ProbabilityLearner(entries), log


def _validate_folds(folds, n):
    joined = np.concatenate(folds)
    if len(joined)!=n or not np.array_equal(np.sort(joined),np.arange(n)):
        raise ValueError('Inner folds must partition the outer-training observations.')


def augmented_predictors(x):
    """Row-major (X_i,a), a=0,1,2,3. Four one-hot action columns, not ordinal."""
    x = _finite_matrix(x)
    return np.ascontiguousarray(np.column_stack([
        np.repeat(x,4,axis=0), np.tile(np.eye(4),(len(x),1))]),dtype=np.float32)


def riesz_residual(alpha, a, scales):
    """-1/2 empirical-loss derivative, up to common 1/n: s_ia - I(A_i=a) alpha_ia."""
    alpha = np.asarray(alpha,float); scales=np.asarray(scales,float)
    if alpha.shape != scales.shape or alpha.ndim!=2 or alpha.shape[1]!=4:
        raise ValueError('Riesz residual expects n x 4 arrays.')
    a=check_actions(a,len(alpha))
    r=scales.copy(); r[np.arange(len(a)),a]-=alpha[np.arange(len(a)),a]
    return r


def riesz_loss(alpha, a, scales):
    alpha=np.asarray(alpha,float);scales=np.asarray(scales,float);a=check_actions(a,len(alpha))
    if alpha.shape!=scales.shape or not np.isfinite(alpha).all() or not np.isfinite(scales).all():
        raise FitFailure('Nonfinite or misaligned Riesz loss.')
    loss=np.mean(alpha[np.arange(len(a)),a]**2 - 2*np.sum(scales*alpha,axis=1))
    if not np.isfinite(loss): raise FitFailure('Nonfinite Riesz validation loss.')
    return float(loss)


@dataclass
class RieszTrees:
    trees: list
    learning_rate: float

    def predict(self, x):
        w=augmented_predictors(x)
        out=np.zeros(len(w))
        for tree in self.trees: out += self.learning_rate*tree.predict(w)
        return out.reshape(len(x),4)


def boost_path(x, a, scales, lr, depth, min_leaf, evaluation_steps, seed,
               validation=None):
    """Algorithm 2 with fixed steps and zero initialization; evaluate candidate iteration counts along one path."""
    x=_finite_matrix(x);a=check_actions(a,len(x));scales=np.asarray(scales,float)
    if scales.shape != (len(x),4) or not np.isfinite(scales).all(): raise ValueError('Invalid query multiplier.')
    w=augmented_predictors(x); pred=np.zeros((len(x),4));trees=[];losses={}
    evaluation_steps=sorted(set(map(int,evaluation_steps)))
    if not evaluation_steps or evaluation_steps[0]<1 or lr<=0: raise ValueError('Invalid boosting schedule.')
    if validation is not None:
        vx,va,vs=validation;vw=augmented_predictors(vx);vp=np.zeros((len(vx),4))
    for it in range(1,max(evaluation_steps)+1):
        r=riesz_residual(pred,a,scales)
        tree=DecisionTreeRegressor(max_depth=int(depth),min_samples_leaf=int(min_leaf),
                                   random_state=int((seed+it)%2147483647))
        tree.fit(w,r.reshape(-1)); trees.append(tree)
        pred += lr*tree.predict(w).reshape(len(x),4)
        if not np.isfinite(pred).all(): raise FitFailure('RieszBoost produced nonfinite predictions.')
        if validation is not None:
            vp += lr*tree.predict(vw).reshape(len(vx),4)
            if it in evaluation_steps: losses[it]=riesz_loss(vp,va,vs)
    return RieszTrees(trees,float(lr)),losses


def fit_riesz_query(x, steps, a, scales, inner_folds, spec, seed, predict_x, predict_steps):
    """Independent trees and observed-loss selection for ONE query. Returns held predictions only."""
    x=_finite_matrix(x);a=check_actions(a,len(x));steps=np.asarray(steps,int)
    scales=np.asarray(scales,float);predict_x=_finite_matrix(predict_x);predict_steps=np.asarray(predict_steps,int)
    _validate_folds(inner_folds,len(x))
    if scales.shape!=(len(x),4):raise ValueError('Each call must contain exactly one query.')
    result=np.zeros((len(predict_x),4));log=[]
    for t in np.unique(steps):
        ix=np.flatnonzero(steps==t);out=predict_steps==t
        if np.all(scales[ix]==0):
            # Exact zero functional at this landmark, not a fitted zero fallback.
            log.append(dict(landmark=int(t),exact_zero_functional=True,selected=True));continue
        check_support(a[ix])
        keys=[(float(lr),int(dep),int(m)) for lr in spec['learning_rate']
              for dep in spec['max_depth'] for m in spec['iterations']]
        loss={k:[] for k in keys};errors={k:[] for k in keys}
        for f,hold in enumerate(inner_folds):
            val=np.asarray(hold,int);val=val[steps[val]==t];tr=np.setdiff1d(ix,val)
            if not len(val):raise FitFailure('Empty RieszBoost validation landmark.')
            check_support(a[tr])
            for lr in spec['learning_rate']:
                for dep in spec['max_depth']:
                    try:
                        _,scores=boost_path(x[tr],a[tr],scales[tr],lr,dep,spec['min_samples_leaf'],
                            spec['iterations'],seed+1009*f+17*int(t),
                            (x[val],a[val],scales[val]))
                        for m,v in scores.items():loss[float(lr),int(dep),int(m)].append((len(val),v))
                    except (ValueError,FitFailure,FloatingPointError) as e:
                        for m in spec['iterations']:errors[float(lr),int(dep),int(m)].append(type(e).__name__+': '+str(e))
        scored=[]
        for key in keys:
            valid=len(loss[key])==len(inner_folds) and not errors[key]
            value=float(sum(n*v for n,v in loss[key])/sum(n for n,v in loss[key])) if valid else None
            log.append(dict(landmark=int(t),exact_zero_functional=False,params=list(key),valid=valid,
                            validation_loss=value,reason='; '.join(errors[key])))
            if valid and np.isfinite(value):scored.append((value,keys.index(key),key))
        if not scored:raise FitFailure('All RieszBoost specifications failed at a landmark.')
        best=min(scored)[2]
        for rec in log:
            if rec['landmark']==int(t):rec['selected']=rec.get('params')==list(best)
        lr,dep,m=best
        model,_=boost_path(x[ix],a[ix],scales[ix],lr,dep,spec['min_samples_leaf'],[m],seed+17*int(t))
        if out.any():result[out]=model.predict(predict_x[out])
    if set(np.unique(predict_steps))-set(np.unique(steps)):raise ValueError('Untrained prediction landmark.')
    if not np.isfinite(result).all():raise FitFailure('Nonfinite held-out Riesz predictions.')
    return result,log
