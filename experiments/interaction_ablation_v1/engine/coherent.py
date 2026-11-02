"""Reuse archived CE/CAL fits; compare observed Riesz vs action-log-loss selection.

No simulator probabilities, true outcome surfaces, or held-query scores are
arguments. Histogram boosting is a standard propensity baseline, not RieszBoost.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import logit_model as old
import sparse_model

@dataclass
class TreeProbability:
    entries: dict
    floor: float
    pbar: np.ndarray
    pilot: bool = False

    def predict_probabilities(self, x, steps):
        result = np.empty((len(x), 4))
        for t in np.unique(steps):
            take = np.asarray(steps) == t
            if self.pilot:
                result[take] = self.pbar[int(t)]
            else:
                fit = self.entries[int(t)]
                raw = np.zeros((take.sum(), 4))
                raw[:, np.asarray(fit.classes_, dtype=int)] = fit.predict_proba(x[take])
                result[take] = self.floor + (1 - 4*self.floor)*raw
        if (not np.isfinite(result).all() or np.any(result <= 0)
                or not np.allclose(result.sum(1), 1, atol=1e-8, rtol=0)):
            raise ValueError('Invalid probability predictions.')
        return result


def action_log_loss(probabilities, actions):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(actions), 4) or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError('Invalid action probability array.')
    return float(-np.log(p[np.arange(len(actions)), np.asarray(actions, int)]).mean())


def observed_scores(model, val, queries, selection, core):
    p = model.predict_probabilities(val['x'], val['steps'])
    u = np.broadcast_to((1/p)[:, None, :], (len(p), len(selection), 4))
    return {'riesz': sparse_model.observed_loss(u, val, queries, selection, core.P_LANDMARK),
            'action_ce': action_log_loss(p, val['actions'])}


def tree_fit(artifact, core, spec, seed):
    entries = {}
    for t in range(core.N_STEPS):
        take = artifact['steps'] == t
        if take.sum() < 8 or len(np.unique(artifact['actions'][take])) < 4:
            raise ValueError('Missing action support in a tree fitting block.')
        model = HistGradientBoostingClassifier(
            learning_rate=.08, max_iter=120, max_leaf_nodes=spec['leaves'],
            min_samples_leaf=spec['min_leaf'], l2_regularization=5.,
            early_stopping=False, random_state=int(seed)+int(t))
        model.fit(artifact['x'][take], artifact['actions'][take])
        entries[int(t)] = model
    return entries


def choose_records(records, rule):
    valid = [r for r in records if r['valid'] and np.isfinite(r[rule])]
    if not valid:
        raise RuntimeError('No valid coherent candidate, including pilot.')
    best = valid[0]
    for r in valid[1:]:
        if r[rule] < best[rule] - 1e-12:
            best = r
    return dict(best)


def fit_observed(artifacts, queries, pbar, core, seed=0, include_tree=True, progress=None):
    """Return models under two matched selection rules, reusing trained paths."""
    say = progress or (lambda _: None)
    selection = np.asarray(queries['indices']['selection'], int)
    pairs = {(loss,r,l) for loss in ('ce','cal') for r in old.RIDGES for l in old.L1_VALUES}
    paths = []
    fit_logs = []
    for fold in range(2):
        say(f'CE/CAL inner fold {fold+1}/2; separate fits for {core.N_STEPS} landmark(s).')
        p, logs, _ = old._fit_paths(artifacts[fold], core, pairs, f'inner_{fold}')
        paths.append(p); fit_logs.extend(logs)
    pilot = old.SharedLogitModel(pbar, {}, dict(pilot=True, floor=0.))
    pilot_scores = [observed_scores(pilot, artifacts[1-f], queries, selection, core) for f in range(2)]
    records = []
    choices = {}
    families = list(old.FAMILIES)
    cache = {}
    for family in families:
        loss = family.split('_')[1]
        options = [dict(family=family, candidate=0, pilot=True, loss=loss,
                        ridge=0., l1=0., floor=0., valid=True, reason='',
                        **{s: float(np.mean([v[s] for v in pilot_scores])) for s in ('riesz','action_ce')})]
        for ridge in old.RIDGES:
            for l1 in old.L1_VALUES if family.endswith('sparse') else (0.,):
                for floor in old.FLOORS:
                    key = (loss, ridge, l1, floor)
                    if key not in cache:
                        scores=[]; reasons=[]
                        for fold in range(2):
                            entry = paths[fold][loss,ridge,l1]
                            if not entry['valid']:
                                reasons.extend(entry['reasons']); continue
                            try:
                                model=old.SharedLogitModel(pbar,entry['entries'],dict(pilot=False,floor=floor))
                                scores.append(observed_scores(model,artifacts[1-fold],queries,selection,core))
                            except (ValueError,RuntimeError,FloatingPointError) as e:
                                reasons.append(type(e).__name__)
                        cache[key] = (scores, reasons)
                    scores,reasons=cache[key]
                    valid=len(scores)==2 and not reasons
                    options.append(dict(family=family,candidate=len(options),pilot=False,loss=loss,
                        ridge=ridge,l1=l1,floor=floor,valid=valid,reason=';'.join(reasons),
                        **{s: float(np.mean([v[s] for v in scores])) if valid else float('nan')
                           for s in ('riesz','action_ce')}))
        for rule in ('riesz','action_ce'):
            choices[family+'__'+rule] = dict(choose_records(options,rule),selector=rule)
        records.extend(options)
    needed={(c['loss'],c['ridge'],c['l1']) for c in choices.values() if not c['pilot']}
    say('Refitting CE/CAL choices after both selectors are frozen.')
    final, logs, _ = old._fit_paths(artifacts[2], core, needed, 'outer_refit')
    fit_logs.extend(logs)
    models={}; failures={}
    for name,c in choices.items():
        if c['pilot']:
            models[name]=pilot
        elif final[c['loss'],c['ridge'],c['l1']]['valid']:
            models[name]=old.SharedLogitModel(pbar,final[c['loss'],c['ridge'],c['l1']]['entries'],dict(c))
        else:
            failures[name]='Selected outer refit failed; no reselection.'
    if include_tree:
        say('Fitting standard histogram-boosted propensity benchmark.')
        family='shared_hgb'
        options=[dict(family=family,candidate=0,pilot=True,floor=0.,leaves=0,min_leaf=0,valid=True,reason='',
                      **{s: float(np.mean([v[s] for v in pilot_scores])) for s in ('riesz','action_ce')})]
        specs=[dict(leaves=leaves,min_leaf=minleaf) for leaves,minleaf in ((7,30),(7,80),(15,30),(15,80))]
        for spec in specs:
            fitted=[]; reasons=[]
            for f in range(2):
                try:
                    fitted.append(tree_fit(artifacts[f],core,spec,seed+f))
                except (ValueError,RuntimeError,FloatingPointError) as e:
                    fitted.append(None); reasons.append(type(e).__name__)
            for floor in old.FLOORS:
                ss=[]
                if not reasons:
                    for f in range(2):
                        model=TreeProbability(fitted[f],floor,pbar)
                        ss.append(observed_scores(model,artifacts[1-f],queries,selection,core))
                valid=len(ss)==2
                options.append(dict(family=family,candidate=len(options),pilot=False,floor=floor,**spec,
                    valid=valid,reason=';'.join(reasons),**{s:float(np.mean([z[s] for z in ss])) if valid else float('nan')
                                                      for s in ('riesz','action_ce')}))
        tree_cache={}
        for rule in ('riesz','action_ce'):
            name=family+'__'+rule
            c=dict(choose_records(options,rule),selector=rule); choices[name]=c
            try:
                if c['pilot']:
                    models[name]=TreeProbability({},0.,pbar,True)
                else:
                    key=(c['leaves'],c['min_leaf'])
                    if key not in tree_cache:
                        tree_cache[key]=tree_fit(artifacts[2],core,c,seed+999)
                    models[name]=TreeProbability(tree_cache[key],c['floor'],pbar)
            except (ValueError,RuntimeError,FloatingPointError):
                failures[name]='Selected tree outer refit failed; no reselection.'
        records.extend(options)
    return dict(models=models,choices=choices,candidates=records,fit_logs=fit_logs,failures=failures)
