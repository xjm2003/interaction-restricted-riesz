"""Coherent CE/CAL probabilities shared across queries, separately by landmark.

Only observed X/actions/steps, known query definitions and the original
marginal pilot are used. No conditional truth or oracle risk is an argument.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import softmax
from frozen import logit_solver as solver
import sparse_model

RIDGES=(.003,.03,.3)
L1_VALUES=(0.,.01,.05)
FLOORS=(.01,.02)
FAMILIES=('shared_ce_linear','shared_ce_sparse','shared_cal_linear','shared_cal_sparse')


def config():
    return dict(families=list(FAMILIES),reference_action=0,free_logits_per_step=3,
        time_steps_fitted_separately=True,query_sharing=True,time_sharing=False,
        features='step_and_fold_fitted_standardized_nonconstant_original_basis_plus_intercept',
        added_nonlinear_features=False,feature_cap=None,training_query_weight=False,
        objective='mean_within_landmark_unweighted_CE_or_published_CAL_divided_by_four',
        CAL_published_loss_multiplier=.25,ridges=list(RIDGES),l1=list(L1_VALUES),
        linear_l1=0.,sparse_l1=list(L1_VALUES),penalty='0.5*ridge*sum(slopes**2)+l1*sum(abs(slopes))',
        intercept_penalty=False,reference_penalty_label_invariant=False,
        floors=list(FLOORS),gamma=None,inverse_component_projection=False,
        probability_rule='eps+(1-4*eps)*softmax([0,Phi@B])',
        explicit_pilot='original_known_pbar_by_landmark_no_extra_smoothing',
        selection='equal_mean_of_two_observation_fold_raw_Riesz_losses_on_selection_queries',
        hyperparameters='one_selected_configuration_per_family_across_all_landmarks',
        candidate_counts={f:19 if f.endswith('sparse') else 7 for f in FAMILIES},
        solver_kkt_absolute_tolerance=solver.KKT_TOLERANCE,max_iterations=solver.MAX_ITERATIONS,
        solver_polish='at_most_one_same_objective_restart',nonfinite_exponential_trials='rejected_not_clipped',
        failed_candidates='recorded_and_ineligible',final_failure='explicit_method_failure_no_silent_fallback')


@dataclass
class LinearMap:
    input_dimension:int
    kept:np.ndarray
    mean:np.ndarray
    scale:np.ndarray

    @classmethod
    def fit(cls,X):
        X=sparse_model._x(X)
        mean,scale,constant,_=sparse_model._scaling(X)
        kept=np.flatnonzero(~constant)
        return cls(X.shape[1],kept,mean[kept],scale[kept])

    def transform(self,X):
        X=sparse_model._x(X)
        if X.shape[1]!=self.input_dimension: raise ValueError('Logit input basis dimension changed.')
        result=np.column_stack([np.ones(len(X)),(X[:,self.kept]-self.mean)/self.scale])
        if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite logit features.')
        return result


@dataclass
class SharedLogitModel:
    pbar:np.ndarray
    entries:dict
    choice:dict

    def predict_probabilities(self,X,steps,unbounded=False):
        X=sparse_model._x(X); steps=np.asarray(steps)
        if steps.shape!=(len(X),) or not np.isin(steps,np.arange(len(self.pbar))).all():
            raise ValueError('Invalid logit prediction landmark.')
        pi=np.empty((len(X),4))
        for t in np.unique(steps):
            mask=steps==t
            if self.choice['pilot']:
                pi[mask]=self.pbar[int(t)]
            else:
                geometry,beta=self.entries[int(t)]
                Phi=geometry.transform(X[mask])
                q=softmax(np.column_stack([np.zeros(mask.sum()),Phi@beta]),axis=1)
                eps=float(self.choice['floor'])
                pi[mask]=q if unbounded else eps+(1.-4.*eps)*q
        if not np.isfinite(pi).all() or not np.allclose(pi.sum(axis=1),1.,rtol=0,atol=1e-8 if self.choice['pilot'] else 2e-14):
            raise FloatingPointError('Invalid joint probability vector.')
        if not unbounded and np.any(pi < .01-1e-10):
            raise FloatingPointError('Final coherent probabilities violate fixed lower bound.')
        return pi

    def predict(self,X,steps,query_indices):
        inverse=1./self.predict_probabilities(X,steps)
        return np.broadcast_to(inverse[:,None,:],(len(X),len(query_indices),4))


def _fit_paths(artifact,core,pairs,phase):
    paths={key:dict(entries={},valid=True,reasons=[],certificates=[]) for key in pairs}
    diagnostics=[]; dimensions=[]
    for t in range(core.N_STEPS):
        mask=np.asarray(artifact['steps'])==t
        if not mask.any():
            for key in pairs:
                paths[key]['valid']=False; paths[key]['reasons'].append(f'No observed rows for step {t}.')
            continue
        X=np.asarray(artifact['x'])[mask]; A=np.asarray(artifact['actions'],dtype=int)[mask]
        geometry=LinearMap.fit(X); Phi=geometry.transform(X)
        dimensions.append(dict(phase=phase,step=int(t),input_dimension=int(X.shape[1]),
                               dimension=int(Phi.shape[1]),within_step_n=int(len(X))))
        for loss in ('ce','cal'):
            initial=None
            for key in sorted([k for k in pairs if k[0]==loss],key=lambda k:(-k[1],-k[2])):
                _,ridge,l1=key; diag=None; reason=''
                try:
                    B,diag=solver.fit_coefficients(Phi,A,loss,ridge,l1,initial)
                    paths[key]['entries'][t]=(geometry,B); paths[key]['certificates'].append(diag)
                    initial=B
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                    reason=f'{type(error).__name__}: {error}'
                    paths[key]['valid']=False; paths[key]['reasons'].append(f'step={t}: {reason}')
                diagnostics.append(dict(phase=phase,sharing='joint_logits',step=int(t),action=None,
                    loss=loss,ridge=float(ridge),l1=float(l1),within_step_n=int(len(X)),
                    minimum_action_n=int(np.bincount(A,minlength=4).min()),dimension=int(Phi.shape[1]),
                    status='invalid_numerical' if reason else 'valid',converged=not bool(reason),
                    failure_reason=reason,kkt_max=None if diag is None else diag['final_kkt_max'],
                    kkt_tolerance=solver.KKT_TOLERANCE,
                    solver_iterations=None if diag is None else diag['optimizer_iterations'],
                    nonzero_slopes=None if diag is None else diag['nonzero_slopes'],
                    rejected_nonfinite_trials=None if diag is None else diag['rejected_nonfinite_trials']))
    return paths,diagnostics,dimensions


def fit_observed(artifacts,queries,pbar,core,progress=None):
    if len(artifacts)!=3: raise ValueError('Two observed folds and full artifact required.')
    pbar=np.asarray(pbar,dtype=float)
    if pbar.shape!=(core.N_STEPS,4) or not np.isfinite(pbar).all() or np.any(pbar<.01-1e-10) or not np.allclose(pbar.sum(axis=1),1.,atol=1e-8,rtol=0):
        raise ValueError('Invalid original known marginal pilot probabilities.')
    for a in artifacts:
        X=sparse_model._x(a['x'])
        if len(X)!=a['n'] or np.asarray(a['actions']).shape!=(len(X),) or np.asarray(a['steps']).shape!=(len(X),):
            raise ValueError('Invalid observed logit artifact.')
        if not np.isin(a['actions'],range(4)).all() or not np.isin(a['steps'],range(core.N_STEPS)).all():
            raise ValueError('Invalid observed action or landmark.')
    selection=np.asarray(queries['indices']['selection'],dtype=int)
    if not len(selection): raise ValueError('Nonempty selection queries required.')
    say=progress or (lambda message:None)
    pairs={(loss,r,l) for loss in ('ce','cal') for r in RIDGES for l in L1_VALUES}
    folds=[]; diagnostics=[]; dimensions=[]; pilot_losses=[]
    pilot_model=SharedLogitModel(pbar,{},dict(pilot=True,floor=0.))
    for fold in range(2):
        train,val=artifacts[fold],artifacts[1-fold]
        say(f'Joint CE/CAL fold {fold+1}/2: {train["n"]} draws; {core.N_STEPS} separate landmark fits.')
        paths,logs,dims=_fit_paths(train,core,pairs,f'validation_fit_{fold}')
        folds.append(paths); diagnostics.extend(logs); dimensions.extend(dims)
        pilot_losses.append(sparse_model.observed_loss(pilot_model.predict(val['x'],val['steps'],selection),val,queries,selection,core.P_LANDMARK))
    if not np.isfinite(pilot_losses).all(): raise FloatingPointError('Nonfinite observed pilot score.')
    score_cache={}; choices={}; rows=[]
    for family in FAMILIES:
        loss=family.split('_')[1]
        candidates=[dict(method_family=family,pilot=True,loss=loss,ridge=0.,l1=0.,floor=0.,
            candidate_index=0,validation_loss=float(np.mean(pilot_losses)),
            fold_0_loss=float(pilot_losses[0]),fold_1_loss=float(pilot_losses[1]),status='valid',failure_reason='')]
        for ridge in RIDGES:
            for l1 in L1_VALUES if family.endswith('sparse') else (0.,):
                for floor in FLOORS:
                    key=(loss,ridge,l1,floor)
                    if key not in score_cache:
                        scores=[]; reasons=[]
                        for fold in range(2):
                            entry=folds[fold][loss,ridge,l1]; val=artifacts[1-fold]
                            if not entry['valid']:
                                scores.append(None); reasons.extend(f'fold={fold}: {r}' for r in entry['reasons']); continue
                            try:
                                model=SharedLogitModel(pbar,entry['entries'],dict(pilot=False,floor=floor))
                                value=sparse_model.observed_loss(model.predict(val['x'],val['steps'],selection),val,queries,selection,core.P_LANDMARK)
                                if not np.isfinite(value): raise FloatingPointError('Nonfinite observed Riesz validation.')
                                scores.append(float(value))
                            except (ValueError,RuntimeError,FloatingPointError) as error:
                                scores.append(None); reasons.append(f'fold={fold}: {type(error).__name__}: {error}')
                        score_cache[key]=(scores,'; '.join(reasons))
                    scores,reason=score_cache[key]
                    candidates.append(dict(method_family=family,pilot=False,loss=loss,ridge=ridge,l1=l1,floor=floor,
                        candidate_index=len(candidates),fold_0_loss=scores[0],fold_1_loss=scores[1],
                        validation_loss=None if reason else float(np.mean(scores)),
                        status='invalid_numerical' if reason else 'valid',failure_reason=reason))
        best=candidates[0]
        for candidate in candidates[1:]:
            if candidate['status']=='valid' and candidate['validation_loss']<best['validation_loss']-1e-12:
                best=candidate
        choices[family]=dict(best); rows.extend(candidates)
    say('Four CE/CAL configurations selected using observed validation; refitting on all observed rows.')
    required={(c['loss'],c['ridge'],c['l1']) for c in choices.values() if not c['pilot']}
    final_paths={}
    if required:
        final_paths,logs,dims=_fit_paths(artifacts[2],core,required,'final')
        diagnostics.extend(logs); dimensions.extend(dims)
    models={}; failures={}; certs={}
    for family,c in choices.items():
        entry=None if c['pilot'] else final_paths[c['loss'],c['ridge'],c['l1']]
        if entry is not None and not entry['valid']:
            failures[family]='; '.join(entry['reasons']); certs[family]=dict(pilot=False,all_kkt_certified=False,status='failed')
            continue
        models[family]=SharedLogitModel(pbar,{} if entry is None else entry['entries'],dict(c))
        certificates=[] if entry is None else entry['certificates']
        certs[family]=dict(status='valid',pilot=bool(c['pilot']),solver_landmarks=len(certificates),
            all_kkt_certified=all(x['final_kkt_max']<=solver.KKT_TOLERANCE for x in certificates),
            maximum_kkt=float(max((x['final_kkt_max'] for x in certificates),default=0.)))
    return dict(models=models,choices=choices,method_failures=failures,
        selection_table=pd.DataFrame(rows),fit_diagnostics=pd.DataFrame(diagnostics),
        checks=dict(logit_candidate_counts={f:sum(r['method_family']==f for r in rows) for f in FAMILIES},
                    logit_invalid_candidate_records=sum(r['status']!='valid' for r in rows),
                    logit_feature_dimensions=dimensions,logit_final_certificates_by_method=certs,
                    logit_final_failures=failures,logit_training_query_weight=False,
                    logit_explicit_original_pilot=True,logit_final_inverse_component_projection=False,
                    logit_separate_landmarks=True,logit_reference_action=0))
