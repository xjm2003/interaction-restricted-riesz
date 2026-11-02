#!/usr/bin/env python3
"""Three external comparators on the CVD/PHQ-9 benchmark.
No CRDM connection. No original result is overwritten. No clinical Y is read.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ext_support import (ROOT, ENGINE, IntegrityError, say, read_json, write_json,
    write_frame, runtime, import_engine, find_reference, read_reference, reference_seed)
from ext_models import fit_probability,fit_riesz_query,FitFailure

METHODS={'logit':'aipw_multinomial_logit','hgb':'aipw_hist_gradient_boosting','rieszboost':'rieszboost_four_arm'}
LEARNERS=('linear','flexible')


def arguments(dataset=None,argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    if dataset is None:p.add_argument('--dataset',choices=['cvd','phq9'],required=True)
    p.add_argument('--reference',help='Read-only archived original n=6000 validation result directory.')
    p.add_argument('--input',help='Original CVD zip or CSV; default unchanged.')
    p.add_argument('--member',help='Original CVD zip member, when needed.')
    p.add_argument('--phq-root',help='Original qaoct_semisynthetic_v1 root, NOT the new clinical index cohort.')
    p.add_argument('--output')
    p.add_argument('--max-seeds',type=int,help='Run the first N seeds from scratch; reports retain the full planned denominator.')
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS),help='Scheduling only. All three methods remain in the fixed plan; default all.')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--check-inputs',action='store_true',help='Check input files, reference result tables and seed availability; no fitting.')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit FICTIONAL software test: reduced grids, 4 queries, 1 seed; never mixed with real runs.')
    a=p.parse_args(argv)
    if dataset:a.dataset=dataset
    if a.threads<1 or (a.max_seeds is not None and a.max_seeds<1):p.error('Positive threads and max-seeds required.')
    a.methods=list(dict.fromkeys(a.methods))
    return a


def effective_plan(smoke,dataset):
    plan=read_json(ROOT/'external_plan.json')
    if not smoke:return plan,6000,list(range(plan['seed_start'][dataset],plan['seed_start'][dataset]+plan['replications']))
    plan=copy.deepcopy(plan);plan['id']='FICTIONAL_SOFTWARE_TEST_external_v1'
    plan['probability_floor_grid']=[.01];plan['logit']['C']=[.1,1.]
    plan['hgb'].update(learning_rate=[.1],max_leaf_nodes=[7],iterations=[5,10],min_samples_leaf=10)
    plan['rieszboost'].update(learning_rate=[.25],max_depth=[2],iterations=[5,10],min_samples_leaf=10)
    return plan,(1200 if dataset=='cvd' else 2400),[119901]


def prepare_sample(engine,data,mu,n,seed,dataset):
    sample=engine.data_bridge.draw_sample(data,n,seed,'patient' if dataset=='phq9' else 'iid_row')
    yrng=np.random.default_rng(np.random.SeedSequence([seed,n,923]))
    y=mu[sample['rows'],sample['actions']]+yrng.normal(0,1.,n)
    # Never pass sample['pbar'] or sample['p_true'] to a new weight learner.
    observed={k:sample[k] for k in ('rows','steps','actions','x','clusters')}
    folds=engine.data_bridge.group_folds(observed['clusters'],3,seed+747)
    return observed,y,folds


def _summary(engine,scores,clusters):
    sm,ss,nu=engine.profile_summaries(scores,clusters)
    return dict(sum_score=sm.tolist(),sumsq_score=ss.tolist(),units=int(nu))


def _score_prediction(engine,scales,alpha,mu,a,y,clusters):
    return _summary(engine,engine.orthogonal_scores(scales,alpha,mu,a,y),clusters)





def _record_failure(error):
    return dict(status='failed', error_type=type(error).__name__, message=str(error))


def process_fold(engine, data, observed, y, train, test, qids, plan, seed, f, scheduled):
    results = {}
    x = observed['x']; st = observed['steps']; a = observed['actions']; clusters = observed['clusters']
    inner = engine.data_bridge.group_folds(clusters[train], 2, seed + f + 172)
    say(f'seed={seed} fold={f+1}/3: refit the two outcome learners once')
    tick = time.monotonic()
    try:
        outcome_models = engine.fit_outcomes(x[train], st[train], a[train], y[train], data['core'].N_STEPS, seed + f)
        mus = {l: engine.predict_outcome(outcome_models[l], x[test], st[test], data['core'].N_STEPS) for l in LEARNERS}
    except (ValueError, FitFailure, ArithmeticError, np.linalg.LinAlgError) as error:
        return {method: _record_failure(error) for method in scheduled}
    outcome_seconds = time.monotonic() - tick
    scales = engine.get_scales(data, qids, observed['rows'][test])
    for method in scheduled:
        tick = time.monotonic()
        try:
            if method in ('logit', 'hgb'):
                say(f'seed={seed} fold={f+1}/3: {method} inner validation + selected refit')
                prob, selection = fit_probability(method, x[train], st[train], a[train], inner, plan, seed + f + 36001)
                p = prob.predict(x[test], st[test]); alpha = scales / p[:, None, :]
                stats = {l: _score_prediction(engine, scales, alpha, mus[l], a[test], y[test], clusters[test]) for l in LEARNERS}
                diagnostic = dict(min_fitted_p=float(p.min()), max_abs_alpha=float(np.abs(alpha).max()),
                    mean_alpha2=float(np.mean(alpha[np.arange(len(test))[:,None], np.arange(len(qids))[None,:], a[test,None]]**2)))
            else:
                train_scales = engine.get_scales(data, qids, observed['rows'][train])
                parts = []; selection = []; query_failures = []; maxabs = 0.; alpha2 = 0.
                for j, qi in enumerate(qids):
                    say(f'seed={seed} fold={f+1}/3: RieszBoost query {j+1}/{len(qids)}')
                    qt = time.monotonic()
                    try:
                        alph, sel = fit_riesz_query(x[train], st[train], a[train], train_scales[:,j,:], inner,
                            plan['rieszboost'], seed+f+70001+int(qi)*103, x[test], st[test])
                        qstats = {l: _score_prediction(engine, scales[:,j:j+1,:], alph[:,None,:], mus[l],
                            a[test], y[test], clusters[test]) for l in LEARNERS}
                        record = dict(stats=qstats, selection=sel, query_index=int(qi),
                            fit_seconds=time.monotonic()-qt, max_abs_alpha=float(np.abs(alph).max()),
                            mean_alpha2=float(np.mean(alph[np.arange(len(test)),a[test]]**2)))
                    except (ValueError, FitFailure, ArithmeticError, np.linalg.LinAlgError):
                        query_failures.append(int(qi))
                        continue
                    parts.append(record)
                    selection.extend(dict(query_index=int(qi), **row) for row in record['selection'])
                    maxabs = max(maxabs, record['max_abs_alpha'])
                    alpha2 += record['mean_alpha2'] / len(qids)
                if query_failures:
                    raise FitFailure(f'RieszBoost unavailable for {len(query_failures)} planned queries; no query deletion or zero replacement.')
                stats = {l: dict(sum_score=[r['stats'][l]['sum_score'][0] for r in parts],
                    sumsq_score=[r['stats'][l]['sumsq_score'][0] for r in parts],
                    units=parts[0]['stats'][l]['units']) for l in LEARNERS}
                diagnostic = dict(max_abs_alpha=maxabs, mean_alpha2=alpha2)
            results[method] = dict(status='complete', stats=stats, selection=selection, diagnostic=diagnostic,
                fit_seconds=time.monotonic()-tick, outcome_refit_seconds=outcome_seconds,
                training_rows=len(train), held_rows=len(test), held_units=int(len(np.unique(clusters[test]))))
            say(f'seed={seed} fold={f+1}/3: {method} complete')
        except (ValueError, FitFailure, ArithmeticError, np.linalg.LinAlgError) as error:
            results[method] = _record_failure(error)
            say(f'seed={seed} fold={f+1}/3: {method} FAILED ({type(error).__name__}); other methods continue')
    return results


def replay_oracle(engine, data, observed, y, folds, mu, probabilities, qids, truth, reference):
    parts = []
    for test in folds:
        test = np.sort(test)
        rows = observed['rows'][test]
        scales = engine.get_scales(data, qids, rows)
        scores = engine.orthogonal_scores(scales, scales / probabilities[rows, None, :],
                    mu[rows], observed['actions'][test], y[test])
        parts.append(engine.profile_summaries(scores, observed['clusters'][test]))
    estimate, se, units = engine.combine_profile_summaries(parts)
    result = dict(independent_units=int(units),
                  max_abs_oracle_error=float(np.max(np.abs(estimate - truth))))
    if reference is not None:
        expected = reference[reference.method.eq('oracle_AIPW')].set_index('query_index')
        if not expected.index.duplicated().any() and set(expected.index) == set(qids):
            expected = expected.loc[qids]
            for name, actual in [('estimate', estimate), ('se', se), ('true_effect', truth)]:
                result[name + '_replay_max_abs_difference'] = float(np.max(np.abs(expected[name].to_numpy(float) - actual)))
    return result


def run_one(engine,data,mu,p,truth,qids,cat,plan,args,out,seed,n,reference):
    slot=out/'runs'/f'n_{n}'/f'seed_{seed}';slot.mkdir(parents=True,exist_ok=True)
    for name in ('query_results.csv', 'method_inventory.csv', 'reference_results.csv', 'selection.json', 'alignment.json'):
        (slot / name).unlink(missing_ok=True)
    observed,y,folds=prepare_sample(engine,data,mu,n,seed,args.dataset)
    ref=None
    if reference is not None:
        ref=reference_seed(reference,n,seed,qids,truth)
    replay=replay_oracle(engine,data,observed,y,folds,mu,p,qids,truth,ref)
    if ref is not None:write_frame(slot/'reference_results.csv',ref)
    write_json(slot/'alignment.json',dict(**replay,phenotype_or_patient_rows_exported=False))
    allix=np.arange(n)
    fold_results=[]
    for f,test in enumerate(folds):
        test=np.sort(test);train=np.setdiff1d(allix,test)
        fold_results.append(process_fold(engine,data,observed,y,train,test,qids,plan,seed,f,args.methods))
    rows=[];inventory=[];selection=[]
    for m,name in METHODS.items():
        records=[]
        for f in range(3):
            r=fold_results[f].get(m)
            records.append(r)
            if r and r['status']=='complete':
                selection.extend([dict(seed=seed,fold=f,method=name,**s) for s in r['selection']])
        state='complete' if all(r and r['status']=='complete' for r in records) else ('failed' if any(r and r['status']=='failed' for r in records) else 'not_processed')
        inventory.append(dict(dataset=args.dataset,n_rows=n,seed=seed,method=name,status=state,
                              completed_folds=sum(r is not None and r['status']=='complete' for r in records)))
        if state!='complete':continue
        for learner in LEARNERS:
            parts=[(np.asarray(r['stats'][learner]['sum_score']),np.asarray(r['stats'][learner]['sumsq_score']),r['stats'][learner]['units']) for r in records]
            estimate,se,nu=engine.combine_profile_summaries(parts)
            for j,qi in enumerate(qids):
                rows.append(dict(dataset=args.dataset,n_rows=n,independent_units=nu,seed=seed,
                    learner=learner,probability_source={'logit':'multinomial_logit','hgb':'hist_gradient_boosting','rieszboost':'direct_rieszboost'}[m],
                    method=name,stage='EXTERNAL',structure='unrestricted',query_index=int(qi),
                    query_id=cat[int(qi)]['query_id'],contrast_family=cat[int(qi)]['contrast_family'],query_split='held_out',
                    estimate=float(estimate[j]),se=float(se[j]),ci_lower=float(estimate[j]-engine.Z975*se[j]),
                    ci_upper=float(estimate[j]+engine.Z975*se[j]),true_effect=float(truth[j]),
                    origin='new_external',weight_training_uses_Y=False,uses_known_marginal_pilot=False,
                    intervals='profile_score_Wald_coverage_evaluated_not_guaranteed'))
    fields=['dataset','n_rows','independent_units','seed','learner','probability_source','method','stage','structure','query_index','query_id','contrast_family','query_split','estimate','se','ci_lower','ci_upper','true_effect','origin']
    write_frame(slot/'query_results.csv',pd.DataFrame(rows) if rows else pd.DataFrame(columns=fields))
    write_frame(slot/'method_inventory.csv',pd.DataFrame(inventory))
    write_json(slot/'selection.json',selection)
    status={r['method']:r['status'] for r in inventory}
    say(f'seed={seed}: '+', '.join(k+'='+v for k,v in status.items()))
    return status


def main(dataset=None,argv=None):
    args=arguments(dataset,argv)
    out=Path(args.output).expanduser().resolve() if args.output else ROOT/(f'software_smoke_{args.dataset}' if args.smoke else f'results_{args.dataset}_external')
    if out==ROOT or out.is_relative_to(ENGINE):raise IntegrityError('Output cannot overwrite the package or frozen engine.')
    if args.report_only:
        from ext_report import summarize,display
        summarize(out);display(out);return
    plan,n,seeds=effective_plan(args.smoke,args.dataset)
    engine=import_engine()
    with threadpool_limits(limits=args.threads):
        say('Loading the archived semi-synthetic covariate design; no CRDM connection and no real outcome extraction.')
        data,steps,p,pbar,phi,feature_info=engine.load_design(args.dataset,args.input,args.member,args.phq_root,args.smoke,patients=320)
        allcat=engine.catalog(data);qids=np.asarray(data['queries']['indices']['held_out'],int)
        if args.smoke:qids=qids[:4]
        cat={r['query_index']:r for r in allcat}
        mu,law=engine.data_bridge.build_outcome_law(data,1.)
        alltruth=engine.data_bridge.finite_population_targets(data,mu)
        held=np.asarray(data['queries']['indices']['held_out'],int)
        truth=alltruth[[int(np.flatnonzero(held==q)[0]) for q in qids]]
        reference=None if args.smoke and not args.reference else find_reference(args.dataset,args.reference)
        if reference:
            if out.is_relative_to(reference) or reference.is_relative_to(out):raise IntegrityError('Output and archived reference directories must be separate and nonnested.')
            read_reference(reference,args.dataset,n,seeds)
        if args.check_inputs:
            say(json.dumps(dict(dataset=args.dataset,source_profiles=int(len(steps)/data['core'].N_STEPS),
                simulated_rows=n,independent_units=n//data['core'].N_STEPS,planned_repeats=len(seeds),held_queries=len(qids),
                original_result_directory=str(reference) if reference else None,runtime=runtime(),fixture=args.smoke),indent=2))
            if reference:
                missing=[s for s in seeds if not (reference/'runs'/f'n_{n}'/f'seed_{s}'/'query_results.csv').is_file()]
                if missing:raise IntegrityError(f'{len(missing)} reference seeds lack query results.')
            return
        cfg=dict(version='external-baselines-v1',dataset=args.dataset,n_rows=n,seeds=seeds,fixture=args.smoke,
            plan=plan,threads=args.threads,runtime=runtime(),held_query_indices=qids.tolist(),
            reference_directory=str(reference) if reference else None)
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'manifest.json',dict(config=cfg))
        write_frame(out/'query_catalog.csv',pd.DataFrame([cat[int(q)] for q in qids]))
        write_json(out/'DGP_provenance.json',law)
        say(f'{"FICTIONAL SOFTWARE TEST" if args.smoke else "SUPPLEMENTARY EXTERNAL COMPARISON"}: {len(seeds)} planned seeds, {len(qids)} queries; methods: {args.methods}')
        for seed in seeds[:args.max_seeds]:
            run_one(engine,data,mu,p,truth,qids,cat,plan,args,out,seed,n,reference)
            from ext_report import summarize
            summarize(out)
        from ext_report import summarize,display
        summarize(out);display(out,brief=True)
        say('Output: '+str(out))

def entrypoint(dataset=None):
    try:main(dataset)
    except KeyboardInterrupt:
        print('\nInterrupted. Finished seed result tables remain available; the next run fits again from scratch.',flush=True);sys.exit(130)
    except (IntegrityError,ValueError,FileNotFoundError) as e:
        print('STOP: '+str(e),flush=True);sys.exit(2)


if __name__=='__main__':
    entrypoint()
