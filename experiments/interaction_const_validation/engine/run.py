#!/usr/bin/env python3
"""Matched A/B/C interaction projection on existing real-covariate benchmarks.
No clinical Y is read. No new outcome, propensity, or structure selector is fitted.
"""
from __future__ import annotations
import bootstrap
from pathlib import Path
from dataclasses import dataclass, field
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from io_utils import write_json, write_csv
from data_design import load_design, catalog
import data_bridge
import baseline_experiment as base
import legacy_run
from outcomes import fit_outcomes, predict_outcome, orthogonal_scores
from projection import (K, STRUCTURES, WeightedProjector, query_ell,
                        project_outcome, project_representer, weight_j)
from score_utils import profile_summaries, combine_profile_summaries, Z975

ROOT = Path(__file__).resolve().parent
VERSION = 'interaction-projection-abc-v1.0'
CAL_NAMES = {'cal_linear': 'shared_cal_linear__action_ce',
             'cal_sparse': 'shared_cal_sparse__action_ce'}
LEGACY_NAMES = ('regularized_B1', 'shared_sparse_riesz')
LEARNERS = ('linear', 'flexible')
DIAG_NAMES = ('representer_mse', 'outcome_mse', 'remainder', 'alpha_second_moment',
              'max_abs_alpha_eval', 'outcome_change_mse', 'interaction_error_j',
              'interaction_error_raw', 'low_probability_alpha_risk')


def say(text):
    print(time.strftime('%H:%M:%S') + ' ' + str(text), flush=True)


def get_scales(data, qids, rows):
    rows = np.asarray(rows, int)
    st = data['population']['step'].to_numpy(int)[rows]
    return base.query_scale(data['queries'], qids, rows, st, data['core'].P_LANDMARK)


@dataclass
class FoldFits:
    models: dict
    outcomes: dict
    geometries: dict
    outcome_coordinates: dict
    query_coordinates: dict
    choices: list
    candidates: list
    projection_checks: list
    gram_diagnostics: list
    legacy_failures: dict = field(default_factory=dict)


def fit_fold(data, raw_phi, observed, y, train, qids, seed, fold):
    """All new projection targets are fitted predictions; no oracle inputs.

    observed must not contain p_true or mu_true. The unchanged archived adapters
    provide the previously allowed marginal pilot and X-defined queries.
    """
    core = data['core']
    artifacts = data_bridge.observed_artifacts(observed, train, data['queries'], core,
                                               observed['pbar'], seed + fold + 172)
    say(f'fold {fold+1}: original CAL/B1/shared-sparse candidates (unchanged grids)')
    models, choices, candidates, failures = legacy_run.fit_library(
        artifacts, data['queries'], observed['pbar'], core, seed + fold, False)
    required = tuple(CAL_NAMES.values())
    missing = [name for name in required if name not in models]
    if missing:
        raise RuntimeError('Required original fit failed; no reselection: ' + ', '.join(missing))
    legacy_failures = {name: 'fit: ' + str(failures.get(name, 'Selected fit unavailable.'))
                       for name in LEGACY_NAMES if name not in models or name in failures}
    models = {name: models[name] for name in required + LEGACY_NAMES if name in models}
    say(f'fold {fold+1}: fit original linear/flexible outcome learners ONCE')
    outcome_models = fit_outcomes(observed['x'][train], observed['steps'][train],
          observed['actions'][train], y[train], core.N_STEPS, seed + fold)
    source = observed['rows'][train]
    xx = observed['x'][train]
    st = observed['steps'][train]
    phi = raw_phi[source]
    scales = get_scales(data, qids, source)
    mus = {l: predict_outcome(outcome_models[l], xx, st, core.N_STEPS) for l in LEARNERS}
    geometries, outcoef, qcoef, checks, diags = {}, {}, {}, [], []
    for source_name, name in CAL_NAMES.items():
        pp = models[name].obj.predict_probabilities(xx, st)
        ell = query_ell(scales, pp)
        for structure in STRUCTURES:
            geo = WeightedProjector.fit(phi, st, pp, structure, core.N_STEPS)
            geometries[source_name, structure] = geo
            qcoef[source_name, structure] = geo.fit_coordinates(ell)
            for row in geo.diagnostics:
                diags.append(dict(fold=fold, probability_source=source_name, **row))
            qr = geo.training_residual_summary(qcoef[source_name, structure], ell)
            for learner in LEARNERS:
                tau = mus[learner] @ K
                coeff = geo.fit_coordinates(tau)
                outcoef[source_name, learner, structure] = coeff
                fitted_tau = geo.predict(coeff, phi, st)[:, 0]
                projected = project_outcome(mus[learner], pp, fitted_tau)
                mr = geo.training_residual_summary(coeff, tau)
                err = float(np.max(np.abs(projected @ K - fitted_tau)))
                checks.append(dict(fold=fold, probability_source=source_name, learner=learner,
                    structure=structure, max_train_mu_change=float(np.max(np.abs(projected-mus[learner]))),
                    train_change_mse=float(np.mean(np.sum(pp*(projected-mus[learner])**2, axis=1))),
                    max_interaction_constraint_error=err,
                    interaction_weighted_residual_mse=mr['weighted_residual_mse'],
                    interaction_normal_moment=mr['max_normal_moment'],
                    query_normal_moment=qr['max_normal_moment'],
                    expected_identity=(learner=='linear' and structure in ('CONST','LOWDIM')),
                    projection_uses_train_refit_predictions=True,
                    outcome_coefficients_refitted_for_B_or_C=False))
        say(f'fold {fold+1}: {source_name} projections frozen; no extra response learner fitted')
    return FoldFits(models, outcome_models, geometries, outcoef, qcoef,
                    choices, candidates, checks, diags, legacy_failures)


def method_specs(with_oracle=True):
    specs = []
    for learner in LEARNERS:
        for source in CAL_NAMES:
            specs.append((learner, source, 'A', 'UNR', 'A_full'))
            for m in STRUCTURES:
                specs.append((learner, source, 'B', m, 'B_'+m))
                specs.append((learner, source, 'C', m, 'C_'+m))
        for name in LEGACY_NAMES:
            specs.append((learner, 'legacy', 'REF', 'legacy', name))
    if with_oracle:
        specs.append(('oracle', 'oracle', 'ORACLE', 'oracle', 'oracle_AIPW'))
    return specs


def predictions(data, phi_bank, rows, scales, qids, fits, mu_true=None, p_true=None):
    """All routes share the same base predictions and probabilities in this batch."""
    st = data['population']['step'].to_numpy(int)[rows]
    xx = np.asarray(data['x_population'][rows], float)
    phi = phi_bank[rows]
    pp = {nick: fits.models[name].obj.predict_probabilities(xx, st)
          for nick, name in CAL_NAMES.items()}
    # C's coefficients and geometry are independent of the outcome learner.
    restricted_alpha = {}
    for source in CAL_NAMES:
        for structure in STRUCTURES:
            geo = fits.geometries[source, structure]
            gg = geo.predict(fits.query_coordinates[source, structure], phi, st)
            restricted_alpha[source, structure] = project_representer(scales, pp[source], gg)
    for learner in LEARNERS:
        raw = predict_outcome(fits.outcomes[learner], xx, st, data['core'].N_STEPS)
        for source in CAL_NAMES:
            full = scales / pp[source][:, None, :]
            yield (learner, source, 'A', 'UNR', 'A_full'), raw, full, raw
            for structure in STRUCTURES:
                geo = fits.geometries[source, structure]
                ti = geo.predict(fits.outcome_coordinates[source, learner, structure], phi, st)[:, 0]
                projected = project_outcome(raw, pp[source], ti)
                # B and C get the exact same ndarray of outcome predictions.
                yield (learner, source, 'B', structure, 'B_'+structure), projected, full, raw
                yield (learner, source, 'C', structure, 'C_'+structure), projected, restricted_alpha[source, structure], raw
        for name in LEGACY_NAMES:
            if name in fits.legacy_failures:
                continue
            if name not in fits.models:
                fits.legacy_failures[name] = 'fit: Selected fit unavailable.'
                continue
            try:
                alpha = np.asarray(fits.models[name].predict(xx, st, scales, qids), float)
                if alpha.shape != scales.shape or not np.isfinite(alpha).all():
                    raise ValueError('Invalid optional baseline prediction shape or nonfinite weights.')
            except Exception as error:
                fits.legacy_failures[name] = f'prediction: {type(error).__name__}: {error}'
                continue
            yield (learner, 'legacy', 'REF', 'legacy', name), raw, alpha, raw
    if mu_true is not None and p_true is not None:
        raw = mu_true[rows]
        yield ('oracle', 'oracle', 'ORACLE', 'oracle', 'oracle_AIPW'), raw, scales/p_true[rows,None,:], raw


def evaluate_fold(data, phi_bank, observed, y, test, qids, fits, mu_true, p_true,
                  eval_rows, with_oracle=True, chunk=512):
    specs = method_specs(with_oracle)
    index = {key:i for i,key in enumerate(specs)}
    q = len(qids)
    sm = np.zeros((len(specs), q)); sq = sm.copy(); nu = np.zeros(len(specs), int)
    di = np.zeros((len(specs), len(DIAG_NAMES), q))
    overlap = np.zeros((len(specs), 4, 4))  # mass, alpha-risk, weight-energy, g-risk contributions
    # Process whole PHQ profiles in each chunk (row order is profile-contiguous).
    T = data['core'].N_STEPS
    if T not in (1,4):
        raise ValueError('Unsupported original profile size.')
    chunk = max(T, (chunk//T)*T)
    oracle_kw = dict(mu_true=mu_true, p_true=p_true) if with_oracle else {}
    for start in range(0, len(test), chunk):
        local = test[start:start+chunk]
        rr = observed['rows'][local]
        ss = get_scales(data, qids, rr)
        for key, gg, aa, raw in predictions(data, phi_bank, rr, ss, qids, fits, **oracle_kw):
            i = index[key]
            try:
                scores = orthogonal_scores(ss, aa, gg, observed['actions'][local], y[local])
                s, s2, nn = profile_summaries(scores, observed['clusters'][local])
                sm[i] += s; sq[i] += s2; nu[i] += nn
            except Exception as error:
                if key[2] != 'REF':
                    raise
                fits.legacy_failures[key[4]] = f'score: {type(error).__name__}: {error}'
    # Independent empirical-covariate subset: diagnostics only, after all fits freeze.
    ne = len(eval_rows)
    for start in range(0, ne, chunk):
        rr = eval_rows[start:start+chunk]
        ss = get_scales(data, qids, rr)
        pt = p_true[rr]
        alpha_true = ss/pt[:,None,:]
        bins = np.searchsorted(np.array([.02,.05,.10]), pt.min(1), side='right')
        for key, gg, aa, raw in predictions(data, phi_bank, rr, ss, qids, fits, **oracle_kw):
            i = index[key]
            try:
                da = aa - alpha_true
                eg = gg - mu_true[rr]
                ar = np.sum(pt[:,None,:]*da**2, axis=2)
                ae = np.sum(pt[:,None,:]*aa**2, axis=2)
                gr = np.sum(pt*eg**2, axis=1)
                dr = -np.sum(pt[:,None,:]*da*eg[:,None,:], axis=2)
                dmu = np.sum(pt*(gg-raw)**2, axis=1)
                taue = (gg-mu_true[rr]) @ K
                di[i,0] += ar.sum(0)/ne
                di[i,1] += gr.sum()/ne
                di[i,2] += dr.sum(0)/ne
                di[i,3] += ae.sum(0)/ne
                di[i,4] = np.maximum(di[i,4], np.abs(aa).max((0,2)))
                di[i,5] += dmu.sum()/ne
                di[i,6] += np.sum(weight_j(pt)*taue**2)/ne
                di[i,7] += np.sum(taue**2)/ne
                di[i,8] += np.sum(pt[:,None,:]*da**2*(pt[:,None,:]<.05),axis=(0,2))/ne
                for b in range(4):
                    keep = bins == b
                    overlap[i,b] += np.array([keep.sum(), ar[keep].sum()/q,
                                             ae[keep].sum()/q, gr[keep].sum()])/ne
            except Exception as error:
                if key[2] != 'REF':
                    raise
                fits.legacy_failures[key[4]] = f'diagnostic: {type(error).__name__}: {error}'
    for i,key in enumerate(specs):
        if key[2] == 'REF' and not all(np.isfinite(value[i]).all()
                                      for value in (sm, sq, di, overlap)):
            fits.legacy_failures[key[4]] = 'evaluation: Nonfinite optional baseline aggregate.'
    keep = np.array([i for i,key in enumerate(specs)
                     if key[2] != 'REF' or key[4] not in fits.legacy_failures], int)
    result = dict(specs=np.asarray(specs)[keep], sum_score=sm[keep], sumsq_score=sq[keep],
                  units=nu[keep], diagnostics=di[keep], overlap=overlap[keep])
    for key, value in result.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite aggregate: '+key)
    result['legacy_failures'] = dict(fits.legacy_failures)
    return result


def run_seed(data, phi_bank, mu, truth, args, out, seed, n):
    slot = out/'runs'/f'n_{n}'/f'seed_{seed}'
    slot.mkdir(parents=True, exist_ok=True)
    # Each invocation recomputes this seed from the requested inputs.
    for filename in ('query_results.csv', 'method_inventory.csv', 'failed.json'):
        (slot/filename).unlink(missing_ok=True)
    sample = data_bridge.draw_sample(data, n, seed, 'patient' if args.dataset=='phq9' else 'iid_row')
    y_rng = np.random.default_rng(np.random.SeedSequence([seed,n,923]))
    y = mu[sample['rows'],sample['actions']] + y_rng.normal(0,1.,n)
    # Restrict observable fit arguments; truth remains in evaluator/generator.
    observed = {k:sample[k] for k in ('rows','steps','actions','x','clusters','pbar')}
    folds = data_bridge.group_folds(observed['clusters'],3,seed+747)
    qids = np.asarray(data['queries']['indices']['held_out'], int)
    N = len(mu)
    rng_eval = np.random.default_rng(np.random.SeedSequence([seed,n,193]))
    eval_rows = np.arange(N) if args.eval_rows==0 or N<=args.eval_rows else np.sort(
        rng_eval.choice(N,args.eval_rows,replace=False))
    _, p_true, _ = base.population_arrays(data)
    all_ids = np.arange(n); stats=[]
    for f,test in enumerate(folds):
        # Old fold contents/order, unchanged; sort only for profile-contiguous chunks.
        test = np.sort(test)
        train = np.setdiff1d(all_ids, test)
        fol=slot/f'fold_{f}'; fol.mkdir(exist_ok=True)
        say(f'n={n} seed={seed} fold={f+1}/3 | train={len(train)} test={len(test)} rows')
        fits = fit_fold(data,phi_bank,observed,y,train,qids,seed,f)
        write_csv(fol/'original_choices.csv',fits.choices)
        write_csv(fol/'original_candidates.csv',fits.candidates)
        write_csv(fol/'projection_checks.csv',fits.projection_checks)
        write_csv(fol/'projection_gram_diagnostics.csv',fits.gram_diagnostics)
        say(f'fold {f+1}: A/B/C frozen; evaluate original targets and diagnostic bank')
        stat = evaluate_fold(data,phi_bank,observed,y,test,qids,fits,mu,p_true,
                             eval_rows,not args.no_oracle_reference)
        stats.append(stat)
    cat={row['query_index']:row for row in catalog(data)}
    indices = [{tuple(map(str,key)):i for i,key in enumerate(st['specs'])} for st in stats]
    successful = set.intersection(*(set(idx) for idx in indices))
    inventory=[]
    for key in method_specs(not args.no_oracle_reference):
        failed_folds = [f for f,idx in enumerate(indices) if key not in idx]
        learner,source,stage,structure,method=key
        reasons = [f'fold={f}: '+stats[f]['legacy_failures'].get(method, 'Incomplete method output.')
                   for f in failed_folds]
        inventory.append(dict(dataset=args.dataset,n_rows=n,seed=seed,learner=learner,
            probability_source=source,stage=stage,structure=structure,method=method,
            status='failed' if failed_folds else 'complete',completed_folds=len(folds)-len(failed_folds),
            planned_folds=len(folds),failed_folds=';'.join(map(str,failed_folds)),
            failure_reason='; '.join(reasons)))
    write_csv(slot/'method_inventory.csv',inventory)
    rows=[]; overs=[]
    for key in method_specs(not args.no_oracle_reference):
        if key not in successful:
            continue
        learner,source,stage,structure,method=map(str,key)
        positions=[idx[key] for idx in indices]
        estimate,se,nu=combine_profile_summaries([(st['sum_score'][i],st['sumsq_score'][i],st['units'][i])
                                                  for st,i in zip(stats,positions)])
        weights=np.array([float(st['units'][i]) for st,i in zip(stats,positions)])/nu
        dd=sum(w*st['diagnostics'][i] for w,st,i in zip(weights,stats,positions))
        dd[4]=np.max([st['diagnostics'][i,4] for st,i in zip(stats,positions)],axis=0)
        oo=sum(w*st['overlap'][i] for w,st,i in zip(weights,stats,positions))
        for pos,qi in enumerate(qids):
            rows.append(dict(dataset=args.dataset,n_rows=n,independent_units=nu,seed=seed,
                learner=learner,probability_source=source,stage=stage,structure=structure,method=method,
                query_index=int(qi),query_id=cat[int(qi)]['query_id'],
                contrast_family=cat[int(qi)]['contrast_family'],query_split='held_out',
                estimate=float(estimate[pos]),se=float(se[pos]),
                ci_lower=float(estimate[pos]-Z975*se[pos]),ci_upper=float(estimate[pos]+Z975*se[pos]),
                true_effect=float(truth[pos]),
                **{name:float(dd[j,pos]) for j,name in enumerate(DIAG_NAMES)},
                role='oracle_reference' if stage=='ORACLE' else ('legacy_reference' if stage=='REF' else 'observable_estimator'),
                intervals='profile_score_Wald_coverage_tested_not_guaranteed',
                sharing_scope='interaction_across_arms_not_query_level',
                outcome_truth_used_for_fitting=False,
                C_extra_query_access='X_only_moment' if stage=='C' else 'none'))
        for b,label in enumerate(('min_p_lt_.02','min_p_.02_to_.05','min_p_.05_to_.10','min_p_ge_.10')):
            overs.append(dict(dataset=args.dataset,n_rows=n,seed=seed,learner=learner,
                probability_source=source,stage=stage,structure=structure,method=method,
                overlap_bin=label,population_mass=float(oo[b,0]),
                alpha_error_contribution=float(oo[b,1]),alpha2_contribution=float(oo[b,2]),
                outcome_error_contribution=float(oo[b,3]),
                scope='postfit_nuisance_diagnostic_only_not_a_redefined_effect_target'))
    write_csv(slot/'query_results.csv',rows)
    write_csv(slot/'overlap_diagnostics.csv',overs)
    write_json(slot/'run_details.json',dict(n_rows=n,seed=seed,independent_units=nu,
        outputs=len(successful),held_queries=len(qids),eval_rows=len(eval_rows),
        optional_baseline_failures=sorted({r['method'] for r in inventory if r['status']=='failed'}),
        new_development_batch=bool(99001<=seed<=99005),original_DGP_unchanged=True,
        patient_rows_or_predictions_saved=False,base_model_fitted_once_per_fold=True))
    say(f'n={n} seed={seed}: complete ({len(rows)} method-query rows)')


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('cvd','phq9'),required=True)
    p.add_argument('--input');p.add_argument('--member');p.add_argument('--phq-root')
    p.add_argument('--output')
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit FICTIONAL software fixture, never real data.')
    p.add_argument('--n-rows',type=int,nargs='+',help='Default 6000; optional frozen grid: 3000 6000 12000.')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--eval-rows',type=int,default=8192,help='0 evaluates diagnostics on the full empirical bank.')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--no-oracle-reference',action='store_true')
    a=p.parse_args(argv)
    if a.n_rows is None:a.n_rows=[(1200 if a.dataset=='cvd' else 2400) if a.smoke else 6000]
    if a.seeds is None:a.seeds=[99001] if a.smoke else list(range(99001,99006))
    if len(set(a.seeds))!=len(a.seeds) or min(a.seeds)<0:p.error('Seeds must be distinct nonnegative integers.')
    if len(set(a.n_rows))!=len(a.n_rows) or min(a.n_rows)<120:p.error('Row counts must be distinct and >=120.')
    if a.dataset=='phq9' and any(n%4 for n in a.n_rows):p.error('PHQ-9 row counts must be divisible by four.')
    if a.eval_rows<0 or a.threads<1:p.error('Invalid evaluation size or thread count.')
    return a


def main(argv=None):
    from analysis_summary import summarize
    args=parse(argv)
    out=Path(args.output) if args.output else ROOT/(
        f'validation_runs/smoke_{args.dataset}' if args.smoke else f'results_{args.dataset}_projection_abc')
    with threadpool_limits(limits=args.threads):
        say('Reading original empirical design; missing real files do NOT trigger mock fallback.')
        data,steps,p,pbar,phi,feature_info=load_design(args.dataset,args.input,args.member,args.phq_root,args.smoke,patients=320)
        if args.check_inputs:
            print(json.dumps(dict(dataset=args.dataset,population_rows=len(steps),
                independent_source_profiles=int(len(steps)/data['core'].N_STEPS),
                original_basis_dimension=data['x_population'].shape[1],projection_dimension=phi.shape[1],
                query_counts={k:len(v) for k,v in data['queries']['indices'].items()},
                used_fictional_fixture=args.smoke,features=feature_info),indent=2))
            return
        out.mkdir(parents=True,exist_ok=True)
        cfg=dict(version=VERSION,dataset=args.dataset,n_rows=args.n_rows,seeds=args.seeds,
            smoke=args.smoke,eval_rows=args.eval_rows,threads=args.threads,
            no_oracle_reference=args.no_oracle_reference,frozen=json.loads((ROOT/'frozen_abc.json').read_text()),
            input_metadata=data['input_metadata'],
            runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                         pandas=pd.__version__,sklearn=sklearn.__version__))
        write_json(out/'manifest.json',dict(config=cfg))
        write_json(out/'input_audit.json',dict(metadata=data['input_metadata'],features=feature_info,
            fixture=args.smoke,no_new_outcome_basis=True,patient_data_exported=False))
        write_csv(out/'query_catalog.csv',catalog(data))
        mu,law=data_bridge.build_outcome_law(data,1.)
        truth=data_bridge.finite_population_targets(data,mu)
        write_json(out/'DGP_provenance.json',law)
        # The planned batch starts fresh; report-only reads remain separate.
        for n in args.n_rows:
            for seed in args.seeds:
                slot=out/'runs'/f'n_{n}'/f'seed_{seed}'
                for filename in ('query_results.csv', 'method_inventory.csv', 'failed.json'):
                    (slot/filename).unlink(missing_ok=True)
        failures=[]
        for n in args.n_rows:
            for seed in args.seeds:
                try:
                    run_seed(data,phi,mu,truth,args,out,seed,n)
                except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as e:
                    slot=out/'runs'/f'n_{n}'/f'seed_{seed}';slot.mkdir(parents=True,exist_ok=True)
                    write_json(slot/'failed.json',dict(type=type(e).__name__,message=str(e),seed=seed,
                         n_rows=n,no_reselection=True,not_replaced=True))
                    say(f'n={n} seed={seed}: FAILED {type(e).__name__}: {e}; retained, not silently replaced')
                    failures.append((n,seed))
                summarize(out)
        if failures:
            say('Some planned replicates are incomplete/failed; see run_inventory.csv.')
            raise SystemExit(2)
        say('Complete. Run python print_results.py; A/B/C are shown together, not ranked.')


if __name__=='__main__':
    main()
