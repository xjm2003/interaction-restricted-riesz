#!/usr/bin/env python3
"""Matched 2x2 interaction ablation on the archived CVD/PHQ-9 benchmark.

Uses the original covariate design, outcome law, query library and random streams.
Only aggregate independent-profile score moments leave a fold. No CRDM connection.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
from pathlib import Path
import argparse
import copy
import json
import sys
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ext_support import (ROOT, ENGINE, IntegrityError, say, read_json,
    write_json, write_frame, runtime, import_engine, find_reference,
    read_reference, reference_seed)
from ext_models import fit_probability, FitFailure
from ablation_core import PROCEDURES, PAIRS, matched_scores, summarize_scores

METHODS = ('logit', 'hgb', 'cal_linear', 'cal_sparse')
LEARNERS = ('linear', 'flexible')
NUMERICAL_ERRORS = (ValueError, FitFailure, ArithmeticError, np.linalg.LinAlgError)


def arguments(dataset=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if dataset is None:
        parser.add_argument('--dataset', choices=['cvd', 'phq9'], required=True)
    parser.add_argument('--reference', help='Read-only original n=6000, 100-seed validation results.')
    parser.add_argument('--input', help='Original CVD zip/CSV; archived default is unchanged.')
    parser.add_argument('--member', help='CVD archive member if needed.')
    parser.add_argument('--phq-root', help='Original qaoct_semisynthetic_v1 directory.')
    parser.add_argument('--output')
    parser.add_argument('--max-seeds', type=int,
                        help='Run the first N seeds from scratch; reports retain the full planned denominator.')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['logit', 'hgb'],
                        help='Schedule sources; all four remain in the plan. Default: logit hgb.')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-inputs', action='store_true', help='Validate inputs/reference without fitting.')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Explicit FICTIONAL software test, reduced external grids, four queries, one seed.')
    args = parser.parse_args(argv)
    if dataset:
        args.dataset = dataset
    if args.threads < 1 or (args.max_seeds is not None and args.max_seeds < 1):
        parser.error('threads and max-seeds must be positive.')
    args.methods = list(dict.fromkeys(args.methods))
    return args


def effective_plan(smoke, dataset):
    plan = read_json(ROOT / 'ablation_plan.json')
    if not smoke:
        start = plan['seed_start'][dataset]
        return plan, 6000, list(range(start, start + plan['replications']))
    plan = copy.deepcopy(plan)
    plan['id'] = 'FICTIONAL_SOFTWARE_TEST_interaction_ablation_v1'
    plan['probability_floor_grid'] = [.01]
    plan['logit']['C'] = [.1, 1.]
    plan['hgb'].update(learning_rate=[.1], max_leaf_nodes=[7],
                       iterations=[5, 10], min_samples_leaf=10)
    # CAL paths remain archived and unmodified, including in explicit smoke tests.
    return plan, (1200 if dataset == 'cvd' else 2400), [119901]


def prepare_sample(engine, data, mu, n, seed, dataset):
    sample = engine.data_bridge.draw_sample(
        data, n, seed, 'patient' if dataset == 'phq9' else 'iid_row')
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 923]))
    y = mu[sample['rows'], sample['actions']] + rng.normal(0, 1., n)
    observed = {key: sample[key] for key in ('rows', 'steps', 'actions', 'x', 'clusters')}
    folds = engine.data_bridge.group_folds(observed['clusters'], 3, seed + 747)
    # This inherited marginal pilot is supplied only to archived CAL fitting.
    return observed, y, folds, sample['pbar']








def _record_failure(error):
    return dict(status='failed', error_type=type(error).__name__, message=str(error))


def process_fold(engine, data, observed, y, pbar, train, test, qids, plan,
                 seed, fold, scheduled):
    results = {}
    x, steps, actions, clusters = (observed[key] for key in ('x', 'steps', 'actions', 'clusters'))
    inner = engine.data_bridge.group_folds(clusters[train], 2, seed + fold + 172)
    say(f'seed={seed} fold={fold+1}/3: refit original ridge/tree outcomes once')
    tick = time.monotonic()
    try:
        outcome_models = engine.fit_outcomes(x[train], steps[train], actions[train],
                           y[train], data['core'].N_STEPS, seed + fold)
        train_mu = {learner: engine.predict_outcome(outcome_models[learner],
                    x[train], steps[train], data['core'].N_STEPS) for learner in LEARNERS}
        test_mu = {learner: engine.predict_outcome(outcome_models[learner],
                    x[test], steps[test], data['core'].N_STEPS) for learner in LEARNERS}
    except NUMERICAL_ERRORS as error:
        for method in scheduled:
            results[method] = _record_failure(error)
        say(f'seed={seed} fold={fold+1}/3: outcome fitting failed; all scheduled source slots recorded')
        return results
    outcome_seconds = time.monotonic() - tick
    train_scales = engine.get_scales(data, qids, observed['rows'][train])
    test_scales = engine.get_scales(data, qids, observed['rows'][test])
    cal_result = None
    cal_error = None
    cal_fit_seconds = 0.
    for method in scheduled:
        tick = time.monotonic()
        try:
            if method in ('logit', 'hgb'):
                say(f'seed={seed} fold={fold+1}/3: {method} tuning/refit, then all five score procedures')
                probability, selection = fit_probability(method, x[train], steps[train],
                    actions[train], inner, plan, seed + fold + 36001)
                p_train = probability.predict(x[train], steps[train])
                p_test = probability.predict(x[test], steps[test])
            else:
                if cal_result is None and cal_error is None:
                    say(f'seed={seed} fold={fold+1}/3: archived CAL paths once; no B1/shared-sparse fit')
                    cal_tick = time.monotonic()
                    artifacts = engine.data_bridge.observed_artifacts(observed, train,
                        data['queries'], data['core'], pbar, seed + fold + 172)
                    try:
                        cal_result = engine.legacy_run.coherent.fit_observed(artifacts, data['queries'],
                            pbar, data['core'], seed + fold, False, say)
                    except NUMERICAL_ERRORS + (RuntimeError,) as error:
                        cal_error = error
                    cal_fit_seconds = time.monotonic() - cal_tick
                if cal_error is not None:
                    raise FitFailure('Archived CAL fit failed: ' + str(cal_error))
                name = engine.CAL_NAMES[method]
                if name not in cal_result['models']:
                    raise FitFailure(cal_result['failures'].get(name, 'Required CAL selected fit missing.'))
                probability = cal_result['models'][name]
                p_train = probability.predict_probabilities(x[train], steps[train])
                p_test = probability.predict_probabilities(x[test], steps[test])
                selection = [dict(kind=method, selected=True, archived_model=name,
                                   **cal_result['choices'][name])]
            stats, paired_stats, diagnostics = {}, {}, {}
            for learner in LEARNERS:
                scores, diag = matched_scores(engine, p_train, p_test,
                    train_mu[learner], test_mu[learner], train_scales, test_scales,
                    steps[train], steps[test], actions[test], y[test], data['core'].N_STEPS)
                moments = summarize_scores(engine, scores, clusters[test], steps[test], data['core'].N_STEPS)
                stats[learner] = moments['stats']
                paired_stats[learner] = moments['paired_stats']
                diagnostics[learner] = diag
            payload = dict(status='complete', stats=stats, paired_stats=paired_stats,
                selection=selection, diagnostics=diagnostics,
                fit_seconds=time.monotonic()-tick, outcome_refit_seconds=outcome_seconds,
                cal_shared_fit_seconds=cal_fit_seconds if method.startswith('cal_') else 0.,
                training_rows=int(len(train)), held_rows=int(len(test)),
                held_units=int(len(np.unique(clusters[test]))),
                uses_known_marginal_pilot=method.startswith('cal_'))
            results[method] = payload
            say(f'seed={seed} fold={fold+1}/3: {method} complete')
        except NUMERICAL_ERRORS as error:
            results[method] = _record_failure(error)
            say(f'seed={seed} fold={fold+1}/3: {method} FAILED ({type(error).__name__}); other sources continue')
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


def _combine(engine, records, learner, category, name):
    parts = []
    for record in records:
        moment = record[category][learner][name]
        parts.append((np.asarray(moment['sum_score']), np.asarray(moment['sumsq_score']), moment['units']))
    return engine.combine_profile_summaries(parts)


def _cal_replay_diagnostics(results, reference, qids):
    checks = []
    if reference is None:
        return checks
    original_names = {'raw_full': 'A_full', 'projected_full': 'B_CONST',
                      'projected_restricted': 'C_CONST'}
    if results.empty:
        return checks
    for source in ('cal_linear', 'cal_sparse'):
        for learner in LEARNERS:
            for procedure, old_name in original_names.items():
                actual = results[(results.probability_source == source) &
                    (results.learner == learner) & (results.method == procedure)]
                if actual.empty:
                    continue
                expected = reference[(reference.probability_source == source) &
                    (reference.learner == learner) & (reference.method == old_name)]
                if expected.query_index.duplicated().any() or set(expected.query_index) != set(qids):
                    continue
                actual = actual.set_index('query_index').loc[qids]
                expected = expected.set_index('query_index').loc[qids]
                errors = {}
                for name in ('estimate', 'se', 'true_effect'):
                    aa, bb = actual[name].to_numpy(float), expected[name].to_numpy(float)
                    errors[name + '_max_abs_difference'] = float(np.max(np.abs(aa-bb)))
                checks.append(dict(probability_source=source, learner=learner,
                    procedure=procedure, original_method=old_name, **errors))
    return checks


def run_one(engine, data, mu, probabilities, truth, qids, catalog, plan, args,
            out, seed, n, reference):
    slot = out / 'runs' / f'n_{n}' / f'seed_{seed}'
    slot.mkdir(parents=True, exist_ok=True)
    for name in ('query_results.csv', 'paired_differences.csv', 'method_inventory.csv',
                 'reference_results.csv', 'selection.json', 'alignment.json'):
        (slot / name).unlink(missing_ok=True)
    observed, y, folds, pbar = prepare_sample(engine, data, mu, n, seed, args.dataset)
    ref = None if reference is None else reference_seed(reference, n, seed, qids, truth)
    replay = replay_oracle(engine, data, observed, y, folds, mu, probabilities, qids, truth, ref)
    if ref is not None:
        write_frame(slot / 'reference_results.csv', ref)
    alignment = dict(**replay, phenotype_or_patient_rows_exported=False)
    write_json(slot / 'alignment.json', alignment)
    all_rows = np.arange(n)
    fold_results = []
    for fold, test in enumerate(folds):
        test = np.sort(test)
        train = np.setdiff1d(all_rows, test)
        fold_results.append(process_fold(engine, data, observed, y, pbar, train, test,
                     qids, plan, seed, fold, args.methods))
    rows, differences, inventory, selection = [], [], [], []
    for source in METHODS:
        records = []
        for fold in range(3):
            record = fold_results[fold].get(source)
            records.append(record)
            if record and record['status'] == 'complete':
                selection.extend(dict(seed=seed, fold=fold, probability_source=source, **entry)
                                 for entry in record['selection'])
        state = ('complete' if all(r and r['status'] == 'complete' for r in records)
                 else 'failed' if any(r and r['status'] == 'failed' for r in records)
                 else 'not_processed')
        inventory.append(dict(dataset=args.dataset, n_rows=n, seed=seed,
            probability_source=source, method=source, status=state,
            completed_folds=sum(bool(r and r['status'] == 'complete') for r in records)))
        if state != 'complete':
            continue
        for learner in LEARNERS:
            for procedure in PROCEDURES:
                estimate, se, units = _combine(engine, records, learner, 'stats', procedure)
                for j, query in enumerate(qids):
                    rows.append(dict(dataset=args.dataset, n_rows=n, independent_units=units,
                        seed=seed, learner=learner, probability_source=source, method=procedure,
                        query_index=int(query), query_id=catalog[int(query)]['query_id'],
                        contrast_family=catalog[int(query)]['contrast_family'], query_split='held_out',
                        estimate=float(estimate[j]), se=float(se[j]),
                        ci_lower=float(estimate[j]-engine.Z975*se[j]),
                        ci_upper=float(estimate[j]+engine.Z975*se[j]), true_effect=float(truth[j]),
                        origin='supplementary_matched_ablation',
                        uses_known_marginal_pilot=source.startswith('cal_'),
                        intervals='nominal_profile_score_Wald; original_target_coverage_evaluated'))
            for pair, (left, right) in PAIRS.items():
                difference, se, units = _combine(engine, records, learner, 'paired_stats', pair)
                for j, query in enumerate(qids):
                    differences.append(dict(dataset=args.dataset, n_rows=n, independent_units=units,
                        seed=seed, learner=learner, probability_source=source, pair=pair,
                        left=left, right=right, query_index=int(query),
                        query_id=catalog[int(query)]['query_id'],
                        estimate_difference=float(difference[j]), se_difference=float(se[j]),
                        ci_lower=float(difference[j]-engine.Z975*se[j]),
                        ci_upper=float(difference[j]+engine.Z975*se[j]),
                        interpretation='nominal_paired_score_difference; not_validated_Hausman_test'))
    fields = ['dataset', 'n_rows', 'independent_units', 'seed', 'learner', 'probability_source',
              'method', 'query_index', 'query_id', 'estimate', 'se', 'ci_lower', 'ci_upper', 'true_effect']
    result = pd.DataFrame(rows) if rows else pd.DataFrame(columns=fields)
    alignment['cal_replay_diagnostics'] = _cal_replay_diagnostics(result, ref, qids)
    write_json(slot / 'alignment.json', alignment)
    write_frame(slot / 'query_results.csv', result)
    diff_fields = ['dataset', 'n_rows', 'independent_units', 'seed', 'learner',
                   'probability_source', 'pair', 'left', 'right', 'query_index',
                   'query_id', 'estimate_difference', 'se_difference', 'ci_lower', 'ci_upper']
    write_frame(slot / 'paired_differences.csv', pd.DataFrame(differences) if differences
                else pd.DataFrame(columns=diff_fields))
    write_frame(slot / 'method_inventory.csv', pd.DataFrame(inventory))
    write_json(slot / 'selection.json', selection)
    status = {row['probability_source']: row['status'] for row in inventory}
    say(f'seed={seed}: ' + ', '.join(key + '=' + value for key, value in status.items()))
    return status


def main(dataset=None, argv=None):
    args = arguments(dataset, argv)
    out = Path(args.output).expanduser().resolve() if args.output else ROOT / (
        f'software_smoke_{args.dataset}' if args.smoke else f'results_{args.dataset}_ablation')
    if out == ROOT or out.is_relative_to(ENGINE) or ROOT.is_relative_to(out):
        raise IntegrityError('Output cannot overwrite the package, frozen engine, or a package ancestor.')
    if args.report_only:
        from ablation_report import summarize, display
        summarize(out)
        display(out)
        return
    plan, n, seeds = effective_plan(args.smoke, args.dataset)
    engine = import_engine()
    with threadpool_limits(limits=args.threads):
        say('Loading archived semi-synthetic design; no CRDM connection or clinical outcome extraction.')
        data, steps, probabilities, pbar, phi, feature_info = engine.load_design(
            args.dataset, args.input, args.member, args.phq_root, args.smoke, patients=320)
        all_catalog = engine.catalog(data)
        qids = np.asarray(data['queries']['indices']['held_out'], int)
        if args.smoke:
            qids = qids[:4]
        catalog = {row['query_index']: row for row in all_catalog}
        mu, law = engine.data_bridge.build_outcome_law(data, 1.)
        all_truth = engine.data_bridge.finite_population_targets(data, mu)
        held = np.asarray(data['queries']['indices']['held_out'], int)
        truth = all_truth[[int(np.flatnonzero(held == query)[0]) for query in qids]]
        reference = None if args.smoke and not args.reference else find_reference(args.dataset, args.reference)
        if reference is not None:
            if out.is_relative_to(reference) or reference.is_relative_to(out):
                raise IntegrityError('Output and reference directories must be separate and nonnested.')
            read_reference(reference, args.dataset, n, seeds)
        if args.check_inputs:
            say(json.dumps(dict(dataset=args.dataset, source_profiles=int(len(steps)/data['core'].N_STEPS),
                simulated_rows=n, independent_units=n//data['core'].N_STEPS,
                planned_repeats=len(seeds), held_queries=len(qids), sources=list(METHODS),
                scheduled_sources=args.methods, original_result_directory=str(reference) if reference else None,
                runtime=runtime(), fixture=args.smoke), indent=2))
            if reference is not None:
                missing = [seed for seed in seeds if not (
                    reference/'runs'/f'n_{n}'/f'seed_{seed}'/'query_results.csv').is_file()]
                if missing:
                    raise IntegrityError(f'{len(missing)} reference seeds lack query results.')
            return
        config = dict(version='interaction-ablation-v1', dataset=args.dataset, n_rows=n,
            seeds=seeds, fixture=args.smoke, plan=plan, threads=args.threads, runtime=runtime(),
            held_query_indices=qids.tolist(),
            reference_directory=str(reference) if reference is not None else None)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'manifest.json', dict(config=config))
        write_frame(out/'query_catalog.csv', pd.DataFrame([catalog[int(q)] for q in qids]))
        write_json(out/'DGP_provenance.json', law)
        label = 'FICTIONAL SOFTWARE TEST' if args.smoke else 'SUPPLEMENTARY MATCHED ABLATION'
        say(f'{label}: {len(seeds)} planned seeds, {len(qids)} queries; sources {args.methods}')
        for seed in seeds[:args.max_seeds]:
            run_one(engine, data, mu, probabilities, truth, qids, catalog, plan,
                    args, out, seed, n, reference)
            from ablation_report import summarize
            summarize(out)
        from ablation_report import summarize, display
        summarize(out)
        display(out, brief=True)
        say('Output: ' + str(out))


def entrypoint(dataset=None):
    try:
        main(dataset)
    except KeyboardInterrupt:
        print('\nInterrupted. Finished seed result tables remain available; the next run fits again from scratch.', flush=True)
        sys.exit(130)
    except (IntegrityError, ValueError, FileNotFoundError) as error:
        print('STOP: ' + str(error), flush=True)
        sys.exit(2)


if __name__ == '__main__':
    entrypoint()
