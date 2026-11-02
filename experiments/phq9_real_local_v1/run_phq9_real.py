#!/usr/bin/env python3
"""Fit exploratory real PHQ-9 adjusted contrasts from existing local exports."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from phq_inputs import load_inputs, InputError

ROOT = Path(__file__).resolve().parent


def say(message):
    print(time.strftime('%H:%M:%S') + ' ' + str(message), flush=True)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows, columns=None):
    temp = path.with_name(path.name + '.tmp')
    pd.DataFrame(rows, columns=columns).to_csv(temp, index=False)
    temp.replace(path)


def audit_text(audit):
    lines = ['REAL PHQ-9 | EXPLORATORY SELECTED-OUTCOME COHORT',
             'Existing local files only; no SQL, simulated outcomes, or treatment imputation.',
             'Patients: ' + str(audit['n_patients']),
             'Real PHQ-9 window: index day 60..120; recorded exposure: [0,30).',
             'Actions: ' + ', '.join(k + '=' + str(v) for k, v in audit['action_counts'].items()),
             'Baseline covariates: ' + ', '.join(x['name'] for x in audit['features'])]
    if audit['candidate_index_inventory']:
        c = audit['candidate_index_inventory']
        lines.append('Candidate index rows: {candidate_rows}; selected outcome matches: {outcome_rows_matched}.'.format(**c))
        lines.append('Unmatched candidate rows are NOT classified as clinical missing outcomes.')
    lines.extend('NOTE: ' + w for w in audit['warnings'])
    return '\n'.join(lines)


def summary(audit, results):
    lines = [audit_text(audit), '',
             'Estimates are adjusted statistical contrasts in this selected cohort.',
             'Causal identification, time-zero/grace-period handling, and full-cohort transport are not certified.',
             'Nominal score intervals do not repair selection bias or structural misspecification.',
             'No causal RMSE or coverage is calculable from these observed outcomes.', '']
    d = pd.DataFrame(results['estimates'])
    if not d.empty:
        own = d[d.is_primary_method]
        if len(own):
            lines.append('YOUR METHOD: CAL-sparse + matched outcome projection + restricted Riesz (CONST).')
            lines.append('CAL penalties and probability floor selected inside training folds; real observed actions only.')
            lines.append('Both outcome learners are shown; dedicated output: own_method_estimates.csv.')
            own_cols = ['learner', 'contrast', 'estimate', 'se', 'ci_lower', 'ci_upper']
            lines.append(own[own_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
            lines.append('')
        else:
            lines.append('CAL-sparse primary method was not requested; own_method_estimates.csv contains headers only.')
        primary_probability = ((d.probability_source.str.startswith('cal_') &
                                d.probability_setting.eq('inner_cv_selected')) |
                               (~d.probability_source.str.startswith('cal_') & d.probability_floor.eq(0)))
        chosen = d[primary_probability & (d.learner == 'flexible') &
                   d.method.isin(['raw_full', 'projected_restricted'])]
        lines.append('Matched display: flexible outcome, selected CAL or unmodified Logit/HGB probabilities.')
        lines.append('Contrast = left arm minus right arm, in PHQ-9 points. Negative means lower recorded score.')
        cols = ['probability_source', 'probability_setting', 'method', 'contrast', 'estimate', 'se', 'ci_lower', 'ci_upper']
        lines.append(chosen[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}') if len(chosen) else
                     'Unmodified propensity results unavailable; inspect fit_diagnostics.json and sensitivity rows.')
    else:
        lines.append('No estimator completed. Inspect fit_diagnostics.json.')
    lines.extend(['', 'Fixed floors 0, 0.01 and 0.02 are additional sensitivity settings; not extra independent fits.',
                  'For CAL, inner_cv_selected uses each fold-selected floor; probability_floor is NA for this setting.',
                  'CAL-sparse includes L1=0 and an empirical intercept-only candidate; consult selected choices.',
                  'Ridge, all five score procedures, and six pairwise contrasts are saved in estimates.csv.',
                  'Full-AIPW is the reference; restricted correction adds the CONST interaction restriction.',
                  'Plugin confidence intervals are not reported. Paired differences are not validated Hausman tests.',
                  'One cohort and one cross-fit partition: no claim of 100 independent replications.'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default=str(Path.home()))
    for flag in ('outcome', 'actions', 'baseline', 'index'):
        ap.add_argument('--' + flag, help='Explicit existing local CSV/CSV.GZ; default original named export.')
    ap.add_argument('--covariates', help='Comma-separated columns in the explicit baseline table; pre-index only.')
    ap.add_argument('--methods', nargs='+', choices=['cal_sparse', 'cal_linear', 'logit', 'hgb'],
                    default=['cal_sparse', 'cal_linear', 'logit', 'hgb'],
                    help='Probability sources; cal_sparse + projected_restricted is the primary method.')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20261022)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--output', default=str(ROOT / 'results_phq9_real_with_cal'))
    ap.add_argument('--check-inputs', action='store_true', help='Validate local real data and report available baseline features without fitting.')
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args(argv)
    if args.folds < 2 or args.threads < 1:
        ap.error('folds must be at least 2 and threads must be positive.')
    out = Path(args.output).expanduser().resolve()
    if out == ROOT or ROOT.is_relative_to(out):
        raise InputError('Output cannot be the package directory or one of its ancestors.')
    if args.report_only:
        print((out / 'summary.txt').read_text(encoding='utf-8'))
        return
    say('Reading original real PHQ-9 exports from disk; no database connection.')
    X, action, y, audit = load_inputs(args.data_root, args.outcome, args.actions,
                                      args.baseline, args.index, args.covariates)
    if np.bincount(action, minlength=4).min() < args.folds:
        raise InputError('Each treatment category needs at least --folds patients. No absent arm is imputed.')
    counts = np.bincount(action, minlength=4)
    if any(m.startswith('cal_') for m in args.methods) and np.any(counts - np.ceil(counts / args.folds).astype(int) < 2):
        raise InputError('CAL needs at least two training patients per treatment arm in each outer fold.')
    config = dict(version='phq9-real-local-v1.1-cal',
                  input_files=audit['input_files'], features=audit['features'], methods=list(dict.fromkeys(args.methods)),
                  folds=args.folds, seed=args.seed, threads=args.threads, probability_floors=[0., .01, .02],
                  cal_selection=dict(inner_folds=2, criterion='observed_action_cross_entropy',
                                     ridges=[.003,.03,.3], l1=[0.,.01,.05], floors=[.01,.02],
                                     pilot='training_fold_empirical_arm_frequencies'),
                  primary_method='cal_sparse/inner_cv_selected/projected_restricted',
                  status='EXPLORATORY_SELECTED_COHORT',
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                               scipy=scipy.__version__, sklearn=sklearn.__version__))
    out.mkdir(parents=True, exist_ok=True)
    if args.check_inputs:
        write_json(out / 'input_audit.json', audit)
        print(audit_text(audit), flush=True)
        say('Probability sources: ' + ', '.join(config['methods']))
        if 'cal_sparse' in args.methods:
            say('Primary method enabled: CAL-sparse + matched CONST projection/restricted Riesz.')
        say('Input check passed. This does not certify causal identification. Run without --check-inputs to fit.')
        return
    result_paths = [out / name for name in ('estimates.csv', 'paired_differences.csv', 'overlap.csv',
                    'own_method_estimates.csv', 'fit_diagnostics.json', 'summary.txt')]
    source_paths = {Path(record['path']).resolve() for record in audit['input_files'].values()}
    if any(path.resolve() in source_paths for path in result_paths):
        raise InputError('A result filename would overwrite an input file; choose another output directory.')
    for path in result_paths:
        path.unlink(missing_ok=True)
    write_json(out / 'manifest.json', {'config': config})
    write_json(out / 'input_audit.json', audit)
    print(audit_text(audit), flush=True)
    from phq_estimation import run_estimation
    results = run_estimation(X, action, y, seed=args.seed, folds=args.folds,
                             propensity_sources=tuple(config['methods']),
                             probability_floors=(0., .01, .02), threads=args.threads, progress=say)
    for name in ('estimates', 'paired_differences', 'overlap'):
        write_csv(out / (name + '.csv'), results[name])
    write_csv(out / 'own_method_estimates.csv',
              [r for r in results['estimates'] if r['is_primary_method']],
              columns=list(results['estimates'][0]))
    write_json(out / 'fit_diagnostics.json', results['fit_diagnostics'])
    report = summary(audit, results)
    (out / 'summary.txt').write_text(report, encoding='utf-8')
    print(report, flush=True)
    say('Output: ' + str(out))


if __name__ == '__main__':
    try:
        main()
    except (InputError, ValueError, FileNotFoundError, RuntimeError, ArithmeticError) as error:
        print('STOP: ' + str(error), flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print('\nInterrupted. Run the command again to refit.', flush=True)
        sys.exit(130)
