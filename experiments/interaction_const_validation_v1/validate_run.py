#!/usr/bin/env python3
"""Run the specified replication plan and summarize its Monte Carlo results."""
from __future__ import annotations
from pathlib import Path
import argparse
import importlib
import sys
from result_io import ROOT, ENGINE, write_json, read_json


def arguments(dataset, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input')
    parser.add_argument('--member')
    parser.add_argument('--phq-root')
    parser.add_argument('--output')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Explicit fictional software check, never patient data.')
    parser.add_argument('--size-grid', action='store_true',
                        help='Run the 3000/6000/12000 sensitivity sizes; primary size stays 6000.')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('Positive threads required.')
    if args.smoke and args.size_grid:
        parser.error('The software smoke is not the sample-size study.')
    return args


def main(dataset, argv=None):
    args = arguments(dataset, argv)
    plan = read_json(ROOT / 'validation_plan.json')
    seeds = [119901] if args.smoke else plan['seeds'][dataset]
    sizes = ([1200 if dataset == 'cvd' else 2400] if args.smoke else
             ([3000, 6000, 12000] if args.size_grid else [6000]))
    out = Path(args.output).expanduser().resolve() if args.output else ROOT / (
        f'validation_runs/smoke_{dataset}' if args.smoke else
        f'results_{dataset}_const_validation_v1' + ('_sizes' if args.size_grid else ''))
    flags = ['--dataset', dataset, '--output', str(out), '--n-rows', *map(str, sizes),
             '--seeds', *map(str, seeds), '--eval-rows', str(plan['default_eval_rows']),
             '--threads', str(args.threads)]
    for key in ('input', 'member', 'phq_root'):
        value = getattr(args, key)
        if value:
            flags.extend(['--' + key.replace('_', '-'), value])
    for flag in ('smoke', 'check_inputs'):
        if getattr(args, flag):
            flags.append('--' + flag.replace('_', '-'))
    sys.path.insert(0, str(ENGINE))
    engine = importlib.import_module('run')
    if args.check_inputs:
        engine.main(flags)
        return

    write_json(out / 'validation_declaration.json', {
        'plan_id': plan['id'], 'dataset': dataset,
        'seeds': seeds, 'n_rows': sizes,
        'fictional_software_check': args.smoke,
        'targeted_confirmation': not args.smoke,
        'fresh_queries': False, 'patient_generalization': False,
        'estimator_changed': False,
    })
    print('REPLICATION: ' + ('FICTIONAL SOFTWARE CHECK' if args.smoke else
                            f'{len(seeds)} repetitions per sample size.'), flush=True)
    engine.main(flags)
    if (out / 'manifest.json').is_file():
        from validation_report import report
        report(out)
    print('Print the aggregate results: python print_results.py' +
          (' ' + str(out) if args.output or args.smoke or args.size_grid else ''), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['cvd', 'phq9'])
    ns, rest = parser.parse_known_args()
    main(ns.dataset, rest)
