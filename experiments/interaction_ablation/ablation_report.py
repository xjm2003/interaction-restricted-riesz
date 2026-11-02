"""Whole-replication reporting for matched interaction ablations.

Patient-score intervals and Monte Carlo performance intervals are distinct.
Partial results are diagnostic; primary metrics require the entire planned block.
"""
from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from ext_support import (IntegrityError, read_json,
                         write_frame, write_json)

SOURCES = ('logit', 'hgb', 'cal_linear', 'cal_sparse')
LEARNERS = ('linear', 'flexible')
PROCEDURES = ('raw_full', 'raw_restricted', 'projected_full',
              'projected_restricted', 'raw_plugin')
PAIRS = {
    'correction_projected': ('projected_restricted', 'projected_full'),
    'correction_raw': ('raw_restricted', 'raw_full'),
    'projection_full': ('projected_full', 'raw_full'),
    'projection_restricted': ('projected_restricted', 'raw_restricted'),
    'overall': ('projected_restricted', 'raw_full'),
}
GROUP = ['dataset', 'n_rows', 'learner', 'probability_source', 'method']
METRICS = ['rmse', 'rmse_mcse', 'rmse_ci_low', 'rmse_ci_high',
           'mean_absolute_query_bias', 'coverage', 'coverage_mcse',
           'coverage_ci_low', 'coverage_ci_high', 'mean_width', 'width_mcse',
           'empirical_sd', 'rms_reported_se', 'se_to_sd',
           'top1_seed_mse_share', 'top5_seed_mse_share']
QUERY_METRICS = ['bias', 'bias_mcse', 'empirical_sd', 'rms_se', 'se_to_sd',
                 'bias_squared_adjusted', 'coverage', 'coverage_mcse',
                 'mean_width', 'rmse', 'rmse_mcse', 'rmse_ci_low', 'rmse_ci_high']
RATIO_METRICS = ['ratio', 'ci_low', 'ci_high', 'family_low', 'family_high',
                 'coverage_difference', 'coverage_difference_mcse', 'width_ratio']


def _mc(values):
    a = np.asarray(values, dtype=float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


def rmse_stats(mse):
    x = np.asarray(mse, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('MSE must be a finite nonnegative vector.')
    empty = dict(rmse=np.nan, rmse_mcse=np.nan, rmse_ci_low=np.nan, rmse_ci_high=np.nan)
    if not len(x):
        return empty
    mean = float(x.mean())
    value = float(np.sqrt(mean))
    if len(x) < 2:
        return dict(empty, rmse=value)
    if mean == 0:
        return {key: 0. for key in empty}
    mc = _mc(x)
    crit = float(student_t.ppf(.975, len(x) - 1))
    log_halfwidth = crit * mc / (2 * mean)
    with np.errstate(over='ignore'):
        low, high = np.exp(np.log(value) + np.array([-1., 1.]) * log_halfwidth)
    return dict(rmse=value, rmse_mcse=mc / (2 * value),
                rmse_ci_low=float(low), rmse_ci_high=float(high))


def paired_ratio(numerator_mse, denominator_mse, family=16):
    a, b = np.asarray(numerator_mse, float), np.asarray(denominator_mse, float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('Paired MSE arrays must have equal nonempty vector shapes.')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError('Paired MSE arrays must be finite and nonnegative.')
    if not isinstance(family, int) or family < 1:
        raise ValueError('The comparison family size must be a positive integer.')
    values = dict(ratio=np.nan, ci_low=np.nan, ci_high=np.nan,
                  family_low=np.nan, family_high=np.nan)
    if a.mean() <= 0 or b.mean() <= 0:
        return values  # log-ratio uncertainty is undefined at this boundary
    ratio = float(np.sqrt(a.mean() / b.mean()))
    values['ratio'] = ratio
    if len(a) < 2:
        return values
    influence = .5 * (a / a.mean() - b / b.mean())
    se = _mc(influence)
    critical = [student_t.ppf(.975, len(a)-1),
                student_t.ppf(1-.05/(2*family), len(a)-1)]
    with np.errstate(over='ignore'):
        ordinary = np.exp(np.log(ratio) + np.array([-1., 1.])*critical[0]*se)
        adjusted = np.exp(np.log(ratio) + np.array([-1., 1.])*critical[1]*se)
    values.update(ci_low=float(ordinary[0]), ci_high=float(ordinary[1]),
                  family_low=float(adjusted[0]), family_high=float(adjusted[1]))
    return values


def _status(complete, failed, fixture, count):
    if fixture:
        return 'FICTIONAL_SOFTWARE_TEST'
    if failed:
        return 'FAILED_SLOTS_FULL_PLAN_NA'
    if complete:
        return 'COMPLETE_SUPPLEMENTARY_ANALYSIS'
    return 'INTERIM_NOT_CONFIRMATORY' if count else 'NOT_YET_PROCESSED'


def _primary_and_interim(values, complete):
    result = {key: (value if complete else np.nan) for key, value in values.items()}
    result.update({'interim_' + key: value for key, value in values.items()})
    return result


def _read_csv(path):
    try:
        return pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')
    except (ValueError, pd.errors.EmptyDataError) as exc:
        raise IntegrityError('Invalid CSV result: ' + str(path)) from exc


def _validate_query_rows(d, cfg, seed, completed):
    required = set(GROUP + ['seed', 'query_index', 'query_id', 'independent_units',
                            'estimate', 'se', 'ci_lower', 'ci_upper', 'true_effect'])
    if not required.issubset(d.columns):
        raise IntegrityError('Completed query results lack required columns.')
    if d.empty and completed:
        raise IntegrityError('Completed source has no query results.')
    if d.empty:
        return
    if (not d.dataset.eq(cfg['dataset']).all() or not d.n_rows.eq(cfg['n_rows']).all()
            or not d.seed.eq(seed).all()):
        raise IntegrityError('Result dataset/sample/seed keys differ from the manifest.')
    if not set(d.probability_source).issubset(completed):
        raise IntegrityError('Results present for a source not marked complete.')
    keys = ['learner', 'probability_source', 'method', 'query_index']
    if d.duplicated(keys).any():
        raise IntegrityError('Duplicate completed query rows.')
    wanted = {(l, m, int(q)) for l in LEARNERS for m in PROCEDURES
              for q in cfg['held_query_indices']}
    for source in completed:
        g = d[d.probability_source.eq(source)]
        got = set(zip(g.learner, g.method, g.query_index))
        if got != wanted:
            raise IntegrityError('A completed source is missing a learner/procedure/query: ' + source)
    numeric = ['estimate', 'se', 'ci_lower', 'ci_upper', 'true_effect', 'independent_units']
    try:
        finite = np.isfinite(d[numeric].to_numpy(float)).all()
    except (TypeError, ValueError) as exc:
        raise IntegrityError('Non-numeric completed result values.') from exc
    if not finite or (d.se < 0).any() or (d.ci_lower > d.ci_upper).any():
        raise IntegrityError('Nonfinite or invalid completed result values.')
    expected_units = cfg['n_rows'] // (4 if cfg['dataset'] == 'phq9' else 1)
    if not d.independent_units.eq(expected_units).all():
        raise IntegrityError('Independent-unit counts differ from the planned sampling units.')


def _validate_pairs(d, cfg, seed, completed):
    if d.empty:
        return
    required = {'dataset', 'n_rows', 'seed', 'learner', 'probability_source',
                'query_index', 'pair', 'estimate_difference', 'se_difference'}
    if not required.issubset(d.columns):
        raise IntegrityError('Paired-score diagnostic lacks required columns.')
    if (not d.dataset.eq(cfg['dataset']).all() or not d.n_rows.eq(cfg['n_rows']).all()
            or not d.seed.eq(seed).all()):
        raise IntegrityError('Paired-score diagnostic keys differ from the manifest.')
    if not set(d.probability_source).issubset(completed):
        raise IntegrityError('Paired diagnostics for an incomplete source.')
    if d.duplicated(['learner', 'probability_source', 'query_index', 'pair']).any():
        raise IntegrityError('Duplicated paired-score diagnostic.')
    wanted = {(l, p, int(q)) for l in LEARNERS for p in PAIRS for q in cfg['held_query_indices']}
    for source in completed:
        g = d[d.probability_source.eq(source)]
        if set(zip(g.learner, g.pair, g.query_index)) != wanted:
            raise IntegrityError('Incomplete paired-score diagnostics for ' + source)
    if (not np.isfinite(d[['estimate_difference', 'se_difference']].to_numpy(float)).all()
            or (d.se_difference < 0).any()):
        raise IntegrityError('Invalid paired-score moments.')


def _metrics(g, seed_stats):
    count = len(seed_stats)
    values = rmse_stats(seed_stats.mse)
    qmean = g.groupby('query_index').error.mean()
    cov = float(seed_stats.coverage.mean())
    cse = _mc(seed_stats.coverage)
    crit = float(student_t.ppf(.975, count-1)) if count > 1 else np.nan
    sd = float(np.sqrt(g.groupby('query_index').error.var(ddof=1).mean())) if count > 1 else np.nan
    rms_se = float(np.sqrt(np.mean(g.se**2)))
    total_mse = float(seed_stats.mse.sum())
    ordered = seed_stats.mse.sort_values(ascending=False)
    values.update(mean_absolute_query_bias=float(qmean.abs().mean()), coverage=cov,
                  coverage_mcse=cse, coverage_ci_low=max(0., cov-crit*cse) if count > 1 else np.nan,
                  coverage_ci_high=min(1., cov+crit*cse) if count > 1 else np.nan,
                  mean_width=float(seed_stats.width.mean()), width_mcse=_mc(seed_stats.width),
                  empirical_sd=sd, rms_reported_se=rms_se,
                  se_to_sd=rms_se/sd if np.isfinite(sd) and sd > 0 else np.nan,
                  top1_seed_mse_share=float(ordered.iloc[0]/total_mse) if total_mse > 0 else np.nan,
                  top5_seed_mse_share=float(ordered.iloc[:5].sum()/total_mse) if total_mse > 0 else np.nan)
    return values


def summarize(out):
    out = Path(out)
    if not (out/'manifest.json').is_file():
        raise IntegrityError('No ablation manifest in ' + str(out))
    manifest = read_json(out/'manifest.json')
    cfg = manifest['config']
    seeds, qids = list(cfg['seeds']), list(cfg['held_query_indices'])
    if not seeds or len(set(seeds)) != len(seeds) or not qids or len(set(qids)) != len(qids):
        raise IntegrityError('Duplicate or empty planned seeds/queries.')
    n, planned, Q = int(cfg['n_rows']), len(seeds), len(qids)
    fixture = bool(cfg['fixture'])
    family = int(cfg.get('plan', {}).get('primary_family_size', 16))
    if family != 16:
        raise IntegrityError('This ablation report requires the fixed 16-comparison family.')
    inventory, chunks, pair_chunks = [], [], []
    processed_slots = 0
    for seed in seeds:
        slot = out/'runs'/f'n_{n}'/f'seed_{seed}'
        path = slot/'query_results.csv'
        d = _read_csv(path) if path.is_file() else pd.DataFrame()
        completed = sorted(set(d.probability_source)) if not d.empty else []
        status = {}
        if (slot/'method_inventory.csv').is_file():
            table = _read_csv(slot/'method_inventory.csv')
            status = dict(zip(table.probability_source, table.status))
        if path.is_file() or status:
            processed_slots += 1
        for source in SOURCES:
            state = ('complete' if source in completed else 'failed'
                     if status.get(source) == 'failed' else 'not_processed')
            inventory.append(dict(dataset=cfg['dataset'], n_rows=n, seed=seed,
                                  probability_source=source, status=state))
        if not path.is_file():
            continue
        _validate_query_rows(d, cfg, seed, completed)
        if not d.empty:
            chunks.append(d)
        path = slot/'paired_differences.csv'
        if path.is_file():
            pdiff = _read_csv(path)
            _validate_pairs(pdiff, cfg, seed, completed)
            if not pdiff.empty:
                pair_chunks.append(pdiff)
    inv = pd.DataFrame(inventory)
    write_frame(out/'run_inventory.csv', inv)
    data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not data.empty:
        if data.duplicated(GROUP+['seed', 'query_index']).any():
            raise IntegrityError('Duplicate cross-seed results.')
        data['error'] = data.estimate - data.true_effect
        data['squared_error'] = data.error**2
        data['covered'] = ((data.ci_lower <= data.true_effect) & (data.true_effect <= data.ci_upper)).astype(float)
        data['interval_width'] = data.ci_upper - data.ci_lower
    rows, byquery, perseed, comparisons = [], [], [], []
    stats_lookup = {}
    for source in SOURCES:
        si = inv[inv.probability_source.eq(source)]
        failed = int(si.status.eq('failed').sum())
        done = int(si.status.eq('complete').sum())
        complete = done == planned and not failed
        status = _status(complete, failed, fixture, done)
        for learner in LEARNERS:
            for procedure in PROCEDURES:
                meta = dict(dataset=cfg['dataset'], n_rows=n, learner=learner,
                            probability_source=source, method=procedure)
                g = data[(data.probability_source.eq(source)) & (data.learner.eq(learner)) & (data.method.eq(procedure))] if not data.empty else pd.DataFrame()
                common = dict(**meta, repetitions=done, planned_repetitions=planned,
                              failed_repetitions=failed, held_queries=Q, full_plan_complete=complete,
                              status=status, diagnostic_successful_subset_only=bool(failed and done),
                              units='profile' if cfg['dataset']=='phq9' else 'row')
                if g.empty:
                    rows.append(dict(**common, **_primary_and_interim({k:np.nan for k in METRICS}, False)))
                    continue
                sp = g.groupby('seed', sort=True).agg(mse=('squared_error','mean'),
                       coverage=('covered','mean'), width=('interval_width','mean'))
                if len(sp) != done:
                    raise IntegrityError('Inventory and completed score rows disagree.')
                stats_lookup[source, learner, procedure] = sp
                for seed, row in sp.iterrows():
                    perseed.append(dict(**meta, seed=int(seed), **row.to_dict()))
                values = _metrics(g, sp)
                rows.append(dict(**common, **_primary_and_interim(values, complete)))
                for qi, gg in g.groupby('query_index', sort=True):
                    sd = float(gg.error.std(ddof=1)) if done > 1 else np.nan
                    rmsse = float(np.sqrt(np.mean(gg.se**2)))
                    bias = float(gg.error.mean())
                    val = dict(bias=bias, bias_mcse=_mc(gg.error), empirical_sd=sd,
                               rms_se=rmsse, se_to_sd=rmsse/sd if np.isfinite(sd) and sd>0 else np.nan,
                               bias_squared_adjusted=bias*bias-sd*sd/done if done>1 else np.nan,
                               coverage=float(gg.covered.mean()), coverage_mcse=_mc(gg.covered),
                               mean_width=float(gg.interval_width.mean()), **rmse_stats(gg.squared_error))
                    byquery.append(dict(**meta, query_index=int(qi), query_id=str(gg.query_id.iloc[0]),
                                        repetitions=done, planned_repetitions=planned,
                                        full_plan_complete=complete, status=status,
                                        **_primary_and_interim(val, complete)))
            for pair, (numerator, denominator) in PAIRS.items():
                a = stats_lookup.get((source, learner, numerator))
                b = stats_lookup.get((source, learner, denominator))
                val = {key:np.nan for key in RATIO_METRICS}
                count = 0
                if a is not None and b is not None:
                    if not a.index.equals(b.index):
                        raise IntegrityError('Matched procedures have different successful seeds.')
                    count = len(a)
                    val = paired_ratio(a.mse, b.mse, family)
                    val.update(coverage_difference=float((a.coverage-b.coverage).mean()),
                               coverage_difference_mcse=_mc(a.coverage-b.coverage),
                               width_ratio=float(a.width.mean()/b.width.mean()) if b.width.mean()>0 else np.nan)
                comparisons.append(dict(dataset=cfg['dataset'], n_rows=n, learner=learner,
                    probability_source=source, pair=pair, numerator=numerator, denominator=denominator,
                    paired_repetitions=count, planned_repetitions=planned, full_plan_complete=complete,
                    failed_repetitions=failed, status=status, family_size=family,
                    comparison_family=pair+'_across_two_datasets_two_learners_four_sources',
                    role='primary_mechanism_comparison' if pair=='correction_projected' else 'secondary_mechanism_comparison',
                    **_primary_and_interim(val, complete)))
    write_frame(out/'by_setting.csv', pd.DataFrame(rows))
    write_frame(out/'by_query.csv', pd.DataFrame(byquery, columns=GROUP+['query_index','query_id','repetitions','planned_repetitions','full_plan_complete','status']+QUERY_METRICS+['interim_'+k for k in QUERY_METRICS]))
    write_frame(out/'per_seed.csv', pd.DataFrame(perseed, columns=GROUP+['seed','mse','coverage','width']))
    write_frame(out/'paired_comparisons.csv', pd.DataFrame(comparisons))
    pairs = pd.concat(pair_chunks, ignore_index=True) if pair_chunks else pd.DataFrame(columns=['dataset','n_rows','seed','learner','probability_source','pair','query_index','estimate_difference','se_difference'])
    if not pairs.empty:
        pairs['interpretation'] = 'descriptive paired score difference; nominal SE; not a validated Hausman test'
    write_frame(out/'paired_differences_all.csv', pairs)
    write_json(out/'report_status.json', dict(planned_repetitions=planned, processed_seed_slots=processed_slots,
        fixture=fixture, primary_family_size=family, query_independence_assumed=False,
        full_plan_metrics_require_all_planned_seeds=True,
        comparison_families='Each named pair has its own 16-comparison family; the union is not controlled.',
        uncertainty='Whole-seed Monte Carlo delta/log-t intervals; separate from nominal patient-score Wald intervals.',
        bias_squared_adjusted='Untruncated mean_error^2 - empirical_error_variance / completed_repetitions.'))
    text = _summary_text(out, brief=True)
    tmp = out/'summary.txt.tmp'
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(out/'summary.txt')
    return out


def _summary_text(out, brief=False):
    out = Path(out)
    cfg = read_json(out/'manifest.json')['config']
    stream = io.StringIO()
    tag = 'FICTIONAL SOFTWARE TEST' if cfg['fixture'] else 'MATCHED INTERACTION ABLATION'
    print(f"{cfg['dataset'].upper()} | {tag}", file=stream)
    print('Original targets; identical fitted nuisance functions within each 2x2 comparison.', file=stream)
    inv = pd.read_csv(out/'run_inventory.csv')
    print('\nAvailability (planned seeds):', file=stream)
    print(pd.crosstab(inv.probability_source, inv.status).to_string(), file=stream)
    d = pd.read_csv(out/'by_setting.csv')
    if brief:
        d = d[d.method.isin(['projected_full','projected_restricted']) & d.repetitions.gt(0)]
    cols = ['learner','probability_source','method','repetitions','rmse','coverage','interim_rmse','interim_coverage','status']
    print('\nPrimary fields remain NA until every planned seed completes. Interim fields are diagnostics.', file=stream)
    print(d.reindex(columns=cols).to_string(index=False, float_format=lambda x:f'{x:.5f}'), file=stream)
    pairs = pd.read_csv(out/'paired_comparisons.csv')
    pairs = pairs[pairs.pair.eq('correction_projected') & pairs.paired_repetitions.gt(0)]
    cols = ['learner','probability_source','paired_repetitions','ratio','family_low','family_high','interim_ratio','status']
    print('\nProjected outcome: restricted / full correction RMSE ratio (below 1 favors restriction):', file=stream)
    print(pairs.reindex(columns=cols).to_string(index=False, float_format=lambda x:f'{x:.5f}'), file=stream)
    print('\nMonte Carlo intervals use seeds, not queries, as independent replications.', file=stream)
    print('Each named comparison has a 16-comparison family; no joint claim across all families.', file=stream)
    print('Paired score SEs are descriptive nominal diagnostics, not validated Hausman tests.', file=stream)
    print('Do not select methods or stop a run based on interim performance.', file=stream)
    return stream.getvalue()


def display(out, brief=False):
    print(_summary_text(out, brief=brief), flush=True)
