"""Original-target performance and paired whole-replication comparisons.

The outcome learner and probability source are never silently collapsed.
No best-model flag or per-query oracle choice is generated.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from io_utils import write_csv

KEYS = ['dataset','n_rows','independent_units','learner','probability_source','stage','structure','method']


def bootstrap_interval(a,b,kind,draws=2000,seed=902137):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<2:return np.nan,np.nan
    ix=np.random.default_rng(seed).integers(0,len(a),(draws,len(a)))
    av=a[ix].mean(1);bv=b[ix].mean(1)
    if kind=='difference':v=av-bv
    else:
        v=np.divide(av,bv,out=np.full_like(av,np.nan),where=bv>0)
        if kind=='rmse_ratio':v=np.sqrt(v)
    v=v[np.isfinite(v)]
    if not len(v):return np.nan,np.nan
    return tuple(np.quantile(v,[.025,.975]))


def add_errors(d):
    d=d.copy()
    d['error']=d.estimate-d.true_effect
    d['sqerr']=d.error**2
    d['covered']=(d.ci_lower<=d.true_effect)&(d.true_effect<=d.ci_upper)
    d['width']=d.ci_upper-d.ci_lower
    return d


def _mcse(x):
    return float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan


def summarize_group(gg,names,planned):
    perquery=gg.groupby('query_id',sort=False)
    seedstats=gg.groupby('seed').agg(mse=('sqerr','mean'),coverage=('covered','mean'),width=('width','mean'))
    empirical_var=perquery.estimate.var(ddof=1).mean()
    rmsse=float(np.sqrt(np.mean(gg.se**2)))
    row=dict(**names,repetitions=int(gg.seed.nunique()),planned_repetitions=planned,
        held_queries=int(gg.query_id.nunique()),
        pooled_effect_rmse=float(np.sqrt(gg.sqerr.mean())),
        mean_absolute_query_bias=float(perquery.error.mean().abs().mean()),
        coverage=float(gg.covered.mean()),coverage_mcse=_mcse(seedstats.coverage.to_numpy()),
        mean_width=float(gg.width.mean()),width_mcse=_mcse(seedstats.width.to_numpy()),
        empirical_sd=float(np.sqrt(empirical_var)),rms_reported_se=rmsse,
        se_to_sd=float(rmsse/np.sqrt(empirical_var)) if np.isfinite(empirical_var) and empirical_var>0 else np.nan,
        unavailable_repetitions=planned-int(gg.seed.nunique()),
        summary_scope='full_plan' if gg.seed.nunique()==planned else 'completed_seeds_only',
        representer_mse=float(gg.representer_mse.mean()),outcome_mse=float(gg.outcome_mse.mean()),
        conditional_bias_rms=float(np.sqrt(np.mean(gg.remainder**2))),
        alpha_second_moment=float(gg.alpha_second_moment.mean()),
        max_abs_alpha_eval=float(gg.max_abs_alpha_eval.max()),
        outcome_change_mse=float(gg.outcome_change_mse.mean()),
        interaction_error_j=float(gg.interaction_error_j.mean()),
        interaction_error_raw=float(gg.interaction_error_raw.mean()),
        interval_scope='empirical_pointwise_profile_score_Wald_not_a_guarantee',
        inference_unit='sampled_profile' if str(names.get('dataset'))=='phq9' else 'sampled_row')
    return row


def make_comparisons(d,planned,by_family=False):
    result=[]
    group_cols=['dataset','n_rows','learner','probability_source']
    if by_family:group_cols+=['contrast_family']
    feasible=d[d.stage.isin(['A','B','C'])]
    for vals,gg in feasible.groupby(group_cols,sort=False):
        names=dict(zip(group_cols,vals))
        if 'A_full' not in set(gg.method):continue
        for st in ['CONST','LOWDIM','NO_INT']:
            pairs=[('B_'+st,'A_full','B_vs_A_response_projection'),
                   ('C_'+st,'B_'+st,'C_vs_B_correction_only'),
                   ('C_'+st,'A_full','C_vs_A_total_change')]
            for ma,mb,label in pairs:
                aa=gg[gg.method.eq(ma)].groupby('seed').agg(mse=('sqerr','mean'),
                    coverage=('covered','mean'),width=('width','mean'),g=('outcome_mse','mean'))
                bb=gg[gg.method.eq(mb)].groupby('seed').agg(mse=('sqerr','mean'),
                    coverage=('covered','mean'),width=('width','mean'),g=('outcome_mse','mean'))
                zz=aa.join(bb,how='inner',lsuffix='_a',rsuffix='_b')
                if zz.empty:continue
                for metric,col,kind in [('effect_rmse_ratio','mse','rmse_ratio'),
                    ('coverage_difference','coverage','difference'),('width_ratio','width','ratio'),
                    ('outcome_mse_ratio','g','ratio')]:
                    a=zz[col+'_a'].to_numpy();b=zz[col+'_b'].to_numpy()
                    if kind=='difference':est=float(np.mean(a-b))
                    elif b.mean()>0:est=float(np.mean(a)/np.mean(b));est=np.sqrt(est) if kind=='rmse_ratio' else est
                    else:est=np.nan
                    lo,hi=bootstrap_interval(a,b,kind)
                    result.append(dict(**names,structure=st,numerator=ma,denominator=mb,
                        comparison=label,metric=metric,estimate=est,ci_low=lo,ci_high=hi,
                        paired_repetitions=len(zz),planned_repetitions=planned,
                        bootstrap_unit='whole_seed',exploratory=True,multiplicity_adjusted=False))
    return result


def summarize(out):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    cfg=manifest['config'];inventory=[];method_inventory=[];paths=[];overs=[];checks=[];gdiags=[]
    for n in cfg['n_rows']:
        for seed in cfg['seeds']:
            slot=out/'runs'/f'n_{n}'/f'seed_{seed}'
            status='not_complete';reason=''
            if (slot/'query_results.csv').exists():
                status='complete';paths.append(slot/'query_results.csv')
                if (slot/'method_inventory.csv').exists():
                    method_inventory.append(pd.read_csv(slot/'method_inventory.csv'))
                if (slot/'overlap_diagnostics.csv').exists():overs.append(pd.read_csv(slot/'overlap_diagnostics.csv'))
                for fol in sorted(slot.glob('fold_*')):
                    for filename,arr in [('projection_checks.csv',checks),('projection_gram_diagnostics.csv',gdiags)]:
                        f=fol/filename
                        if f.exists():
                            v=pd.read_csv(f);v['dataset']=cfg['dataset'];v['n_rows']=n;v['seed']=seed;arr.append(v)
            elif (slot/'failed.json').exists():
                status='failed';reason=json.loads((slot/'failed.json').read_text()).get('type','')
            inventory.append(dict(dataset=cfg['dataset'],n_rows=n,seed=seed,status=status,failure_type=reason))
    write_csv(out/'run_inventory.csv',inventory)
    if method_inventory:
        write_csv(out/'method_inventory.csv',pd.concat(method_inventory,ignore_index=True).to_dict('records'))
    if not paths:return
    d=add_errors(pd.concat([pd.read_csv(p) for p in paths],ignore_index=True))
    unique=KEYS+['seed','query_id']
    if d.duplicated(unique).any():raise ValueError('Duplicate method/source/learner/seed/query output.')
    planned=len(cfg['seeds'])
    settings=[];queries=[];contrasts=[]
    for vals,gg in d.groupby(KEYS,sort=False):
        names=dict(zip(KEYS,vals));settings.append(summarize_group(gg,names,planned))
        for qid,g in gg.groupby('query_id',sort=False):
            sd=float(g.estimate.std(ddof=1));rmsse=float(np.sqrt(np.mean(g.se**2)))
            queries.append(dict(**names,query_id=qid,query_index=int(g.query_index.iloc[0]),
                contrast_family=str(g.contrast_family.iloc[0]),repetitions=len(g),
                true_effect=float(g.true_effect.iloc[0]),bias=float(g.error.mean()),
                bias_mcse=_mcse(g.error.to_numpy()),effect_rmse=float(np.sqrt(g.sqerr.mean())),
                empirical_sd=sd,mean_se=float(g.se.mean()),rms_se=rmsse,
                se_to_sd=rmsse/sd if np.isfinite(sd) and sd>0 else np.nan,
                planned_repetitions=planned,unavailable_repetitions=planned-len(g),
                summary_scope='full_plan' if len(g)==planned else 'completed_seeds_only',
                coverage=float(g.covered.mean()),mean_width=float(g.width.mean()),
                representer_mse=float(g.representer_mse.mean()),outcome_mse=float(g.outcome_mse.mean()),
                remainder=float(g.remainder.mean()),conditional_bias_rms=float(np.sqrt(np.mean(g.remainder**2))),
                outcome_change_mse=float(g.outcome_change_mse.mean())))
        for family,g in gg.groupby('contrast_family',sort=False):
            contrasts.append(summarize_group(g,dict(**names,contrast_family=family),planned))
    write_csv(out/'by_setting.csv',settings)
    write_csv(out/'by_query.csv',queries)
    write_csv(out/'by_contrast.csv',contrasts)
    write_csv(out/'paired_comparisons.csv',make_comparisons(d,planned))
    write_csv(out/'paired_by_contrast.csv',make_comparisons(d,planned,True))
    # Reference comparisons use matched seeds with a complete optional baseline.
    refs=[]
    for (dataset,n,learner),gg in d[d.stage.ne('ORACLE')].groupby(['dataset','n_rows','learner'],sort=False):
        for source in ['cal_linear','cal_sparse']:
            for method in ['A_full','B_CONST','B_LOWDIM','C_CONST','C_LOWDIM']:
                aa=gg[(gg.probability_source==source)&(gg.method==method)].groupby('seed').sqerr.mean()
                for ref in ['regularized_B1','shared_sparse_riesz']:
                    bb=gg[gg.method==ref].groupby('seed').sqerr.mean();z=pd.concat([aa,bb],axis=1,keys=['a','b']).dropna()
                    if z.empty:continue
                    lo,hi=bootstrap_interval(z.a.to_numpy(),z.b.to_numpy(),'rmse_ratio')
                    refs.append(dict(dataset=dataset,n_rows=n,learner=learner,probability_source=source,
                        method=method,reference=ref,effect_rmse_ratio=float(np.sqrt(z.a.mean()/z.b.mean())),
                        ci_low=lo,ci_high=hi,paired_repetitions=len(z),planned_repetitions=planned,
                        numerator_repetitions=len(aa),reference_repetitions=len(bb),
                        comparison_scope='performance_reference_full_plan' if len(z)==planned else 'performance_reference_completed_pairs_only'))
    write_csv(out/'reference_comparisons.csv',refs)
    if overs:
        v=pd.concat(overs,ignore_index=True)
        cols=['population_mass','alpha_error_contribution','alpha2_contribution','outcome_error_contribution']
        g=v.groupby(['dataset','n_rows','learner','probability_source','stage','structure','method','overlap_bin'],sort=False)[cols].mean().reset_index()
        write_csv(out/'by_overlap_diagnostics.csv',g.to_dict('records'))
    if checks:write_csv(out/'projection_checks.csv',pd.concat(checks,ignore_index=True).to_dict('records'))
    if gdiags:write_csv(out/'projection_gram_diagnostics.csv',pd.concat(gdiags,ignore_index=True).to_dict('records'))
    # Minimal per-seed view, including oracle. Never average n or learner together.
    z=d.groupby(KEYS+['seed'],sort=False).agg(mse=('sqerr','mean'),coverage=('covered','mean'),
        width=('width','mean'),rmsse=('se',lambda x:float(np.sqrt(np.mean(x*x))))).reset_index()
    z['effect_rmse']=np.sqrt(z.pop('mse'))
    write_csv(out/'per_seed.csv',z.to_dict('records'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('result_directory')
    summarize(parser.parse_args().result_directory)
