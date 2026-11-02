"""Whole-seed Monte Carlo reporting. No query independence, no successful-subset main metrics."""
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from ext_support import read_json,write_frame,write_json,IntegrityError

NEW_NAMES=('aipw_multinomial_logit','aipw_hist_gradient_boosting','rieszboost_four_arm')
GROUP=['dataset','n_rows','learner','probability_source','method']
LABELS={
    'aipw_multinomial_logit':'AIPW: multinomial logistic',
    'aipw_hist_gradient_boosting':'AIPW: boosted propensity',
    'rieszboost_four_arm':'RieszBoost: four-arm adaptation',
    'A_full':'Full CAL-AIPW',
    'B_CONST':'Projected-outcome CAL-AIPW',
    'C_CONST':'Constant-interaction Riesz',
    'shared_sparse_riesz':'Shared sparse Riesz (internal)',
    'regularized_B1':'Pilot-corrected B1 (internal)',
    'oracle_AIPW':'Oracle reference (not feasible)'}


def _mc(x):
    a=np.asarray(x,float)
    return float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else np.nan


def rmse_stats(mse):
    x=np.asarray(mse,float);r=len(x)
    if r==0:return dict(rmse=np.nan,rmse_mcse=np.nan,rmse_ci_low=np.nan,rmse_ci_high=np.nan)
    avg=float(x.mean());v=float(np.sqrt(avg))
    if r<2:return dict(rmse=v,rmse_mcse=np.nan,rmse_ci_low=np.nan,rmse_ci_high=np.nan)
    if avg==0:return dict(rmse=0.,rmse_mcse=0.,rmse_ci_low=0.,rmse_ci_high=0.)
    se=_mc(x)/(2*v);se_log=_mc(x)/(2*avg);crit=float(student_t.ppf(.975,r-1))
    return dict(rmse=v,rmse_mcse=se,rmse_ci_low=float(v*np.exp(-crit*se_log)),
                rmse_ci_high=float(v*np.exp(crit*se_log)))


def paired_ratio(a,b,family=8):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape or len(a)<1:raise ValueError('Paired arrays must have equal nonzero lengths.')
    if a.mean()<=0 or b.mean()<=0:return dict(ratio=np.nan,ci_low=np.nan,ci_high=np.nan,family_low=np.nan,family_high=np.nan)
    est=float(np.sqrt(a.mean()/b.mean()))
    if len(a)<2:return dict(ratio=est,ci_low=np.nan,ci_high=np.nan,family_low=np.nan,family_high=np.nan)
    influence=.5*(a/a.mean()-b/b.mean());se=_mc(influence)
    z=float(student_t.ppf(.975,len(a)-1));zf=float(student_t.ppf(1-.05/(2*family),len(a)-1))
    return dict(ratio=est,ci_low=float(est*np.exp(-z*se)),ci_high=float(est*np.exp(z*se)),
                family_low=float(est*np.exp(-zf*se)),family_high=float(est*np.exp(zf*se)))


def add_errors(d):
    d=d.copy()
    d['error']=d.estimate-d.true_effect;d['squared_error']=d.error**2
    d['covered']=((d.ci_lower<=d.true_effect)&(d.true_effect<=d.ci_upper)).astype(float)
    d['interval_width']=d.ci_upper-d.ci_lower
    return d


def summarize(out):
    out=Path(out);mp=out/'manifest.json'
    if not mp.is_file():raise IntegrityError('No supplement manifest in '+str(out))
    manifest=read_json(mp);cfg=manifest['config']
    seeds=list(cfg['seeds']);n=cfg['n_rows'];Q=len(cfg['held_query_indices']);planned=len(seeds)
    inv=[];pieces=[];reference_pieces=[]
    for seed in seeds:
        slot=out/'runs'/f'n_{n}'/f'seed_{seed}'
        q=slot/'query_results.csv'
        d=pd.read_csv(q,dtype={'query_id':str},float_precision='round_trip') if q.is_file() else pd.DataFrame()
        present=set(d.method) if not d.empty else set()
        status={}
        if (slot/'method_inventory.csv').is_file():
            table=pd.read_csv(slot/'method_inventory.csv')
            status=dict(zip(table.method,table.status))
        for m in NEW_NAMES:
            state='complete' if m in present else 'failed' if status.get(m)=='failed' else 'not_processed'
            inv.append(dict(dataset=cfg['dataset'],n_rows=n,seed=seed,method=m,status=state))
        if not d.empty:pieces.append(d)
        q=slot/'reference_results.csv'
        if q.is_file():reference_pieces.append(pd.read_csv(q,dtype={'query_id':str},float_precision='round_trip'))
    inventory=pd.DataFrame(inv);write_frame(out/'run_inventory.csv',inventory)
    allparts=pieces+reference_pieces
    if not allparts:
        write_json(out/'report_status.json',dict(planned_repetitions=planned,processed_seed_slots=0,fixture=cfg['fixture']))
        return
    d=add_errors(pd.concat(allparts,ignore_index=True))
    key=GROUP+['seed','query_index']
    if d.duplicated(key).any():raise IntegrityError('Duplicate performance rows.')
    numeric=['estimate','true_effect','se','ci_lower','ci_upper']
    if not np.isfinite(d[numeric].to_numpy(float)).all() or (d.se<0).any():raise IntegrityError('Nonfinite/invalid result rows.')
    if not set(d.query_index.unique())==set(cfg['held_query_indices']):raise IntegrityError('Query catalog changed.')
    counts=d.groupby(GROUP+['seed']).size()
    if not counts.eq(Q).all():raise IntegrityError('A completed method/replication is missing queries.')
    rows=[];perseed=[];byquery=[]
    for names,g in d.groupby(GROUP,sort=False):
        meta=dict(zip(GROUP,names));R=g.seed.nunique();method=meta['method']
        failed=int(inventory[inventory.method.eq(method)].status.eq('failed').sum()) if method in NEW_NAMES else 0
        status=('FAILED_SLOTS_FULL_PLAN_NA' if failed else 'COMPLETE' if R==planned else 'INTERIM_NOT_CONFIRMATORY')
        if cfg['fixture']:status='FICTIONAL_SOFTWARE_TEST'
        sp=g.groupby('seed').agg(mse=('squared_error','mean'),coverage=('covered','mean'),width=('interval_width','mean'))
        for seed,r in sp.iterrows():perseed.append(dict(**meta,seed=int(seed),**r.to_dict()))
        m=rmse_stats(sp.mse)
        coverage=float(sp.coverage.mean());cse=_mc(sp.coverage);wse=_mc(sp.width)
        tc=float(student_t.ppf(.975,R-1)) if R>1 else np.nan
        sq=g.groupby('query_id').error
        ev=float(sq.var(ddof=1).mean()) if R>1 else np.nan
        rmsse=float(np.sqrt(np.mean(g.se**2)))
        result=dict(**meta,label=LABELS.get(method,method),repetitions=R,planned_repetitions=planned,
            failed_repetitions=failed,held_queries=Q,status=status,**m,
            mean_absolute_query_bias=float(sq.mean().abs().mean()),coverage=coverage,
            coverage_mcse=cse,coverage_ci_low=max(0.,coverage-tc*cse) if R>1 else np.nan,
            coverage_ci_high=min(1.,coverage+tc*cse) if R>1 else np.nan,
            mean_width=float(sp.width.mean()),width_mcse=wse,
            empirical_sd=float(np.sqrt(ev)),rms_reported_se=rmsse,
            se_to_sd=float(rmsse/np.sqrt(ev)) if np.isfinite(ev) and ev>0 else np.nan,
            units='profile' if cfg['dataset']=='phq9' else 'row',
            role='external_comparator' if method in NEW_NAMES else 'archived_reference')
        if failed:
            for k in list(result):
                if k in ('rmse','rmse_mcse','rmse_ci_low','rmse_ci_high','mean_absolute_query_bias','coverage','coverage_mcse','coverage_ci_low','coverage_ci_high','mean_width','width_mcse','empirical_sd','rms_reported_se','se_to_sd'):result[k]=np.nan
        rows.append(result)
        for qi,gg in g.groupby('query_index',sort=False):
            sd=float(gg.estimate.std(ddof=1));rmsse=float(np.sqrt(np.mean(gg.se**2)))
            one=dict(**meta,query_index=int(qi),query_id=str(gg.query_id.iloc[0]),repetitions=len(gg),status=status,
                     bias=float(gg.error.mean()),bias_mcse=_mc(gg.error),coverage=float(gg.covered.mean()),
                     empirical_sd=sd,rms_se=rmsse,
                     se_to_sd=rmsse/sd if np.isfinite(sd) and sd>0 else np.nan,
                     coverage_mcse=_mc(gg.covered),mean_width=float(gg.interval_width.mean()),**rmse_stats(gg.squared_error))
            if failed:
                for k in ('bias','bias_mcse','coverage','coverage_mcse','mean_width','rmse','rmse_mcse','rmse_ci_low','rmse_ci_high','empirical_sd','rms_se','se_to_sd'):one[k]=np.nan
            byquery.append(one)
    # Add explicit unavailable rows even if a new method has no successful seed.
    present={(r['method'],r['learner']) for r in rows}
    for method in NEW_NAMES:
        for learner in ('linear','flexible'):
            if (method,learner) in present:continue
            ii=inventory[inventory.method.eq(method)];fail=int(ii.status.eq('failed').sum())
            rows.append(dict(dataset=cfg['dataset'],n_rows=n,method=method,learner=learner,label=LABELS[method],
                probability_source={'aipw_multinomial_logit':'multinomial_logit','aipw_hist_gradient_boosting':'hist_gradient_boosting','rieszboost_four_arm':'direct_rieszboost'}[method],
                repetitions=0,planned_repetitions=planned,failed_repetitions=fail,held_queries=Q,
                status='FAILED_SLOTS_FULL_PLAN_NA' if fail else 'NOT_YET_PROCESSED',role='external_comparator'))
    write_frame(out/'by_setting.csv',pd.DataFrame(rows))
    write_frame(out/'per_seed.csv',pd.DataFrame(perseed))
    write_frame(out/'by_query.csv',pd.DataFrame(byquery))
    ps=pd.DataFrame(perseed);pairs=[]
    for learner in ('linear','flexible'):
        for cal in ('cal_linear','cal_sparse'):
            for numerator in ('C_CONST','A_full','B_CONST'):
                aa=ps[(ps.learner==learner)&(ps.probability_source==cal)&(ps.method==numerator)].set_index('seed')
                if aa.empty:continue
                for comparator in NEW_NAMES:
                    bb=ps[(ps.learner==learner)&(ps.method==comparator)].set_index('seed')
                    joined=aa[['mse','coverage','width']].join(bb[['mse','coverage','width']],how='inner',lsuffix='_num',rsuffix='_den')
                    fail=int(inventory[inventory.method.eq(comparator)].status.eq('failed').sum())
                    values=paired_ratio(joined.mse_num,joined.mse_den,8) if len(joined) and not fail else dict(ratio=np.nan,ci_low=np.nan,ci_high=np.nan,family_low=np.nan,family_high=np.nan)
                    complete=len(joined)==planned and not fail
                    # All 48 targets must agree before any comparison is interpreted.
                    status='COMPLETE' if complete else 'FAILED_SLOTS_FULL_PLAN_NA' if fail else 'INTERIM_NOT_CONFIRMATORY'
                    if cfg['fixture']:status='FICTIONAL_SOFTWARE_TEST'
                    if numerator!='C_CONST':values['family_low']=np.nan;values['family_high']=np.nan
                    pairs.append(dict(dataset=cfg['dataset'],n_rows=n,learner=learner,archived_CAL=cal,
                        numerator=numerator,denominator=comparator,paired_repetitions=len(joined),
                        planned_repetitions=planned,comparison_type='pipeline_performance_not_pure_ablation',
                        family_size=8 if numerator=='C_CONST' else 1,status=status,**values,
                        coverage_difference=float((joined.coverage_num-joined.coverage_den).mean()) if len(joined) and not fail else np.nan,
                        coverage_difference_mcse=_mc(joined.coverage_num-joined.coverage_den) if len(joined) and not fail else np.nan,
                        width_ratio=float(joined.width_num.mean()/joined.width_den.mean()) if len(joined) and not fail and joined.width_den.mean()>0 else np.nan))
    write_frame(out/'paired_comparisons.csv',pd.DataFrame(pairs))
    write_json(out/'report_status.json',dict(planned_repetitions=planned,processed_seed_slots=int(inventory.seed[inventory.status.isin(['complete','failed','not_processed'])].nunique()),
        fixture=cfg['fixture'],query_independence_assumed=False,primary_family_size=8,
        calibration_ratio='se_to_sd = RMS reported SE / empirical Monte Carlo SD; below 1 means smaller reported SE',
        uncertainty='whole-seed paired delta/log-t; nominal performance MC intervals, not causal effect confidence bands',
        new_comparators_use_known_marginal_pilot=False,archived_procedures_keep_their_original_information_access=True))


def display(out,brief=False):
    out=Path(out);cfg=read_json(out/'manifest.json')['config']
    print('\n'+('='*100))
    print(('FICTIONAL SOFTWARE TEST | ' if cfg['fixture'] else '')+cfg['dataset'].upper()+' | EXTERNAL BASELINE SUPPLEMENT')
    print('Same original DGP/queries/folds; original results are read-only. Pointwise score-Wald coverage is evaluated, not guaranteed.')
    inv=pd.read_csv(out/'run_inventory.csv')
    print('\nMETHOD AVAILABILITY (planned seeds, not queries)')
    print(pd.crosstab(inv.method,inv.status).to_string())
    q=out/'by_setting.csv'
    if q.exists():
        d=pd.read_csv(q)
        print('\nORIGINAL-TARGET EFFECT RESULTS (NA means unavailable, never zero)')
        cols=['learner','probability_source','label','repetitions','rmse','rmse_mcse','mean_absolute_query_bias','coverage','mean_width','status']
        for learner in ('linear','flexible'):
            print('\nOutcome = '+learner)
            take=d[d.learner.eq(learner)]
            if brief:take=take[take.method.isin(list(NEW_NAMES)+['C_CONST','shared_sparse_riesz'])]
            print(take.reindex(columns=cols).to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    q=out/'paired_comparisons.csv'
    if q.exists() and q.stat().st_size>1:
        d=pd.read_csv(q)
        if not d.empty:
            print('\nC_CONST / NEW EXTERNAL COMPARATOR (ratio < 1 favors C_CONST)')
            print(d[d.numerator.eq('C_CONST')][['learner','archived_CAL','denominator','paired_repetitions','ratio','ci_low','ci_high','family_low','family_high','status']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nReferences retain their original known-marginal-pilot access; no equal-information or equal-compute claim.')
    print('RieszBoost is an independently tuned four-arm Python adaptation of Algorithm 2, not the authors\' package.')
    print('Any partial run is interim. Do not stop or choose methods based on these interim scores.')
