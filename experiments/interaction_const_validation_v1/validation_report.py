#!/usr/bin/env python3
"""Summarize stored seed/query results with Monte Carlo uncertainty.

Never treats queries as independent repetitions. Results stay relative to the original
target. The statistical calculations here do not change any fitted estimator or CI.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, norm
from result_io import ROOT,read_json,write_json

GROUP=['dataset','n_rows','learner','probability_source','method']
KEY=GROUP+['seed','query_id']
FAMILIES=[('primary_correction','B_CONST','same'),
          ('secondary_total','A_full','same'),
          ('secondary_shared_sparse','shared_sparse_riesz','legacy'),
          ('secondary_B1','regularized_B1','legacy')]
ALPHA=.05
FAMILY_SIZE=8  # two data sets x two learners x two propensity backends

def write_frame(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)

def mc_mean(x,alpha=.05,bounded=False):
    x=np.asarray(x,float);n=len(x)
    if not n or not np.isfinite(x).all():raise ValueError('Finite, nonempty MC sample required.')
    mean=float(x.mean())
    if n<2:return mean,np.nan,np.nan,np.nan
    se=float(x.std(ddof=1)/np.sqrt(n));crit=student_t.ppf(1-alpha/2,n-1)
    lo,hi=mean-crit*se,mean+crit*se
    if bounded:lo,hi=max(0.,lo),min(1.,hi)
    return mean,se,float(lo),float(hi)

def paired_ratio(x,y,power=.5,draws=5000,rng_seed=770231):
    """Ratio of averages with paired MC influence functions and seed bootstrap.
    For MSE inputs, power=.5 produces RMSE ratio (NOT average seedwise ratios).
    """
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=1 or x.shape!=y.shape or len(x)==0 or not np.isfinite(x+y).all():
        raise ValueError('Paired finite one-dimensional arrays required.')
    if (x<0).any() or (y<0).any():raise ValueError('Nonnegative risks/widths required.')
    n=len(x);mx,my=x.mean(),y.mean()
    keys=['ratio','logratio_mcse','ci_low','ci_high','family_ci_low','family_ci_high',
          'bootstrap_ci_low','bootstrap_ci_high','paired_p_two_sided']
    out={k:np.nan for k in keys}
    out.update(paired_repetitions=n,seed_win_fraction=float(np.mean(x<y)))
    if my<=0:return out
    out['ratio']=float((mx/my)**power)
    if n<2 or mx<=0:return out
    center=power*np.log(mx/my)
    influence=power*((x-mx)/mx-(y-my)/my)
    se=float(np.std(influence,ddof=1)/np.sqrt(n));out['logratio_mcse']=se
    for name,alpha in [('ci',ALPHA),('family_ci',ALPHA/FAMILY_SIZE)]:
        crit=student_t.ppf(1-alpha/2,n-1)
        out[name+'_low']=float(np.exp(np.clip(center-crit*se,-745,709)))
        out[name+'_high']=float(np.exp(np.clip(center+crit*se,-745,709)))
    if se>0:out['paired_p_two_sided']=float(2*student_t.sf(abs(center/se),n-1))
    # The t/delta and bootstrap intervals are asymptotic MC approximations.
    rng=np.random.default_rng(rng_seed);vals=[]
    for start in range(0,draws,250):
        ix=rng.integers(0,n,(min(250,draws-start),n))
        den=y[ix].mean(1);num=x[ix].mean(1);ok=den>0
        vals.extend(((num[ok]/den[ok])**power).tolist())
    if vals:
        out['bootstrap_ci_low'],out['bootstrap_ci_high']=map(float,np.quantile(vals,[.025,.975]))
    return out

def wilson(k,n,alpha=.05):
    if n<1:return np.nan,np.nan
    z=norm.ppf(1-alpha/2);p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,float(mid-half)),min(1.,float(mid+half))

def load_runs(out,exploratory=False):
    """Read available per-seed result tables using the recorded run configuration."""
    out=Path(out);m=read_json(out/'manifest.json');cfg=m['config']
    dataset=cfg['dataset'];smoke=bool(cfg.get('smoke'))
    if dataset not in ('cvd','phq9'):raise ValueError('Unknown data set.')
    catalog=pd.read_csv(out/'query_catalog.csv',dtype={'query_id':str})
    qids=set(catalog.loc[catalog.query_split.eq('held_out'),'query_id'])
    if len(qids)!=48:raise ValueError('Expected original 48 held queries; target set changed.')
    records=[];pieces=[]
    for n in cfg['n_rows']:
        for seed in cfg['seeds']:
            slot=out/'runs'/f'n_{n}'/f'seed_{seed}'
            row=dict(dataset=dataset,n_rows=n,seed=seed,status='not_complete')
            if (slot/'query_results.csv').is_file():
                row['status']='complete'
                d=pd.read_csv(slot/'query_results.csv',dtype={'query_id':str})
                if not d.dataset.eq(dataset).all() or not d.n_rows.eq(n).all() or not d.seed.eq(seed).all():
                    raise ValueError('Seed/dataset metadata do not match directory.')
                validate_rows(d,qids,with_oracle=not cfg.get('no_oracle_reference',False))
                pieces.append(d)
            elif (slot/'failed.json').exists():
                row['status']='failed';row['failure_type']=read_json(slot/'failed.json').get('type','unavailable')
            records.append(row)
    d=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
    inventory=pd.DataFrame(records)
    return d,inventory,cfg,smoke

def validate_rows(d,qids,with_oracle=True):
    if d.duplicated(KEY).any():raise ValueError('Duplicate method/seed/query result.')
    vals=d[['estimate','se','ci_lower','ci_upper','true_effect']].to_numpy(float)
    if not np.isfinite(vals).all() or (d.se<0).any():raise ValueError('Invalid effect/SE values.')
    if (d.ci_upper<d.ci_lower).any():raise ValueError('Reversed interval.')
    if not np.allclose(d.ci_upper-d.ci_lower,2*norm.ppf(.975)*d.se,rtol=1e-9,atol=1e-12):
        raise ValueError('Original score-Wald intervals changed.')
    per=d.groupby(['learner','probability_source','method'],sort=False)
    core_methods=('A_full','B_CONST','C_CONST','B_LOWDIM','C_LOWDIM','B_NO_INT','C_NO_INT')
    required={(learner,source,method) for learner in ('linear','flexible')
              for source in ('cal_linear','cal_sparse') for method in core_methods}
    if with_oracle:required.add(('oracle','oracle','oracle_AIPW'))
    legacy={(learner,'legacy',method) for learner in ('linear','flexible')
            for method in ('regularized_B1','shared_sparse_riesz')}
    present=set(per.groups)
    if not required.issubset(present) or present-(required|legacy):
        raise ValueError('Required A/B/C, controls, or oracle omitted, or unknown method.')
    for method in ('regularized_B1','shared_sparse_riesz'):
        expected={(learner,'legacy',method) for learner in ('linear','flexible')}
        if present&expected and not expected.issubset(present):
            raise ValueError('Incomplete optional baseline learners: '+method)
    for key,g in per:
        if set(g.query_id)!=qids or len(g)!=48:raise ValueError('Missing query or duplicate: '+str(key))
    expected_units=np.where(d.dataset.eq('phq9'),d.n_rows/4,d.n_rows)
    if not np.array_equal(d.independent_units.to_numpy(),expected_units):raise ValueError('Wrong independent units.')
    if not d.query_split.eq('held_out').all():raise ValueError('Unexpected query split.')
    if d.groupby('query_id').true_effect.nunique().max()!=1:raise ValueError('Methods use different truths.')
    for (_,_,),g in d[d.stage.isin(['A','B','C'])].groupby(['learner','probability_source']):
        aa=g[g.method.eq('A_full')].set_index('query_id').sort_index()
        for st in ['CONST','LOWDIM','NO_INT']:
            bb=g[g.method.eq('B_'+st)].set_index('query_id').sort_index()
            cc=g[g.method.eq('C_'+st)].set_index('query_id').sort_index()
            for col in ['outcome_mse','outcome_change_mse']:
                if not np.array_equal(bb[col],cc[col]):raise ValueError('B/C outcome not identical.')
            if not np.array_equal(aa.representer_mse,bb.representer_mse):raise ValueError('B full correction differs from A.')
            if aa.learner.iloc[0]=='linear' and st!='NO_INT':
                if not np.allclose(aa[['estimate','se']],bb[['estimate','se']],rtol=1e-10,atol=2e-11):
                    raise ValueError('Linear B=A identity failed.')

def add_errors(d):
    d=d.copy();d['error']=d.estimate-d.true_effect;d['sqerr']=d.error*d.error
    d['covered']=((d.ci_lower<=d.true_effect)&(d.true_effect<=d.ci_upper)).astype(float)
    d['width']=d.ci_upper-d.ci_lower
    return d

def summary_tables(d,planned):
    settings=[];queries=[];seedrows=[];families=[]
    def aggregate(g,names):
        per=g.groupby('seed').agg(mse=('sqerr','mean'),coverage=('covered','mean'),width=('width','mean'))
        cov,se,lo,hi=mc_mean(per.coverage,bounded=True)
        mse,mse_se,mse_lo,mse_hi=mc_mean(per.mse)
        _,width_se,width_lo,width_hi=mc_mean(per.width)
        rmse_se=(mse_se/(2*np.sqrt(mse))) if mse>0 and np.isfinite(mse_se) else np.nan
        v=g.groupby('query_id').estimate.var(ddof=1).mean()
        rmsse=float(np.sqrt(np.mean(g.se*g.se)))
        return dict(**names,repetitions=len(per),planned_repetitions=planned,queries=g.query_id.nunique(),
            effect_rmse=float(np.sqrt(g.sqerr.mean())),effect_rmse_mcse=float(rmse_se),
            effect_rmse_mc_low=float(np.sqrt(max(0.,mse_lo))) if np.isfinite(mse_lo) else np.nan,
            effect_rmse_mc_high=float(np.sqrt(max(0.,mse_hi))) if np.isfinite(mse_hi) else np.nan,
            mean_absolute_query_bias=float(g.groupby('query_id').error.mean().abs().mean()),
            coverage=cov,coverage_mcse=se,coverage_mc_low=lo,coverage_mc_high=hi,
            mean_width=float(g.width.mean()),width_mcse=width_se,width_mc_low=width_lo,width_mc_high=width_hi,empirical_sd=float(np.sqrt(v)),rms_se=rmsse,
            se_to_sd=float(rmsse/np.sqrt(v)) if np.isfinite(v) and v>0 else np.nan,
            unavailable_repetitions=planned-len(per),
            summary_scope='full_plan' if len(per)==planned else 'completed_seeds_only',
            min_query_coverage=float(g.groupby('query_id').covered.mean().min()),
            fraction_queries_coverage_below_90=float((g.groupby('query_id').covered.mean()<.90).mean()),
            MC_upper_below_nominal_95=bool(np.isfinite(hi) and hi<.95),
            target='original',coverage_claim='not_certified',
            independent_unit='profile' if names['dataset']=='phq9' else 'row')
    for key,g in d.groupby(GROUP,sort=False):
        names=dict(zip(GROUP,key));settings.append(aggregate(g,names))
        for seed,s in g.groupby('seed'):
            seedrows.append(dict(**names,seed=seed,mse=float(s.sqerr.mean()),coverage=float(s.covered.mean()),
                                 width=float(s.width.mean()),rmse=float(np.sqrt(s.sqerr.mean()))))
        for family,s in g.groupby('contrast_family',sort=False):
            families.append(aggregate(s,dict(**names,contrast_family=family)))
        for q,s in g.groupby('query_id',sort=False):
            n=len(s);k=int(s.covered.sum());lo,hi=wilson(k,n);sl,sh=wilson(k,n,alpha=.05/48)
            sd=float(s.estimate.std(ddof=1));rmsse=float(np.sqrt(np.mean(s.se*s.se)))
            queries.append(dict(**names,query_id=q,contrast_family=s.contrast_family.iloc[0],repetitions=n,
                true_effect=float(s.true_effect.iloc[0]),bias=float(s.error.mean()),
                bias_mcse=float(s.error.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan,
                effect_rmse=float(np.sqrt(s.sqerr.mean())),coverage=k/n,
                coverage_mcse=float(np.sqrt((k/n)*(1-k/n)/n)),
                coverage_wilson_low=lo,coverage_wilson_high=hi,
                coverage_queryfamily_low=sl,coverage_queryfamily_high=sh,
                mean_width=float(s.width.mean()),empirical_sd=sd,rms_se=rmsse,
                se_to_sd=rmsse/sd if np.isfinite(sd) and sd>0 else np.nan,
                planned_repetitions=planned,unavailable_repetitions=planned-n,
                summary_scope='full_plan' if n==planned else 'completed_seeds_only'))
    return (pd.DataFrame(settings),pd.DataFrame(queries),pd.DataFrame(seedrows),pd.DataFrame(families))

def comparison_table(seedstats,by_family=False):
    rows=[];grp=['dataset','n_rows','learner','probability_source']
    for key,g in seedstats[seedstats.probability_source.isin(['cal_linear','cal_sparse'])].groupby(grp,sort=False):
        names=dict(zip(grp,key));aa=g[g.method.eq('C_CONST')].set_index('seed')
        for family,ref,where in FAMILIES:
            if where=='same':bb=g[g.method.eq(ref)].set_index('seed')
            else:
                bb=seedstats[(seedstats.dataset==names['dataset'])&(seedstats.n_rows==names['n_rows'])&
                 (seedstats.learner==names['learner'])&(seedstats.probability_source=='legacy')&
                 (seedstats.method==ref)].set_index('seed')
            if aa.empty or (where=='same' and set(aa.index)!=set(bb.index)):
                raise ValueError('Unpaired core seed comparison.')
            numerator_repetitions=len(aa);reference_repetitions=len(bb)
            common=aa.index.intersection(bb.index,sort=False)
            pair_a=aa.loc[common];pair_b=bb.loc[common]
            pair_scope=('all_available_core_seeds' if len(common)==len(aa) else
                        'completed_pairs_only' if len(common) else 'unavailable')
            # This deterministic bootstrap seed is unrelated to file verification.
            rngseed=int(hashlib.sha256((str(key)+family).encode()).hexdigest()[:8],16)
            if len(common):
                stats=paired_ratio(pair_a.mse.to_numpy(),pair_b.mse.to_numpy(),rng_seed=rngseed)
                coverage,mcse,cl,ch=mc_mean((pair_a.coverage-pair_b.coverage).to_numpy())
                width=paired_ratio(pair_a.width.to_numpy(),pair_b.width.to_numpy(),power=1.,draws=5000,rng_seed=rngseed+1)
            else:
                stats={key:np.nan for key in ('ratio','logratio_mcse','ci_low','ci_high',
                       'family_ci_low','family_ci_high','bootstrap_ci_low','bootstrap_ci_high',
                       'paired_p_two_sided','seed_win_fraction')}
                stats['paired_repetitions']=0
                coverage=mcse=cl=ch=np.nan
                width={'ratio':np.nan,'ci_low':np.nan,'ci_high':np.nan}
            rows.append(dict(**names,comparison_family=family,numerator='C_CONST',denominator=ref,
                **stats,coverage_difference=coverage,coverage_difference_mcse=mcse,
                coverage_difference_ci_low=cl,coverage_difference_ci_high=ch,
                width_ratio=width['ratio'],width_ratio_ci_low=width['ci_low'],width_ratio_ci_high=width['ci_high'],
                MC_family_size=8,interval_type='paired_MC_delta_log_t_and_Bonferroni8',
                bootstrap_unit='whole_seed',no_coverage_validity_claim=True,
                numerator_repetitions=numerator_repetitions,reference_repetitions=reference_repetitions,
                comparison_scope=pair_scope))
    return pd.DataFrame(rows)

def report(out,exploratory=False):
    out=Path(out).expanduser().resolve()
    d,inventory,cfg,smoke=load_runs(out,exploratory)
    dest=out/'verification';dest.mkdir(exist_ok=True)
    write_frame(dest/'run_inventory.csv',inventory)
    method_records=[]
    for n in cfg['n_rows']:
        for seed in cfg['seeds']:
            path=out/'runs'/f'n_{n}'/f'seed_{seed}'/'method_inventory.csv'
            if path.is_file():method_records.append(pd.read_csv(path))
    method_inventory=pd.concat(method_records,ignore_index=True) if method_records else pd.DataFrame()
    if not method_inventory.empty:write_frame(dest/'method_inventory.csv',method_inventory)
    planned=len(cfg['seeds']);counts=inventory.groupby('n_rows').status.apply(lambda x:int((x=='complete').sum())).to_dict()
    full=all(v==planned for v in counts.values())
    state='SOFTWARE_FIXTURE' if smoke else ('EXPLORATORY_OLD_OUTPUT' if exploratory else ('COMPLETE_FROZEN_REPLICATION' if full else 'INTERIM_NOT_CONFIRMATORY'))
    status=dict(scope=state,dataset=cfg['dataset'],planned_per_size=planned,complete_per_size={str(k):v for k,v in counts.items()},
        failures=int((inventory.status=='failed').sum()),
        no_new_queries=True,patient_data_read=False,patient_performance_rerun=False,
        original_estimators_unchanged=True,
        calibration_ratio='se_to_sd = RMS reported SE / empirical Monte Carlo SD; below 1 means smaller reported SE',
        complete_seed_definition='All required CAL A/B/C methods; optional legacy failures do not remove core results',
        optional_baseline_failed_slots=(int(method_inventory.loc[
            method_inventory.probability_source.eq('legacy')&method_inventory.status.eq('failed'),
            ['n_rows','seed','method']].drop_duplicates().shape[0]) if not method_inventory.empty else 0),
        numerical_statistical_validation_only=bool(smoke),coverage_validated=False,
        plan_id=(read_json(out/'validation_declaration.json').get('plan_id')
                 if (out/'validation_declaration.json').is_file() else None))
    if not d.empty:
        d=add_errors(d)
        if d.duplicated(KEY).any():raise ValueError('Duplicate rows across seeds.')
        if d.groupby(['dataset','n_rows','query_id']).true_effect.nunique().max()!=1:
            raise ValueError('Targets changed across Monte Carlo repetitions.')
        setting,query,seeds,contrast=summary_tables(d,planned)
        comparisons=comparison_table(seeds)
        for frame in [setting,query,seeds,contrast,comparisons]:frame['validation_scope']=state
        for filename,frame in [('by_setting.csv',setting),('by_query.csv',query),('per_seed.csv',seeds),
                               ('by_contrast.csv',contrast),('const_comparisons.csv',comparisons)]:
            write_frame(dest/filename,frame)
        status['checked_method_query_rows']=len(d)
        status['linear_B_identity_and_BC_outcome_match']=True
    write_json(dest/'status.json',status)
    print(f'Result summary [{state}]: {counts}; expected {planned} seeds per size. {dest}',flush=True)
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',nargs='+')
    p.add_argument('--exploratory',action='store_true',help='Read OLD ABC output without calling it confirmation.')
    a=p.parse_args()
    for root in a.results:report(root,a.exploratory)
