#!/usr/bin/env python3
"""Actual CVD/PHQ-9 empirical-covariate performance screen.
No patient dataset is downloaded. Real inputs are required unless --smoke.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import argparse, csv, json, platform, time
import numpy as np
from threadpoolctl import threadpool_limits
import baseline_experiment as base
import sparse_model
import phq_adapter,cvd_adapter
from data_bridge import build_outcome_law,finite_population_targets,draw_sample,group_folds,observed_artifacts,empirical_pilot
from outcomes import fit_outcomes,predict_outcome,orthogonal_scores
import coherent

VERSION='own-data-performance-v1.0'

def say(message):
    print(time.strftime('%H:%M:%S')+' '+message,flush=True)


def serial(x):
    if isinstance(x,(np.integer,np.floating)): return x.item()
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.bool_): return bool(x)
    raise TypeError(type(x).__name__)


def write_json(path,content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(content,ensure_ascii=False,indent=2,default=serial))
    tmp.replace(path)


def write_csv(path,records):
    records=list(records)
    if not records: return
    fields=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)


class Predictor:
    def __init__(self,kind,obj,pbar,core): self.kind=kind; self.obj=obj; self.pbar=pbar; self.core=core
    def predict(self,x,steps,scale,held):
        if self.kind=='pilot': return scale/self.pbar[steps,None,:]
        if self.kind in ('B1','projected_B1'):
            out=scale/self.pbar[steps,None,:]
            for t in range(self.core.N_STEPS):
                take=steps==t
                if take.any(): out[take]+=np.einsum('nd,qad->nqa',x[take],self.obj[:,t],optimize=True)
            return base.project_representer(out,scale) if self.kind=='projected_B1' else out
        if self.kind=='sparse': return scale*np.clip(self.obj.predict_raw(x,steps,held),1.,100.)
        if self.kind=='prob': return scale/self.obj.predict_probabilities(x,steps)[:,None,:]
        raise ValueError('Unknown predictor.')


def fit_library(artifacts,queries,pbar,core,seed,include_tree):
    say('Fitting original B1 and shared/independent sparse controls.')
    models={'pilot':Predictor('pilot',None,pbar,core)}
    choices=[];tables=[];failures={}
    try:
        beta,choice,btable=base.fit_original_B1(core,queries,artifacts)
        models['regularized_B1']=Predictor('B1',beta,pbar,core)
        models['projected_regularized_B1']=Predictor('projected_B1',beta,pbar,core)
        choices.append(dict(method='regularized_B1',**choice))
        tables.extend(dict(origin='B1',**r) for r in btable.to_dict('records'))
    except Exception as error:
        reason=f'{type(error).__name__}: {error}'
        failures.update({name:reason for name in ('regularized_B1','projected_regularized_B1')})
        models.pop('regularized_B1',None);models.pop('projected_regularized_B1',None)
        say('Optional B1 fit failed: '+reason)
    try:
        sparse=sparse_model.fit_observed(artifacts,queries,pbar,core,say)
        models.update({name:Predictor('sparse',model,pbar,core) for name,model in sparse['models'].items()})
        choices.extend(dict(method=m,**c) for m,c in sparse['choices'].items())
        tables.extend(dict(origin='sparse',**r) for r in sparse['selection_table'].to_dict('records'))
    except Exception as error:
        reason=f'{type(error).__name__}: {error}'
        failures.update({name:reason for name in sparse_model.FAMILIES})
        for name in sparse_model.FAMILIES:models.pop(name,None)
        say('Optional sparse controls failed: '+reason)
    new=coherent.fit_observed(artifacts,queries,pbar,core,seed,include_tree,say)
    models.update({name:Predictor('prob',model,pbar,core) for name,model in new['models'].items()})
    choices += [dict(method=m,**c) for m,c in new['choices'].items()]
    tables += [dict(origin='coherent',**r) for r in new['candidates']]
    failures.update(new['failures'])
    return models,choices,tables,failures


def evaluate_fold(data,models,mu_models,mu_truth,sample,test,held,eval_rows,y):
    core=data['core']; Q=len(held)
    names=list(models)+['oracle','outcome_plugin']
    learners=list(mu_models)
    units=np.unique(sample['clusters'][test]); counts=np.bincount(np.searchsorted(units,sample['clusters'][test]))
    if not np.all(counts==counts[0]): raise ValueError('Sampling profiles split across folds.')
    sums=np.zeros((len(names),len(learners),Q)); sums2=np.zeros_like(sums)
    risk=np.zeros((len(names),Q)); tail=np.zeros_like(risk)
    remainder=np.zeros_like(sums)
    # Scores on OUTER-held outcomes: no candidate chooses on these observations.
    rows=sample['rows'][test]; x=sample['x'][test]; st=sample['steps'][test]; a=sample['actions'][test]
    scale=base.query_scale(data['queries'],held,rows,st,core.P_LANDMARK)
    ghat={l:predict_outcome(m,x,st,core.N_STEPS) for l,m in mu_models.items()}
    for j,name in enumerate(names):
        if name=='oracle': alpha=scale/sample['p_true'][test,None,:]
        elif name=='outcome_plugin': alpha=np.zeros_like(scale)
        else: alpha=models[name].predict(x,st,scale,held)
        for k,l in enumerate(learners):
            scores=orthogonal_scores(scale,alpha,ghat[l],a,y[test])
            if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite orthogonal scores.')
            unit_scores=np.zeros((len(units),Q))
            np.add.at(unit_scores,np.searchsorted(units,sample['clusters'][test]),scores)
            unit_scores/=counts[:,None]
            sums[j,k]=unit_scores.sum(0); sums2[j,k]=(unit_scores**2).sum(0)
    # Separate empirical-population risk bank, accessed AFTER freeze only.
    pop_steps,p,_=base.population_arrays(data); N=len(eval_rows)
    for start in range(0,N,512):
        rr=eval_rows[start:start+512]; xx=np.asarray(data['x_population'][rr],float); tt=pop_steps[rr]
        ss=base.query_scale(data['queries'],held,rr,tt,core.P_LANDMARK)
        truth=ss/p[rr,None,:]
        errg={l:predict_outcome(m,xx,tt,core.N_STEPS)-mu_truth[rr] for l,m in mu_models.items()}
        for j,name in enumerate(names):
            if name=='oracle': aa=truth
            elif name=='outcome_plugin': aa=np.zeros_like(ss)
            else: aa=models[name].predict(xx,tt,ss,held)
            delta=aa-truth; weighted=p[rr,None,:]*delta**2
            risk[j]+=weighted.sum((0,2))/N
            tail[j]+=(weighted*(p[rr,None,:]<.05)).sum((0,2))/N
            for k,l in enumerate(learners):
                remainder[j,k]-=(p[rr,None,:]*delta*errg[l][:,None,:]).sum((0,2))/N
    return dict(names=np.asarray(names),learners=np.asarray(learners),sum_score=sums,
                sumsq_score=sums2,independent_units=np.asarray(len(units)),risk=risk,
                tail_risk=tail,remainder=remainder,test_rows=np.asarray(len(test)))


def run_replication(data,mu_truth,target,n_rows,seed,args,run_dir):
    (run_dir/'query_results.csv').unlink(missing_ok=True)
    (run_dir/'failed.json').unlink(missing_ok=True)
    core=data['core']; held=np.asarray(data['queries']['indices']['held_out'],int)
    sample=draw_sample(data,n_rows,seed,args.phq_sampling)
    sigma=args.noise_sd
    rng=np.random.default_rng(np.random.SeedSequence([seed,n_rows,923]))
    y=mu_truth[sample['rows'],sample['actions']]+rng.normal(0,sigma,len(sample['rows']))
    folds=group_folds(sample['clusters'],3,seed+747)
    all_indices=np.arange(n_rows)
    N=len(data['x_population'])
    rng_eval=np.random.default_rng(np.random.SeedSequence([seed,n_rows,193]))
    eval_rows=np.arange(N) if args.eval_rows==0 or N<=args.eval_rows else np.sort(rng_eval.choice(N,args.eval_rows,replace=False))
    results=[]; failures=[]
    for f,test in enumerate(folds):
        slot=run_dir/f'fold_{f}'; slot.mkdir(exist_ok=True)
        say(f'Outer fold {f+1}/3: train only on other sampled units.')
        train=np.setdiff1d(all_indices,test)
        pbar=sample['pbar'] if args.pilot_source=='archived' else empirical_pilot(sample,train,core)
        artifacts=observed_artifacts(sample,train,data['queries'],core,pbar,seed+f+172)
        models,choices,table,method_failures=fit_library(artifacts,data['queries'],pbar,core,seed+f,args.include_tree)
        write_csv(slot/'selected_parameters.csv',choices)
        write_csv(slot/'selection_table.csv',table)
        # Freeze the selected nuisance library BEFORE evaluation or outcome truth access.
        write_json(slot/'fit_frozen.json',dict(fold=f,
              methods=list(models),method_failures=method_failures,
              held_query_truth_used_in_selection=False,outcome_truth_used_in_selection=False))
        mu_models=fit_outcomes(sample['x'][train],sample['steps'][train],sample['actions'][train],
                               y[train],core.N_STEPS,seed+f)
        say('Choices frozen. Evaluating held-out causal scores and population representer errors.')
        metrics=evaluate_fold(data,models,mu_models,mu_truth,sample,test,held,eval_rows,y)
        failures.extend(dict(fold=f,method=m,reason=r) for m,r in method_failures.items())
        results.append(metrics)
    # Intersection is explicitly logged; any failed fold makes that method fail for this seed.
    names=sorted(set.intersection(*(set(r['names'].tolist()) for r in results)))
    allnames=sorted(set.union(*(set(r['names'].tolist()) for r in results)))
    failed_names=sorted(set(allnames)-set(names)|{x['method'] for x in failures})
    total_units=sum(int(r['independent_units']) for r in results)
    records=[]
    for name in names:
        for k,learner in enumerate(results[0]['learners']):
            totals=sum(r['sum_score'][list(r['names']).index(name),k] for r in results)
            totals2=sum(r['sumsq_score'][list(r['names']).index(name),k] for r in results)
            theta=totals/total_units
            variance=np.maximum((totals2-totals**2/total_units)/(total_units-1),0)
            se=np.sqrt(variance/total_units)
            risks=sum(float(r['test_rows'])/n_rows*r['risk'][list(r['names']).index(name)] for r in results)
            tails=sum(float(r['test_rows'])/n_rows*r['tail_risk'][list(r['names']).index(name)] for r in results)
            rem=sum(float(r['test_rows'])/n_rows*r['remainder'][list(r['names']).index(name),k] for r in results)
            for j,q in enumerate(held):
                records.append(dict(dataset=args.dataset,n_rows=n_rows,independent_units=total_units,
                    sampling_unit=sample['sampling_unit'],seed=seed,method=name,outcome_learner=str(learner),
                    query_id=str(data['query_library'].iloc[q]['query_id']),theta=float(theta[j]),
                    theta_true=float(target[j]),se=float(se[j]),ci_lower=float(theta[j]-1.96*se[j]),
                    ci_upper=float(theta[j]+1.96*se[j]),error=float(theta[j]-target[j]),
                    covered=bool(abs(theta[j]-target[j])<=1.96*se[j]),width=float(3.92*se[j]),
                    representer_mse=float(risks[j]),tail_error_contribution=float(tails[j]),
                    remainder_diagnostic=float(rem[j]),risk_eval_rows=len(eval_rows),
                    risk_exact_full_population=bool(len(eval_rows)==N),real_covariates_used=not args.smoke))
    write_csv(run_dir/'query_results.csv',records)
    for name in failed_names:
        failures.append(dict(method=name,reason='No complete estimate for every outer fold; excluded from paired comparison.'))
    write_json(run_dir/'run_details.json',dict(method_failures=failures,
             complete_methods=names,sampling_unit=sample['sampling_unit'],independent_units=total_units))
    return records


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['cvd','phq9'],required=True)
    p.add_argument('--input',default=cvd_adapter.DEFAULT_INPUT)
    p.add_argument('--member')
    p.add_argument('--phq-root',default=str(phq_adapter.DEFAULT_ROOT))
    p.add_argument('--output',required=True)
    p.add_argument('--sample-sizes',nargs='+',type=int,default=[6000])
    p.add_argument('--seeds',nargs='+',type=int,default=[67001,67002,67003,67004,67005])
    p.add_argument('--query-seed',type=int,default=cvd_adapter.DEFAULT_QUERY_SEED)
    p.add_argument('--phq-sampling',choices=['patient','iid_row'],default='patient')
    p.add_argument('--pilot-source',choices=['archived'],default='archived')
    p.add_argument('--eval-rows',type=int,default=16384,help='0: full empirical-population integration; positive: fixed-size evaluation subsample')
    p.add_argument('--noise-sd',type=float,default=1.)
    p.add_argument('--effect-scale',type=float,default=1.,help='0 provides a null-treatment sensitivity analysis')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--no-tree',dest='include_tree',action='store_false',default=True)
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Explicit fictional software fixture; never a patient-data result')
    a=p.parse_args()
    if a.smoke: a.sample_sizes=[1200 if a.dataset=='cvd' else 2400]; a.seeds=[99001]; a.eval_rows=256
    if min(a.sample_sizes)<600 or a.eval_rows<0 or a.noise_sd<=0 or a.threads<1:
        p.error('Invalid sample size, evaluation size, noise or thread count.')
    return a


def main():
    args=parse_args(); out=Path(args.output).expanduser(); out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=args.threads):
        say('Loading FICTIONAL software fixture.' if args.smoke else 'Loading your archived empirical covariate design; no data leave this environment.')
        if args.dataset=='cvd':
            data=cvd_adapter.load_mock(patients=300,query_seed=args.query_seed) if args.smoke else cvd_adapter.load_real(args.input,args.member,args.query_seed)
        else:
            data=phq_adapter.load_mock(patients=300) if args.smoke else phq_adapter.load_real(args.phq_root)
        metadata=dict(data['input_metadata'])
        metadata['warning']='Known treatment and outcome laws are simulations; this is NOT clinical treatment-effect validation.'
        metadata['requested_sampling']='patient_profile' if args.dataset=='phq9' and args.phq_sampling=='patient' else 'iid_row'
        write_json(out/'input_audit.json',metadata)
        say(f'Input passed: {len(data["x_population"])} population rows; {data["core"].N_STEPS} landmark(s); {len(data["query_library"])} queries.')
        if args.check_inputs: return
        mu,law=build_outcome_law(data,args.effect_scale); law['residual_sd']=args.noise_sd
        target=finite_population_targets(data,mu)
        import scipy,sklearn,pandas
        config={k:v for k,v in vars(args).items() if k not in ('check_inputs','output','input','phq_root','member')}
        config.update(version=VERSION,input_metadata=metadata,outcome_law=law,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pandas.__version__),
            primary_comparison='within-family action_ce versus original Riesz selection; no universal-winner claim',
            historical_comparability='NEW outer cross-fitting/outcome protocol. PHQ primary samples profiles, not old iid landmark rows.')
        write_json(out/'manifest.json',config)
        for n in args.sample_sizes:
            for seed in args.seeds:
                rdir=out/'runs'/f'n{n}_seed{seed}'
                for filename in ('query_results.csv', 'failed.json'):
                    (rdir/filename).unlink(missing_ok=True)
        for n in args.sample_sizes:
            for seed in args.seeds:
                rdir=out/'runs'/f'n{n}_seed{seed}'; rdir.mkdir(parents=True,exist_ok=True)
                failed=rdir/'failed.json'
                say(f'START n_rows={n}, seed={seed}.')
                try:
                    run_replication(data,mu,target,n,seed,args,rdir)
                    say(f'FINISHED n_rows={n}, seed={seed}.')
                except Exception as exc:
                    write_json(failed,dict(seed=seed,n_rows=n,error_type=type(exc).__name__,
                        message=str(exc)))
                    say(f'FAILED {type(exc).__name__}; recorded, not removed from the study inventory.')
                    if args.smoke: raise
        import summarize
        summarize.summarize(out)
        say('Complete. Inspect by_setting.csv, paired_comparisons.csv and run_inventory.csv.')

if __name__=='__main__':
    main()
