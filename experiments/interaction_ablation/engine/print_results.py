#!/usr/bin/env python3
"""Terminal-only, paginated A/B/C review. No model fits, rankings, or CSV exports."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = ('cal_linear','cal_sparse')
NICK = {'cal_linear':'CAL-linear','cal_sparse':'CAL-sparse'}
METHODS = ['A_full','B_CONST','C_CONST','B_LOWDIM','C_LOWDIM','B_NO_INT','C_NO_INT']


def table(df,cols,names=None):
    if df.empty:return ['No completed rows for this panel.']
    z=df[cols].copy()
    if names:z=z.rename(columns=names)
    return z.to_string(index=False,float_format=lambda x:f'{x:.5g}').splitlines()


def pages_for(out,details=False):
    out=Path(out)
    if not (out/'manifest.json').exists():
        yield [f'MISSING: {out}', 'Run the corresponding data runner; no mock fallback.'];return
    cfg=json.loads((out/'manifest.json').read_text())['config']
    inv=pd.read_csv(out/'run_inventory.csv') if (out/'run_inventory.csv').exists() else pd.DataFrame()
    if not (out/'by_setting.csv').exists():
        yield [str(out), 'No complete result table yet.']+([] if inv.empty else inv.to_string(index=False).splitlines());return
    d=pd.read_csv(out/'by_setting.csv')
    pair=pd.read_csv(out/'paired_comparisons.csv')
    for n in cfg['n_rows']:
        complete=0 if inv.empty else int(((inv.n_rows==n)&(inv.status=='complete')).sum())
        header=[out.name, f'{cfg["dataset"].upper()} | n_rows={n:,} | completed={complete}/{len(cfg["seeds"])} development seeds',
                'Actual ORIGINAL-target effects. Same original DGP and base learners.',
                'A: no added interaction restriction; B: project outcome; C: B + restricted correction.',
                'Arm-structure comparison, NOT a query-sharing ablation. No winner selected.']
        if cfg['smoke']:header.insert(1,'FICTIONAL SOFTWARE FIXTURE -- NOT YOUR PATIENT DATA')
        for learner in ('linear','flexible'):
            for source in SOURCES:
                g=d[(d.n_rows==n)&(d.learner==learner)&(d.probability_source==source)].copy()
                g['order']=g.method.map({m:i for i,m in enumerate(METHODS)});g=g.sort_values('order')
                g['Cover%']=100*g.coverage
                lines=header+['',f'A/B/C | outcome={learner} | propensity={NICK[source]}']
                lines+=table(g,['method','pooled_effect_rmse','mean_absolute_query_bias','Cover%','mean_width','se_to_sd'],
                    {'method':'Method','pooled_effect_rmse':'RMSE','mean_absolute_query_bias':'|bias|','mean_width':'Width','se_to_sd':'RMS SE/SD'})
                lines+=['RMS SE/SD below 1 means reported standard errors are smaller than empirical Monte Carlo SD.']
                small=[]
                pp=pair[(pair.n_rows==n)&(pair.learner==learner)&(pair.probability_source==source)]
                for st in ('CONST','LOWDIM','NO_INT'):
                    item={'Structure':st}
                    for label,num,den in [('B/A','B_'+st,'A_full'),('C/B','C_'+st,'B_'+st),('C/A','C_'+st,'A_full')]:
                        rr=pp[(pp.numerator==num)&(pp.denominator==den)&(pp.metric=='effect_rmse_ratio')]
                        item[label]=float(rr.estimate.iloc[0]) if len(rr) else np.nan
                    small.append(item)
                lines+=['','Paired RMSE ratios (smaller than 1 favors numerator):']+table(pd.DataFrame(small),['Structure','B/A','C/B','C/A'])
                if learner=='linear':
                    lines+=['Linear base already has constant interaction: B_CONST/B_LOWDIM should equal A.',
                            'This is an identity check, not missing training or a claim of sharing gain.']
                else:lines+=['Flexible B/A is the main response-sharing test; C/B isolates correction.',
                             'NO_INT remains a misspecification control. Small RMSE alone is not success.']
                yield lines
                if details:
                    metrics=header+['',f'DIAGNOSTICS | outcome={learner} | {NICK[source]}']
                    g['gRMSE']=np.sqrt(g.outcome_mse);g['alphaRMSE-U']=np.sqrt(g.representer_mse)
                    metrics+=table(g,['method','gRMSE','alphaRMSE-U','conditional_bias_rms','alpha_second_moment','outcome_change_mse'],
                                   {'conditional_bias_rms':'DriftRMS','alpha_second_moment':'Ealpha2','outcome_change_mse':'dMuMSE'})
                    metrics+=['alpha reference is unrestricted truth, not restricted efficiency.',
                              'Drift integrates truth AFTER fitting; never selects projection or models.']
                    rr=pp[(pp.metric=='effect_rmse_ratio')&pp.comparison.isin(['B_vs_A_response_projection','C_vs_B_correction_only'])].copy()
                    rr['Pair']=rr.numerator+'/'+rr.denominator
                    rr['95% interval']=[f'[{lo:.3f}, {hi:.3f}]' for lo,hi in zip(rr.ci_low,rr.ci_high)]
                    metrics+=['','Whole-seed bootstrap intervals (exploratory, no multiplicity adjustment):']
                    metrics+=table(rr,['Pair','estimate','95% interval','paired_repetitions'])
                    yield metrics
        ref=d[(d.n_rows==n)&d.stage.isin(['A','REF','ORACLE'])].copy()
        ref['Cover%']=ref.coverage*100
        yield header+['','UNCHANGED BASELINES / ORACLE REFERENCE (not a new best-model selection)']+table(ref,
            ['learner','probability_source','method','repetitions','planned_repetitions','pooled_effect_rmse','Cover%','mean_width','summary_scope'],
            {'pooled_effect_rmse':'RMSE','mean_width':'Width'})+[
                'Oracle uses true p AND mu only as a reference; it never enters A/B/C fitting.',
                'A should replay the original CAL + same outcome learner on matching seeds.',
                'Five seeds and correlated queries do not validate nominal 95% coverage.']
        inventory_path=out/'method_inventory.csv'
        if inventory_path.is_file():
            available=pd.read_csv(inventory_path)
            failed=available[(available.n_rows==n)&available.status.ne('complete')]
            if not failed.empty:
                yield header+['','OPTIONAL BASELINE FAILURES (main CAL results retained)']+table(failed,
                    ['seed','learner','method','status','completed_folds','planned_folds','failure_reason'])
        if details and (out/'projection_checks.csv').exists():
            checks=pd.read_csv(out/'projection_checks.csv');checks=checks[checks.n_rows==n]
            z=checks.groupby(['probability_source','learner','structure'],sort=False).agg(
                max_mu_change=('max_train_mu_change','max'),constraint_error=('max_interaction_constraint_error','max'),
                max_moment=('interaction_normal_moment','max'),expected_identity=('expected_identity','first')).reset_index()
            yield header+['','PROJECTION CHECKS -- aggregate only, no coefficients saved']+table(z,list(z.columns))+[
                'Small constraint residual checks algebra only; it does not imply small causal bias.']
    if not inv.empty and (inv.status!='complete').any():
        yield ['INCOMPLETE / FAILED PLANNED REPLICATES -- do not silently discard them']+inv.to_string(index=False).splitlines()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='*')
    p.add_argument('--details',action='store_true',help='Also print nuisance, intervals and projection diagnostics.')
    p.add_argument('--no-pause',action='store_true')
    p.add_argument('--page',type=int,help='Print only this 1-based page.')
    args=p.parse_args(argv)
    roots=[Path(x) for x in args.directories] if args.directories else [
        ROOT/'results_cvd_projection_abc',ROOT/'results_phq9_projection_abc']
    pages=[page for r in roots for page in pages_for(r,args.details)]
    if args.page is not None:
        if args.page<1 or args.page>len(pages):p.error('Page number outside available pages.')
        chosen=[(args.page,pages[args.page-1])]
    else:chosen=list(enumerate(pages,1))
    try:
        for j,(i,page) in enumerate(chosen):
            print('\n'+'='*112);print(f'PAGE {i}/{len(pages)}');print('\n'.join(page));print('='*112,flush=True)
            if not args.no_pause and args.page is None and sys.stdin.isatty() and j+1<len(chosen):
                answer=input('Enter: next | q: stop > ').strip().lower()
                if answer=='q':break
    except (BrokenPipeError,EOFError,KeyboardInterrupt):
        return


if __name__=='__main__':main()
