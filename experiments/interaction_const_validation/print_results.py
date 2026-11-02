#!/usr/bin/env python3
"""Paginated aggregate validation report for screenshot-only environments."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from result_io import ROOT,read_json
from validation_report import report

def num(x,d=3):
    return f'{float(x):.{d}f}' if pd.notna(x) and np.isfinite(float(x)) else 'NA'

def interval(lo,hi):return '['+num(lo)+', '+num(hi)+']'

def pct(x):return num(100*float(x),1)+'%' if pd.notna(x) else 'NA'

def pages_for(root,exploratory=False,details=False,refresh=True):
    root=Path(root).expanduser().resolve()
    if not (root/'manifest.json').exists():
        return [[f'MISSING result directory: {root.name}', 'No model is fitted by this printer.']]
    if refresh:report(root,exploratory)
    dest=root/'verification';status=read_json(dest/'status.json');pages=[]
    if not (dest/'by_setting.csv').exists():
        return [[status['dataset'].upper(),status['scope'],'No completed seed yet; unfinished/failed seeds are retained.']]
    ss=pd.read_csv(dest/'by_setting.csv');pp=pd.read_csv(dest/'const_comparisons.csv')
    qs=pd.read_csv(dest/'by_query.csv')
    for n in sorted(ss.n_rows.unique()):
        s=ss[ss.n_rows==n];p=pp[pp.n_rows==n]
        done=status['complete_per_size'][str(int(n))];planned=status['planned_per_size']
        header=[f"{status['dataset'].upper()} | n_rows={n:,} | completed={done}/{planned} | {status['scope']}",
         'SAME original target/DGP/estimators. New simulation seeds; NOT new-query or clinical validation.',
         'Uncertainty is across whole Monte Carlo seeds, not the 48 correlated queries.',
         'C_CONST was chosen after development. New repetitions must all finish; no interim winner.']
        if status['scope']!='COMPLETE_FROZEN_REPLICATION':header+=['INTERIM / SOFTWARE / EXPLORATORY: do not make a confirmation claim.']
        if int(n)!=6000 and status['scope']!='SOFTWARE_FIXTURE':header+=['SAMPLE-SIZE SENSITIVITY. Prespecified primary sample size is 6000.']
        def get(l,pr,m):
            a=s[(s.learner==l)&(s.probability_source==pr)&(s.method==m)]
            if len(a)!=1:raise ValueError('Incomplete or ambiguous summary group.')
            return a.iloc[0]
        page=header+['','A: FROZEN C_CONST AGAINST MATCHED A/B',
            'Outcome    CAL     RMSE_A  RMSE_B  RMSE_C   C/A    C/B    Cover_A Cover_C Width_C/A']
        for l in ['linear','flexible']:
            for pr in ['cal_linear','cal_sparse']:
                a,b,c=[get(l,pr,m) for m in ['A_full','B_CONST','C_CONST']]
                page.append(f'{l:10} {pr[4:]:6} {a.effect_rmse:7.5f} {b.effect_rmse:7.5f} {c.effect_rmse:7.5f} '
                  f'{c.effect_rmse/a.effect_rmse:6.3f} {c.effect_rmse/b.effect_rmse:6.3f} '
                  f'{pct(a.coverage):>7} {pct(c.coverage):>7} {c.mean_width/a.mean_width:8.3f}')
        page+=['','A=original full CAL-AIPW. B=project outcome, full correction. C=same B outcome, restricted correction.',
               'Linear B_CONST=A is an identity, not evidence of new response sharing.',
               'Coverage near 95% or an MC interval containing 95% is NOT proof of honest inference.']
        pages.append(page)
        page=header+['','B: PAIRED RMSE-RATIO UNCERTAINTY',
            'Family     Outcome   CAL    Ratio    95% MC interval       Family(8) interval     Bootstrap95']
        for fam,label in [('primary_correction','C/B'),('secondary_total','C/A')]:
            for _,r in p[p.comparison_family==fam].iterrows():
                page.append(f'{label:10} {r.learner:9} {r.probability_source[4:]:6} {num(r.ratio):>6} '
                  f'{interval(r.ci_low,r.ci_high):22} {interval(r.family_ci_low,r.family_ci_high):22} '
                  f'{interval(r.bootstrap_ci_low,r.bootstrap_ci_high)}')
        page+=['','RMSE ratio = sqrt(mean seed MSE_C / mean seed MSE_reference), not mean seedwise ratio.',
          'MC: paired delta/log-t interval. Family(8): Bonferroni across BOTH datasets x learners x CAL sources.',
          'Each comparison family is separate. Bootstrap is a marginal sensitivity check, not multiplicity-adjusted.',
          'These are uncertainty intervals for simulation performance, NOT patient-level causal intervals.']
        pages.append(page)
        page=header+['','C: STRONG REFERENCES AND COVERAGE (NO ORACLE SELECTION)',
            'Reference      Outcome   CAL    C/ref   95% MC interval       Family(8) interval   Paired/core']
        for fam,label in [('secondary_shared_sparse','shared sparse'),('secondary_B1','B1')]:
            for _,r in p[p.comparison_family==fam].iterrows():
                page.append(f'{label:14} {r.learner:9} {r.probability_source[4:]:6} {num(r.ratio):>6} '
                  f'{interval(r.ci_low,r.ci_high):22} {interval(r.family_ci_low,r.family_ci_high)} '
                  f'{int(r.paired_repetitions)}/{int(r.numerator_repetitions)} [{r.comparison_scope}]')
        page+=['','C_CONST coverage: mean, MCSE, marginal 95% MC interval; minimum query coverage is noisy.']
        for l in ['linear','flexible']:
            for pr in ['cal_linear','cal_sparse']:
                c=get(l,pr,'C_CONST')
                warning=' [MC upper < .95]' if bool(c.MC_upper_below_nominal_95) else ''
                page.append(f'{l:9}/{pr[4:]:6}: {pct(c.coverage):>6} MCSE={pct(c.coverage_mcse):>5} '
                 f'CI={interval(c.coverage_mc_low,c.coverage_mc_high)} minQ={pct(c.min_query_coverage):>6}'
                 f' RMS SE/SD={num(c.se_to_sd)}{warning}')
        page+=['RMS SE/SD below 1 means reported standard errors are smaller than empirical Monte Carlo SD.',
               'Optional baseline comparisons use only complete matched seeds; counts above disclose missing baselines.',
               'NO inference-valid/pass label is produced. Check bias, widths and every query, not only the pooled mean.']
        pages.append(page)
        if details:
            base=header+['','D: LOWDIM / NO_INT CONTROLS (NOT SELECTED WINNERS)',
              'Outcome   CAL    Method       RMSE    Bias     Coverage Width    RMS SE/SD']
            for l in ['linear','flexible']:
                for pr in ['cal_linear','cal_sparse']:
                    for method in ['B_LOWDIM','C_LOWDIM','B_NO_INT','C_NO_INT']:
                        c=get(l,pr,method)
                        base.append(f'{l:9} {pr[4:]:6} {method:11} {c.effect_rmse:7.5f} {c.mean_absolute_query_bias:7.5f} '
                         f'{pct(c.coverage):>7} {c.mean_width:7.4f} {num(c.se_to_sd):>6}')
            pages.append(base)
            qo=qs[(qs.n_rows==n)&(qs.method=='C_CONST')]
            for (l,pr),g in qo.groupby(['learner','probability_source'],sort=False):
                # Show every query, not just favorable subsets.
                g=g.sort_values('query_id')
                for start in range(0,len(g),12):
                    b=header+['',f'E: ALL QUERY DIAGNOSTICS C_CONST {l}/{pr}; {start+1}-{min(start+12,len(g))}',
                        'Query           RMSE    Bias    Cover  Wilson95        Wilson-family48']
                    for _,r in g.iloc[start:start+12].iterrows():
                        b.append(f'{str(r.query_id):15} {r.effect_rmse:7.4f} {r.bias:7.4f} {pct(r.coverage):>6} '
                         f'{interval(r.coverage_wilson_low,r.coverage_wilson_high):17} '
                         f'{interval(r.coverage_queryfamily_low,r.coverage_queryfamily_high)}')
                    pages.append(b)
    return pages

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',nargs='*');ap.add_argument('--details',action='store_true')
    ap.add_argument('--all',action='store_true',help='Print without pauses.')
    ap.add_argument('--page',type=int);ap.add_argument('--exploratory',action='store_true')
    a=ap.parse_args(argv)
    paths=[Path(x) for x in a.results] or [ROOT/f'results_{ds}_const_validation' for ds in ['cvd','phq9']]
    pages=[]
    for path in paths:
        if a.results or (path/'manifest.json').is_file():pages+=pages_for(path,a.exploratory,a.details)
    if not pages:
        print('No validation results yet. Run python -u run_cvd.py or run_phq9.py.');return
    if a.page is not None:
        if not 1<=a.page<=len(pages):ap.error(f'Page must be 1..{len(pages)}')
        print(f'PAGE {a.page}/{len(pages)}\n'+'\n'.join(pages[a.page-1]));return
    try:
        for i,page in enumerate(pages):
            print('\n'+'='*108+f'\nPAGE {i+1}/{len(pages)}\n'+'\n'.join(page)+'\n'+'='*108)
            if not a.all and sys.stdin.isatty() and i<len(pages)-1:
                if input('Enter: next | q: stop > ').strip().lower()=='q':break
    except (EOFError,KeyboardInterrupt):pass

if __name__=='__main__':main()
