#!/usr/bin/env python3
"""Optional collection of AGGREGATE results only. Not needed for terminal review."""
from pathlib import Path
import argparse,json,zipfile

ALLOW = ('by_setting.csv','by_query.csv','by_contrast.csv','paired_comparisons.csv',
         'paired_by_contrast.csv','reference_comparisons.csv','run_inventory.csv','per_seed.csv',
         'by_overlap_diagnostics.csv','projection_checks.csv','projection_gram_diagnostics.csv','query_catalog.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');a=p.parse_args()
    r=Path(a.directory)
    if not (r/'manifest.json').exists():p.error('Missing result manifest.')
    m=json.loads((r/'manifest.json').read_text());cfg=m['config']
    safe={k:cfg[k] for k in ('version','dataset','n_rows','seeds','smoke','eval_rows','frozen','runtime')}
    target=r.parent/(r.name+'_review.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('REVIEW_SCOPE.json',json.dumps({'config':safe,
            'contains_patient_rows_or_models':False,'note':'Aggregate output only; institutional sharing rules still apply.'},indent=2))
        for f in ALLOW:
            if (r/f).exists():z.write(r/f,f)
    print(target)

if __name__=='__main__':main()
