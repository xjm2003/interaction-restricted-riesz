#!/usr/bin/env python3
from pathlib import Path
import argparse
from ext_report import summarize, display
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='*');p.add_argument('--all-rows',action='store_true');a=p.parse_args()
    dirs=[Path(s).expanduser() for s in a.directories] if a.directories else [ROOT/'results_cvd_external',ROOT/'results_phq9_external']
    for out in dirs:
        if not (out/'manifest.json').is_file():print('NOT RUN: '+str(out));continue
        summarize(out);display(out,brief=not a.all_rows)
