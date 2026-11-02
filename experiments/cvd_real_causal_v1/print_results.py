#!/usr/bin/env python3
"""Read-only report; safe to use while a different process is fitting."""
from pathlib import Path
import argparse
from cvd_real import print_report
p = argparse.ArgumentParser()
p.add_argument('output', nargs='?', default=str(Path(__file__).resolve().parent/'results_cvd_real_v1'))
a = p.parse_args()
print_report(Path(a.output).expanduser())
