#!/usr/bin/env python3
"""Print verified summaries from an interaction_ablation output directory."""
from pathlib import Path
import argparse
from ablation_report import summarize, display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='One ablation result directory.')
    parser.add_argument('--all-rows', action='store_true', help='Include all five score procedures.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    outputs = [args.output.expanduser()] if args.output else [root/f'results_{d}_ablation' for d in ('cvd','phq9')]
    found = False
    for out in outputs:
        if (out/'manifest.json').is_file():
            summarize(out)
            display(out, brief=not args.all_rows)
            found = True
        elif args.output:
            parser.error('No manifest.json in ' + str(out))
    if not found:
        parser.error('No default result directories found. Supply --output /path/to/results.')


if __name__ == '__main__':
    main()
