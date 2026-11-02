#!/usr/bin/env python3
"""Read existing 5-seed ABC files and paired uncertainty, without any refitting."""
from pathlib import Path
import sys
from result_io import ROOT
from print_results import main

if __name__=='__main__':
    paths=sys.argv[1:]
    if not paths:
        for ds in ['cvd','phq9']:
            name=f'results_{ds}_projection_abc_v1'
            candidates=[ROOT.parent/'interaction_projection_abc_v1'/name,ROOT.parent/name,Path.cwd()/name]
            found=next((p for p in candidates if (p/'manifest.json').is_file()),None)
            if found:paths.append(str(found))
    if not paths:
        print('No OLD ABC output found. No model was run.')
        print('Place this validation folder beside interaction_projection_abc_v1, or specify the OLD result directory:')
        print('python review_development.py ../interaction_projection_abc_v1/results_cvd_projection_abc_v1')
        raise SystemExit(1)
    main(['--exploratory',*paths])
