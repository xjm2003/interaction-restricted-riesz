from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'

class IntegrityError(RuntimeError):
    """Invalid experiment inputs or incompatible result table contents."""

def say(message):
    print(time.strftime('%H:%M:%S') + ' ' + str(message), flush=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)

def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, pandas=pd.__version__, sklearn=sklearn.__version__)

def import_engine():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ENGINE))
    spec = importlib.util.spec_from_file_location('archived_abc_engine', ENGINE / 'run.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def find_reference(dataset, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / 'manifest.json').exists():
            path = path / f'results_{dataset}_const_validation'
        if not (path / 'manifest.json').is_file():
            raise IntegrityError('No reference manifest under --reference.')
        return path
    candidates = [
        Path.home() / 'interaction_const_validation' / f'results_{dataset}_const_validation',
        Path.home() / f'results_{dataset}_const_validation',
        ROOT.parent / 'interaction_const_validation' / f'results_{dataset}_const_validation',
    ]
    found = list(dict.fromkeys(path.resolve() for path in candidates if (path / 'manifest.json').is_file()))
    if len(found) != 1:
        raise IntegrityError('Specify --reference /path/to/results_' + dataset + '_const_validation')
    return found[0]

def read_reference(reference, dataset, n, seeds):
    """Read the original experiment settings for downstream reporting."""
    return read_json(reference / 'manifest.json')

def reference_seed(reference, n, seed, qids, truth):
    path = reference / 'runs' / f'n_{n}' / f'seed_{seed}' / 'query_results.csv'
    frame = pd.read_csv(path, dtype={'query_id': str}, float_precision='round_trip')
    frame = frame[frame.query_index.isin(qids)].copy()
    if frame.empty or not frame.seed.eq(seed).all() or not frame.n_rows.eq(n).all():
        raise IntegrityError('Reference result sample-size/seed keys do not match.')
    keys = ['learner', 'probability_source', 'method', 'query_index']
    if frame.duplicated(keys).any():
        raise IntegrityError('Duplicated reference method/query records.')
    for _, group in frame.groupby(['learner', 'probability_source', 'method']):
        if set(group.query_index) != set(qids):
            raise IntegrityError('Reference method is missing a query.')
    keep = frame.method.isin(['A_full', 'B_CONST', 'C_CONST', 'shared_sparse_riesz',
                              'regularized_B1', 'oracle_AIPW'])
    frame = frame[keep].copy()
    frame['origin'] = 'archived_reference'
    return frame

