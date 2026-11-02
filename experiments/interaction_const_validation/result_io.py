"""Paths and JSON helpers for experiment settings and final result summaries."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / 'engine'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    tmp.replace(path)
