from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,np.floating,np.bool_)): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)

def write_json(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=serial,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def write_csv(path,rows,fields=None):
    rows=list(rows); fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

