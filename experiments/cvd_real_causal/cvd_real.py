#!/usr/bin/env python3
"""Real CVD competing-risk estimation. No simulation in the production path.

Four recorded medication arms, cross-fitted AIPTW/AIPCW risk estimation,
matched constant-interaction projections, and optional patient-level refit
bootstrap. Clinical definitions remain explicit WORKING assumptions.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import datetime as dt
import gzip
import io
import json
import os
from pathlib import Path
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn import __version__ as SK_VERSION
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

VERSION = 'cvd_real_causal_v1.0'
INPUT_ENV_VAR = 'CVD_DATA_PATH'
NUMERIC = ('age', 'sbp', 'bmi', 'tc', 'hdlc', 'egfr', 'adi_natrank')
CATEGORICAL = ('sex', 'race', 'diabetes', 'smoker', 'insurance_type')
ARMS = ('neither', 'antihtn_only', 'statin_only', 'both')
K = np.array([1., -1., -1., 1.])
ROUTES = ('full_survival_aipw', 'projected_risk_full_aipw', 'constant_interaction_riesz')
LABELS = {
 'full_survival_aipw': 'Full survival AIPW/AIPCW',
 'projected_risk_full_aipw': 'Projected risk + full correction',
 'constant_interaction_riesz': 'Constant-interaction Riesz (survival extension)',
}
NA = {'', 'na', 'nan', 'null', 'none', '<na>', 'n/a', 'nat'}
ASSUMPTIONS = {
 'status': 'WORKING_ASSUMPTIONS_NOT_CLINICALLY_VERIFIED',
 'time_zero': 'Stored INDEX_DATE; not reconstructed or certified as incident/new-user eligibility.',
 'exposure': 'Stored statin/antihtn are provisionally interpreted as baseline treatment indicators; not ever-use, adherence, or sustained regimens.',
 'covariates': 'The 12 specified fields are provisionally baseline/pre-treatment; chronology, upstream imputation and transforms remain unverified. No clinical cutoffs are applied.',
 'endpoint': 'The stored cvd_5y label and time2cvd_5y define the RECORDED CVD endpoint; no claim that its codebook or incident/first-ever status is verified.',
 'units': 'followup_years and time2*_5y are assumed to use years from INDEX_DATE.',
 'event_free_times': 'For label=0, time2*_5y is assumed to be the endpoint-specific ascertainment stop, not an unverified imputed event time.',
 'common_ascertainment': 'Stop is min(followup_years, horizon, relevant non-event ascertainment stops). Recorded CVD/death before this stop terminates first-event follow-up.',
 'death': 'The recorded CVD endpoint is provisionally assumed NOT to include death-only events. Death before recorded CVD is a competing event, not ordinary censoring; later death after CVD is not a second observation.',
 'censoring': 'Conditional independent right censoring given specified X and baseline A, with support to the requested horizon; fitted positive G is not proof of positivity.',
 'causal': 'Consistency, a well-defined baseline exposure intervention, no interference, conditional exchangeability and treatment positivity are required; they are not verified by this program.',
 'projection': 'Additional constant interaction applies to horizon-specific CVD CUMULATIVE INCIDENCE, not hazards/log odds. This is a new survival application of the projection, not the old continuous-outcome experiment.',
 'inference': 'Pointwise nominal score-Wald intervals. Structural bias, clipping, nuisance misspecification and survival-extension validity are not repaired by cross-fitting or bootstrap.',
 'target_population': 'Patients with valid, coherent records after explicitly recorded quality holds and any pre-specified baseline calendar filter; not automatically all source patients.',
}

class SafeError(RuntimeError):
    """Metadata-only errors: do not include identifiers, dates or row values."""


def log(s):
    print(dt.datetime.now().strftime('%H:%M:%S') + ' ' + str(s), flush=True)


def clean(s):
    z = s.astype(str).str.strip()
    return z.mask(z.str.lower().isin(NA), '')


def numeric(s):
    x = pd.to_numeric(clean(s), errors='coerce').to_numpy(dtype=float)
    return np.where(np.isfinite(x), x, np.nan)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.chmod(tmp, 0o600); tmp.replace(path)


def write_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf8', newline='') as f:
        frame.to_csv(f, index=False)
    os.chmod(tmp, 0o600); tmp.replace(path)


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.chmod(tmp, 0o600); tmp.replace(path)


def suppress(n):
    return '<11' if 0 < int(n) < 11 else str(int(n))


@contextmanager
def csv_stream(source, member):
    p = Path(source)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            hits = [v for v in z.infolist() if not v.is_dir() and v.filename == member]
            if len(hits) != 1:
                raise SafeError('Requested ZIP member missing/duplicated; no alternate table is silently selected.')
            with z.open(hits[0]) as b, io.TextIOWrapper(b, encoding='utf-8-sig', newline='') as f:
                yield f
    elif p.name.endswith('.gz'):
        with gzip.open(p, 'rt', encoding='utf-8-sig', newline='') as f:
            yield f
    else:
        with open(p, encoding='utf-8-sig', newline='') as f:
            yield f


def resolve_input_path(source=None):
    """Explicit --input takes precedence over the optional environment variable."""
    value = source if source is not None else os.environ.get(INPUT_ENV_VAR)
    if value is None or not str(value).strip():
        raise SafeError('CVD input is required: pass --input PATH or set CVD_DATA_PATH.')
    return Path(value).expanduser().resolve()


def read_input(source, member):
    """Read selected columns only. Check row widths and keep textual identifiers."""
    source = resolve_input_path(source)
    if not source.is_file():
        raise SafeError('Input file not found. Run on the server with the recorded CVD archive.')
    required = set(NUMERIC + CATEGORICAL + ('patid', 'index_date', 'statin', 'antihtn',
                   'cvd_5y', 'time2cvd_5y', 'death_5y', 'time2death_5y', 'followup_years'))
    optional = {f'{e}_5y' for e in ('mi', 'stroke', 'hf')}
    rows = []
    with csv_stream(source, member) as stream:
        reader = csv.reader(stream, strict=True)
        header = next((r for r in reader if r), None)
        if not header:
            raise SafeError('Empty CSV.')
        header = [v.strip().lower() for v in header]
        nonblank = [v for v in header if v]
        if len(set(nonblank)) != len(nonblank):
            raise SafeError('Duplicate nonempty normalized column names.')
        absent = sorted(required - set(header))
        if absent:
            raise SafeError('Required columns absent: ' + ', '.join(absent))
        chosen = [i for i, c in enumerate(header) if c in required | optional]
        names = [header[i] for i in chosen]
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise SafeError('Malformed CSV row width; no silent row dropping.')
            rows.append([row[i].strip() for i in chosen])
    raw = pd.DataFrame(rows, columns=names)
    if len(raw) == 0:
        raise SafeError('No input observations.')
    raw['patid'] = clean(raw.patid)
    if raw.patid.eq('').any() or raw.patid.duplicated().any():
        raise SafeError('Missing or duplicated PATID: requires one record per patient; no automatic deduplication.')
    provenance = dict(source=str(source), zip_member=member if zipfile.is_zipfile(source) else None,
        source_rows=len(raw), selected_columns=names)
    return raw, provenance


def parse_index(values):
    """Only documented date-only / midnight ISO or US date strings; no arbitrary epoch guessing."""
    s = clean(values)
    out = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns]')
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f', '%m/%d/%Y %H:%M:%S'):
        remaining = out.isna()
        if remaining.any():
            out.loc[remaining] = pd.to_datetime(s.loc[remaining], format=fmt, errors='coerce')
    # Non-midnight timestamps cannot be silently treated as dates.
    return out.mask(out.notna() & out.ne(out.dt.normalize()))


def build_cohort(raw, horizon=5., ties='hold', index_end=None, tolerance=1e-6):
    n = len(raw); reasons = [[] for _ in range(n)]
    def flag(name, mask):
        for i in np.flatnonzero(np.asarray(mask, bool)):
            reasons[int(i)].append(name)
    a0, a1 = numeric(raw.antihtn), numeric(raw.statin)
    ev, de = numeric(raw.cvd_5y), numeric(raw.death_5y)
    te, td, fu = numeric(raw.time2cvd_5y), numeric(raw.time2death_5y), numeric(raw.followup_years)
    idx = parse_index(raw.index_date)
    flag('invalid_index_date', idx.isna())
    flag('invalid_medication_code', ~np.isin(a0, [0,1]) | ~np.isin(a1, [0,1]))
    flag('invalid_endpoint_code', ~np.isin(ev, [0,1]) | ~np.isin(de, [0,1]))
    flag('missing_or_nonpositive_followup', ~np.isfinite(fu) | (fu <= 0))
    flag('invalid_cvd_time', ~np.isfinite(te) | (te <= 0))
    flag('invalid_death_time', ~np.isfinite(td) | (td <= 0))
    flag('cvd_positive_after_label_horizon', (ev == 1) & (te > 5. + tolerance))
    flag('death_positive_after_label_horizon', (de == 1) & (td > 5. + tolerance))
    flag('cvd_after_recorded_followup', (ev == 1) & (te > fu + tolerance))
    # Registry death after a shorter CVD ascertainment period is allowed, but does
    # not magically extend CVD follow-up or remove its earlier censoring.
    flag('cvd_after_death', (ev == 1) & (de == 1) & (td < te - tolerance))
    tied = (ev == 1) & (de == 1) & (np.abs(te-td) <= tolerance) & (te <= horizon + tolerance)
    if ties == 'hold':
        flag('unresolved_cvd_death_tie', tied)
    c = np.minimum(fu, horizon)
    c = np.minimum(c, np.where(ev == 0, te, np.inf))
    c = np.minimum(c, np.where(de == 0, td, np.inf))
    flag('invalid_common_observation_stop', ~np.isfinite(c) | (c <= 0))
    flag('cvd_after_common_ascertainment', (ev == 1) & (te <= horizon) & (te > c + tolerance))
    ce = np.where((ev == 1) & (te <= horizon+tolerance), te, np.inf)
    cd = np.where((de == 1) & (td <= horizon+tolerance), td, np.inf)
    event1 = (ce <= c+tolerance) & (ce < cd-tolerance)
    event2 = (cd <= c+tolerance) & (cd < ce-tolerance)
    if ties == 'cvd_first':
        event1 |= tied & (ce <= c+tolerance)
    elif ties == 'death_first':
        event2 |= tied & (cd <= c+tolerance)
    stop = np.minimum(c, np.minimum(ce, cd))
    stop = np.minimum(stop, horizon)
    kind = np.where(event1, 1, np.where(event2, 2, np.where(stop < horizon-tolerance, 3, 0))).astype(int)
    # Codes: 0 administrative horizon; 1 recorded CVD; 2 death first; 3 early right censor.
    held = np.array([bool(r) for r in reasons])
    excluded_calendar = np.zeros(n, bool)
    if index_end:
        boundary = pd.to_datetime(index_end, format='%Y-%m-%d', errors='coerce')
        if pd.isna(boundary):
            raise SafeError('--index-end must be a YYYY-MM-DD baseline calendar cutoff.')
        excluded_calendar = idx.gt(boundary).to_numpy()
    keep = ~held & ~excluded_calendar
    cov = raw.loc[keep, list(NUMERIC+CATEGORICAL)].copy().reset_index(drop=True)
    cov['index_calendar_year'] = idx.loc[keep].dt.year.astype(float).to_numpy()
    data = dict(x=cov, a=(2*a1[keep]+a0[keep]).astype(int), t=stop[keep], kind=kind[keep],
                source_rows=np.flatnonzero(keep), index_year=idx.loc[keep].dt.year.to_numpy(),
                patient_id=raw.patid.loc[keep].to_numpy())
    counts = {}
    for row in reasons:
        for reason in row:
            counts[reason] = counts.get(reason, 0) + 1
    audit = dict(source_rows=n, retained_rows=int(keep.sum()), held_rows=int(held.sum()),
        baseline_calendar_excluded=int((~held & excluded_calendar).sum()),
        hold_counts=counts, arm_counts=np.bincount(data['a'], minlength=4).tolist(),
        endpoint_type_counts={str(k):int(np.sum(data['kind']==k)) for k in range(4)},
        future_followup_complete_case_selection=False, death_tie_policy=ties,
        retrospective_component_identity_not_assumed=True,
        full_cohort_target_not_claimed_if_quality_holds=True)
    if all(f'{z}_5y' in raw for z in ('mi','stroke','hf')):
        parts = np.column_stack([numeric(raw[f'{z}_5y']) for z in ('mi','stroke','hf')])
        known = np.isin(parts,[0,1]).all(axis=1) & np.isin(ev,[0,1])
        audit['stored_cvd_vs_component_union_disagreements_DIAGNOSTIC_ONLY'] = int(np.sum(known & (ev != parts.max(axis=1))))
    ledger = pd.DataFrame(dict(source_row=np.arange(n), PATID=raw.patid,
        retained=keep, quality_hold=held, baseline_calendar_excluded=excluded_calendar,
        hold_reasons=[';'.join(r) for r in reasons]))
    return data, audit, ledger


class Basis:
    """Train-only numeric missingness/standardization; categorical unseen level retained."""
    def fit(self, frame):
        self.num = {}; self.cat = {}
        self.num_names = list(NUMERIC) + ['index_calendar_year']
        for c in self.num_names:
            x = numeric(frame[c]); good = x[np.isfinite(x)]
            if not len(good):
                raise SafeError('All training values missing for a required baseline covariate: ' + c)
            med = float(np.median(good)); fill = np.where(np.isfinite(x),x,med)
            self.num[c] = (med, float(np.mean(fill)), max(float(np.std(fill)), 1e-8))
        for c in CATEGORICAL:
            vals = clean(frame[c]).replace('', '__MISSING__')
            count = vals.value_counts()
            self.cat[c] = list(count.index[:20])
        return self
    def transform(self, frame):
        cols = []
        for c in self.num_names:
            med,mean,sd = self.num[c]; x = numeric(frame[c]); missing = ~np.isfinite(x)
            z = np.clip((np.where(missing,med,x)-mean)/sd, -8., 8.)
            cols.extend([z, z*z, missing.astype(float)])
        for c, levels in self.cat.items():
            vals = clean(frame[c]).replace('', '__MISSING__').to_numpy()
            cols.extend([(vals==lev).astype(float) for lev in levels])
            cols.append((~np.isin(vals, levels)).astype(float))
        result = np.column_stack(cols)
        if not np.isfinite(result).all():
            raise SafeError('Nonfinite transformed covariates.')
        return result


def smooth_prob(p, floor):
    if p.shape[1] != 4 or not np.isfinite(p).all() or np.any(p < 0):
        raise SafeError('Invalid four-arm propensity prediction.')
    p = p/p.sum(axis=1,keepdims=True)
    return floor + (1.-4.*floor)*p


def fit_propensity(x, a, method, seed, cfg):
    if len(np.unique(a)) != 4:
        raise SafeError('A training fold lacks a recorded arm; no silent fallback.')
    if method == 'logit':
        model = LogisticRegression(C=cfg['logit_C'], solver='lbfgs', max_iter=1500, tol=1e-6)
    else:
        model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=cfg['iterations'],
            max_leaf_nodes=15, min_samples_leaf=cfg['min_leaf'], l2_regularization=1.,
            early_stopping=False, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning); model.fit(x,a)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise SafeError('Propensity solver did not converge; no unconverged estimate reported.')
    if not np.array_equal(model.classes_,np.arange(4)):
        raise SafeError('Propensity class ordering changed.')
    return model


def design_time(x, a, period, nbins, learner):
    arm = np.eye(4)[np.asarray(a,int),1:]
    period = np.asarray(period,int)
    if learner == 'poisson':
        time_basis = np.eye(nbins)[period,1:]
        # Arm-specific slopes and arm-specific interval intercepts; no constant
        # risk-interaction restriction is built into this baseline outcome fit.
        interactions = (x[:,:,None]*arm[:,None,:]).reshape(len(x),-1)
        tx_time = (time_basis[:,:,None]*arm[:,None,:]).reshape(len(x),-1)
        return np.column_stack((x,arm,time_basis,interactions,tx_time))
    return np.column_stack((x,arm,(period+.5)/nbins))


def risk_records(x,a,t,kind,edges,learner):
    idx=[]; bins=[]; exposure=[]; terminal=[]
    delta=np.diff(edges); last=np.searchsorted(edges[1:], t, side='left')
    last=np.minimum(last,len(delta)-1)
    for j in range(len(delta)):
        use=np.flatnonzero(t>edges[j])
        idx.append(use); bins.append(np.full(len(use),j,int))
        exposure.append(np.minimum(t[use],edges[j+1])-edges[j])
        terminal.append(np.where(last[use]==j,kind[use],0))
    ids=np.concatenate(idx); periods=np.concatenate(bins); durations=np.concatenate(exposure)
    eventtype=np.concatenate(terminal)
    if np.any(durations<=0):
        raise SafeError('Nonpositive risk-set exposure.')
    features=design_time(x[ids],a[ids],periods,len(delta),learner)
    return features,durations,eventtype


class HazardModels:
    """Piecewise-constant hazards with exact event/censor times and exposure likelihood."""
    def fit(self,x,a,t,kind,edges,seed,cfg):
        self.edges=edges; self.learner=cfg['hazard_learner']; self.models={}; self.fit_notes={}
        features,exposure,eventtype=risk_records(x,a,t,kind,edges,self.learner)
        for event,name in ((1,'cvd'),(2,'death'),(3,'censor')):
            count=int(np.sum(eventtype==event)); self.fit_notes[name+'_events']=count
            if event==1 and count < cfg['min_events']:
                raise SafeError('Too few CVD events in a training fold for the specified fit.')
            if count==0:
                self.models[event]=None; self.fit_notes[name+'_zero_rate_model']=True
                continue
            y=(eventtype==event).astype(float)/exposure
            if self.learner=='poisson':
                model=PoissonRegressor(alpha=cfg['poisson_alpha'],max_iter=1500,tol=1e-7)
            else:
                model=HistGradientBoostingRegressor(loss='poisson',learning_rate=.05,
                    max_iter=cfg['iterations'],max_leaf_nodes=15,min_samples_leaf=cfg['min_leaf'],
                    l2_regularization=1.,early_stopping=False,random_state=seed+event)
            log('    fitting '+name+' hazard (person-time likelihood)')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(features,y,sample_weight=exposure)
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                raise SafeError('Hazard solver did not converge: '+name)
            self.models[event]=model
        self.fit_notes['person_period_rows']=int(len(exposure))
        return self
    def predict(self,x,a):
        n=len(x); m=len(self.edges)-1
        flat=design_time(np.repeat(x,m,axis=0),np.repeat(a,m),np.tile(np.arange(m),n),m,self.learner)
        rates=[]
        for event in (1,2,3):
            model=self.models[event]
            z=np.zeros(n*m) if model is None else model.predict(flat)
            if not np.isfinite(z).all() or (z<0).any():
                raise SafeError('Invalid hazard prediction; no silent rate replacement.')
            rates.append(z.reshape(n,m))
        return rates
    def all_risks(self,x,chunk=1024):
        out=np.empty((len(x),4))
        for left in range(0,len(x),chunk):
            xx=x[left:left+chunk]
            for a in range(4):
                h1,h2,_=self.predict(xx,np.full(len(xx),a))
                out[left:left+len(xx),a]=risk_tail(h1,h2,self.edges)[0]
        return out


def risk_tail(h1,h2,edges):
    """F(tau) and future conditional CIF at each interval RIGHT endpoint."""
    n,m=h1.shape; tail=np.zeros(n); right=np.zeros((n,m)); delta=np.diff(edges)
    for j in range(m-1,-1,-1):
        right[:,j]=tail
        total=h1[:,j]+h2[:,j]; survive=np.exp(-total*delta[j])
        p=np.divide(h1[:,j],total,out=np.zeros(n),where=total>0)
        tail=p*(-np.expm1(-total*delta[j]))+survive*tail
    return tail,right


def exp_integral(rate,duration):
    """Integral_0^duration exp(-rate*s) ds, with exact zero-rate limit."""
    return np.divide(-np.expm1(-rate*duration),rate,out=np.array(duration,dtype=float,copy=True),where=rate>1e-14)


def aipcw_pseudo(h1,h2,hc,t,kind,edges,g_floor):
    """Counting-process augmented censoring pseudo-outcome, continuous event times.

    U=F1(tau)+sum_j int[1{j=1}-v(u)]/max(G(u),floor) dM_j(u).
    v(u)=(F1(tau)-F1(u))/S(u). Exact analytic integration within hazard
    intervals; censor/death times are NOT rounded into binary outcomes.
    At g_floor=0 it also equals I(CVD)/G + int v/G dM_c.
    """
    f,right=risk_tail(h1,h2,edges); u=f.copy(); n,m=h1.shape
    cumulative_c=np.zeros(n); gstop=np.ones(n); clipped_time=np.zeros(n)
    last=np.minimum(np.searchsorted(edges[1:],t,side='left'),m-1)
    for j in range(m):
        width=edges[j+1]-edges[j]
        d=np.clip(t-edges[j],0.,width)
        total=h1[:,j]+h2[:,j]; b=hc[:,j]
        gl=np.exp(-cumulative_c)
        # Exact splitting if the stabilization threshold is crossed mid-interval.
        if g_floor>0:
            crossing=np.where(gl>g_floor,np.inf,0.)
            eligible=(gl>g_floor)&(b>0)
            crossing[eligible]=(np.log(gl[eligible])-np.log(g_floor))/b[eligible]
            d0=np.minimum(d,np.maximum(crossing,0.))
        else:
            if np.any((d>0) & (gl*np.exp(-b*d)<1e-12)):
                raise SafeError('Censoring survival below numerical support; use an explicitly declared stabilization or redesign the target.')
            d0=d
        integral=np.zeros(n)
        use=d0>0
        # c*exp(-total*width)*int exp(total*s)/G(s) ds.
        coef=h1[:,j]-total*right[:,j]
        integral[use]=coef[use]*np.exp(-total[use]*(width-d0[use])+b[use]*d0[use]) / gl[use] * exp_integral(total[use]+b[use],d0[use])
        if g_floor>0:
            tail_d=d-d0
            integral+=coef*np.exp(-total*(width-d))*exp_integral(total,tail_d)/g_floor
            clipped_time+=tail_d
        u-=integral
        at=(last==j)
        if at.any():
            p=np.divide(h1[:,j],total,out=np.zeros(n),where=total>0)
            v=p+(right[:,j]-p)*np.exp(-total*(width-d))
            rawg=gl*np.exp(-b*d); denom=np.maximum(rawg,g_floor)
            gstop[at]=rawg[at]
            evt=at & np.isin(kind,[1,2])
            u[evt]+=((kind[evt]==1).astype(float)-v[evt])/denom[evt]
        cumulative_c+=b*width
    if not np.isfinite(u).all():
        raise SafeError('Nonfinite augmented survival pseudo-outcome.')
    return u,dict(g_at_stop=gstop,clipped_time=clipped_time)


def fit_projection(p,m):
    j=1./np.sum(1./p,axis=1)
    delta=float(np.sum(j*(m@K))/np.sum(j))
    ell=K[None,:]/p
    gamma=np.sum(j[:,None]*ell,axis=0)/np.sum(j)
    return delta,gamma


def arm_scores(p,m,a,u,delta,gamma):
    n=len(p); j=1./np.sum(1./p,axis=1)
    mr=m-(j[:,None]*K[None,:]/p)*((m@K)-delta)[:,None]
    alpha=np.eye(4)[a]/p[np.arange(n),a,None]
    ar=alpha-(K[a]*j/p[np.arange(n),a])[:,None]*(K[None,:]/p-gamma[None,:])
    if not np.allclose(mr@K,delta,atol=1e-9,rtol=1e-9):
        raise SafeError('Constant-interaction projection identity failed.')
    out={
      'full_survival_aipw':m+alpha*(u-m[np.arange(n),a])[:,None],
      'projected_risk_full_aipw':mr+alpha*(u-mr[np.arange(n),a])[:,None],
      'constant_interaction_riesz':mr+ar*(u-mr[np.arange(n),a])[:,None],
    }
    if not all(np.isfinite(v).all() for v in out.values()):
        raise SafeError('Nonfinite causal score.')
    return out,mr


def query_catalog():
    rows=[]
    for a,label in enumerate(ARMS):
        rows.append(('risk_'+label,'risk',np.eye(4)[a]))
    for name,left,right in (
        ('antihtn_vs_neither',1,0),('statin_vs_neither',2,0),('both_vs_neither',3,0),
        ('both_vs_statin',3,2),('both_vs_antihtn',3,1),('statin_vs_antihtn',2,1)):
        c=np.eye(4)[left]-np.eye(4)[right]; rows.append((name,'risk_difference',c))
    rows.extend([('antihtn_main_equal_weight','risk_difference',np.array([-.5,.5,-.5,.5])),
                 ('statin_main_equal_weight','risk_difference',np.array([-.5,-.5,.5,.5])),
                 ('factorial_interaction','risk_difference',K.copy())])
    return rows


def counterfactual_diagnostics(e_raw,e,m,mr,a,gd,fold,source):
    observed=e[np.arange(len(a)),a]
    joint=1./observed/np.maximum(gd['g_at_stop'],gd['floor'])
    d=dict(fold=fold,propensity=source,test_patients=int(len(a)),
        combined_observed_weight_max=float(np.max(joint)), combined_observed_weight_p99=float(np.quantile(joint,.99)),
        min_raw_probability=float(e_raw.min()), min_used_probability=float(e.min()),
        fraction_any_raw_p_below_001=float(np.mean(e_raw.min(axis=1)<.01)),
        fraction_any_raw_p_below_005=float(np.mean(e_raw.min(axis=1)<.05)),
        min_G_at_stop=float(gd['g_at_stop'].min()), fraction_censor_floor_active=float(np.mean(gd['clipped_time']>0)),
        base_risk_min=float(m.min()),base_risk_max=float(m.max()),
        projected_risk_out_of_bounds_fraction=float(np.mean((mr<0)|(mr>1))),
        projected_risks_not_clipped=True)
    for arm in range(4):
        w=(a==arm)/e[:,arm]
        d['ipw_ess_'+ARMS[arm]]=float(w.sum()**2/max(np.square(w).sum(),1e-300))
    return d


def run_fold(data,draw,folds,f,cfg,fold_path):
    train=draw[folds[draw]!=f]; test=draw[folds[draw]==f]
    if not len(train) or not len(test) or np.intersect1d(np.unique(train),np.unique(test)).size:
        raise SafeError('Empty fold or patient leakage across folds.')
    log('  fold %s/%s: train=%s test=%s; real events and censoring' % (f+1,cfg['folds'],len(train),len(test)))
    pre=Basis().fit(data['x'].iloc[train]); xt=pre.transform(data['x'].iloc[train]); xv=pre.transform(data['x'].iloc[test])
    at,av=data['a'][train],data['a'][test]
    edges=np.linspace(0.,cfg['horizon'],cfg['bins']+1)
    models=HazardModels().fit(xt,at,data['t'][train],data['kind'][train],edges,cfg['seed']+f*31,cfg)
    log('    predicting four-arm cumulative incidences')
    mt=models.all_risks(xt,cfg['chunk']); mv=models.all_risks(xv,cfg['chunk'])
    u=np.empty(len(test)); gstop=np.empty(len(test)); clipped=np.empty(len(test))
    for l in range(0,len(test),cfg['chunk']):
        sl=slice(l,min(l+cfg['chunk'],len(test))); rows=test[sl]
        h1,h2,hc=models.predict(xv[sl],av[sl])
        u[sl], gd=aipcw_pseudo(h1,h2,hc,data['t'][rows],data['kind'][rows],edges,cfg['g_floor'])
        gstop[sl]=gd['g_at_stop'];clipped[sl]=gd['clipped_time']
    cats=query_catalog(); contrasts=np.stack([z[2] for z in cats])
    total=[];squares=[];diagnostics=[];proj=[]
    for psource in cfg['propensity']:
        log('    fitting propensity: '+psource)
        prop=fit_propensity(xt,at,psource,cfg['seed']+f*31,cfg)
        pt=smooth_prob(prop.predict_proba(xt),cfg['p_floor']); raw=prop.predict_proba(xv)
        pv=smooth_prob(raw,cfg['p_floor']);delta,gamma=fit_projection(pt,mt)
        sc,mr=arm_scores(pv,mv,av,u,delta,gamma)
        diagnostics.append(counterfactual_diagnostics(raw,pv,mv,mr,av,dict(g_at_stop=gstop,clipped_time=clipped,floor=cfg['g_floor']),f,psource))
        proj.append(dict(fold=f,propensity=psource,delta_risk_interaction=delta,
                         gamma_risk_queries=gamma.tolist()))
        for route in ROUTES:
            s=sc[route]@contrasts.T
            total.append(s.sum(axis=0));squares.append(np.square(s).sum(axis=0))
    result=dict(total=np.stack(total),squares=np.stack(squares),n=np.array(len(test),dtype=np.int64))
    fold_path.mkdir(parents=True,exist_ok=True)
    write_json(fold_path/'diagnostics.json',dict(weights=diagnostics,projection=proj,hazard_fit=models.fit_notes,
        covariate_fit_scope='outer_training_patients',pseudo_outcome_min=float(u.min()),pseudo_outcome_max=float(u.max()),
        train_unique_patients=int(len(np.unique(train))),test_unique_patients=int(len(np.unique(test))),
        unique_patient_overlap=0))
    log('  fold '+str(f+1)+' complete')
    return result


def run_sample(data,draw,folds,cfg,path):
    agg=[run_fold(data,draw,folds,f,cfg,path/('fold_'+str(f))) for f in range(cfg['folds'])]
    n=int(sum(int(z['n']) for z in agg));total=sum(z['total'] for z in agg);squares=sum(z['squares'] for z in agg)
    if n!=len(draw):
        raise SafeError('Cross-fit fold counts do not cover the analysis sample.')
    estimate=total/n
    variance=np.maximum(0.,(squares-total*total/n)/(n-1))
    return estimate,np.sqrt(variance/n)


def result_table(estimate,se,cfg,n):
    rows=[]; cats=query_catalog(); row=0
    for psource in cfg['propensity']:
        for route in ROUTES:
            for q,(name,kind,c) in enumerate(cats):
                theta=float(estimate[row,q]); s=float(se[row,q])
                rows.append(dict(propensity=psource,method=route,label=LABELS[route],query=name,quantity=kind,
                    horizon_years=cfg['horizon'],n_patients=n,estimate=theta,se_score=s,
                    ci_low_score=theta-1.95996398454*s,ci_high_score=theta+1.95996398454*s,
                    estimate_percent_or_pp=100*theta,se_percent_or_pp=100*s,
                    ci_low_percent_or_pp=100*(theta-1.95996398454*s),ci_high_percent_or_pp=100*(theta+1.95996398454*s),
                    interval_type='nominal_pointwise_score_Wald',
                    causal_status='WORKING_ASSUMPTIONS_UNVERIFIED',
                    survival_extension=route!='full_survival_aipw',
                    estimate_outside_natural_range=bool((kind=='risk' and not 0<=theta<=1) or (kind=='risk_difference' and name!='factorial_interaction' and not -1<=theta<=1))))
            row+=1
    return pd.DataFrame(rows)


def write_support(out,data,cfg):
    rows=[]
    for a in range(4):
        for year in range(0,int(np.ceil(cfg['horizon']))+1):
            use=data['a']==a
            rows.append(dict(arm=ARMS[a],time_years=year,cohort_n=suppress(use.sum()),
                             at_risk_n=suppress(np.sum(use & (data['t']>=year)))))
    write_csv(out/'support_by_arm_time.csv',pd.DataFrame(rows))
    rows=[]
    for y in np.unique(data['index_year']):
        use=data['index_year']==y
        observable=np.isin(data['kind'],[0,1,2])
        rows.append(dict(index_calendar_year=int(y),cohort_n=suppress(use.sum()),
            reached_horizon_event_free_n=suppress(np.sum(use & (data['kind']==0))),
            cvd_before_horizon_n=suppress(np.sum(use & (data['kind']==1))),
            death_first_n=suppress(np.sum(use & (data['kind']==2))),
            censored_early_n=suppress(np.sum(use & (data['kind']==3))),
            no_event_free_tail_support=bool(not np.any(use & (data['t']>=cfg['horizon'])))))
    write_csv(out/'support_by_index_year.csv',pd.DataFrame(rows))


def bootstrap_summary(out,point,cfg,n,B,samples,records):
    write_csv(out/'bootstrap_inventory.csv',pd.DataFrame(records))
    table=result_table(point,np.full_like(point,np.nan),cfg,n)
    table=table.drop(columns=[c for c in table if c.startswith('se_') or c.startswith('ci_') or c=='interval_type'])
    table['bootstrap_planned']=B;table['bootstrap_complete']=len(samples)
    if len(samples)==B and B>=2:
        bs=np.std(np.stack(samples),axis=0,ddof=1).ravel()
        table['se_refit']=bs;table['ci_low_refit']=table.estimate-1.95996398454*bs;table['ci_high_refit']=table.estimate+1.95996398454*bs
        table['interval_type']='nominal_full_refit_bootstrap_normal_original_center'
    else:
        table['se_refit']=np.nan;table['ci_low_refit']=np.nan;table['ci_high_refit']=np.nan
        table['interval_type']='UNAVAILABLE_until_all_planned_draws_complete'
    write_csv(out/'estimates_refit_bootstrap.csv',table)
    return records


def print_report(out):
    out=Path(out)
    p=out/'estimates.csv'
    if not p.exists():
        print('No completed causal fit in this output directory yet.');return
    table=pd.read_csv(p); config=json.loads((out/'manifest.json').read_text())['config']
    print('\nCVD REAL RECORDED OUTCOME FIT | NO SEMI-SYNTHETIC Y')
    print('WORKING DEFINITIONS UNVERIFIED; causal interpretation requires the documented assumptions.')
    print('Death is competing; early censoring is not a known non-event. No effect RMSE/coverage is available.')
    print('Pointwise nominal score-Wald intervals; no automatic validity or superiority label.\n')
    for psource in config['propensity']:
        print('Propensity = '+psource+' | horizon = '+str(config['horizon'])+' years')
        view=table[(table.propensity==psource)&table.quantity.eq('risk_difference')]
        for route in ROUTES:
            print('\n'+LABELS[route])
            for _,r in view[view.method==route].iterrows():
                print('%-29s %9.3f [%9.3f,%9.3f] pp' %(r.query,r.estimate_percent_or_pp,r.ci_low_percent_or_pp,r.ci_high_percent_or_pp))
        print()
    print('All arm risks, contrasts, SEs and intervals: '+str(p))
    print('Check support_by_index_year.csv, overlap.csv and quality_counts.json before clinical interpretation.')
    print('Patient-level *_LOCAL files must remain in the approved environment.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',help='CVD input file; defaults to CVD_DATA_PATH when set.')
    p.add_argument('--member',default='strict_cohort2/all_data.csv')
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'results_cvd_real'))
    p.add_argument('--accept-working-assumptions',action='store_true',help='Run a research fit under the documented, not clinically verified, field definitions.')
    p.add_argument('--propensity',nargs='+',choices=['logit','hgb'],default=['logit','hgb'])
    p.add_argument('--hazard-learner',choices=['poisson','hgb'],default='hgb')
    p.add_argument('--horizon',type=float,default=5.)
    p.add_argument('--bins',type=int,default=10,help='Piecewise-constant hazard intervals; exact event times are retained.')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--seed',type=int,default=640021)
    p.add_argument('--iterations',type=int,default=120)
    p.add_argument('--min-leaf',type=int,default=100)
    p.add_argument('--min-events',type=int,default=5)
    p.add_argument('--min-arm',type=int,default=20)
    p.add_argument('--logit-C',type=float,default=1.)
    p.add_argument('--poisson-alpha',type=float,default=.0001)
    p.add_argument('--p-floor',type=float,default=.005,help='Fixed uniform smoothing floor; changes weights, not the cohort.')
    p.add_argument('--g-floor',type=float,default=.02,help='Fixed censor-survival lower bound; active regularization can affect consistency.')
    p.add_argument('--ties',choices=['hold','cvd_first','death_first'],default='hold')
    p.add_argument('--index-end',help='Optional PRE-SPECIFIED baseline calendar cutoff, YYYY-MM-DD; never derived from outcome availability.')
    p.add_argument('--max-held-fraction',type=float,default=.01,help='Operational guard against large silent record exclusions, not a statistical validity threshold.')
    p.add_argument('--chunk',type=int,default=1024)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--bootstrap',type=int,default=0,help='Optional patient pairs full-refit bootstrap; 0 means none.')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args(argv);out=Path(args.output).expanduser().resolve()
    if args.report_only:
        print_report(out);return 0
    if not args.accept_working_assumptions:
        raise SafeError('Use --accept-working-assumptions for the requested RESEARCH fit. This acknowledges assumptions, not verified clinical definitions. See README_ZH.md.')
    args.input=resolve_input_path(args.input)
    if not (0<args.horizon<=5 and args.bins>=1 and args.folds>=2 and 0<=args.p_floor<.25 and 0<=args.g_floor<1):
        raise SafeError('Invalid horizon/grid/folds/probability regularization settings.')
    if min(args.iterations,args.min_leaf,args.min_events,args.min_arm,args.chunk,args.threads)<1:
        raise SafeError('Positive fitting settings required.')
    if args.bootstrap not in (0,) and args.bootstrap<2:
        raise SafeError('Bootstrap requires at least 2 planned draws; this is not a coverage certification.')
    if args.logit_C<=0 or args.poisson_alpha<0:
        raise SafeError('Invalid regularization parameters.')
    if len(set(args.propensity))!=len(args.propensity):
        raise SafeError('Duplicate propensity specifications.')
    if not 0<=args.max_held_fraction<1:
        raise SafeError('Invalid quality-hold guard.')
    os.umask(0o077)
    cfg={key:getattr(args,key) for key in ('horizon','bins','folds','seed','propensity','hazard_learner',
        'iterations','min_leaf','min_events','min_arm','logit_C','poisson_alpha','p_floor','g_floor',
        'ties','index_end','max_held_fraction','chunk')}
    out.mkdir(parents=True,exist_ok=True)
    os.chmod(out,0o700)
    with threadpool_limits(limits=args.threads):
        log('Reading original CVD archive; no database connection, simulated treatment, or simulated outcome.')
        raw,provenance=read_input(args.input,args.member)
        data,audit,ledger=build_cohort(raw,args.horizon,args.ties,args.index_end)
        result_paths=[out/name for name in ('estimates.csv','overlap.csv',
            'estimates_refit_bootstrap.csv','bootstrap_inventory.csv','bootstrap_plan.json')]
        if Path(args.input).expanduser().resolve() in result_paths:
            raise SafeError('A result filename would overwrite the input; choose another output directory.')
        for path in result_paths:
            path.unlink(missing_ok=True)
        base=dict(version=VERSION,config=cfg,input=provenance,
            runtime=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,sklearn=SK_VERSION))
        write_json(out/'manifest.json',base)
        write_json(out/'assumptions.json',ASSUMPTIONS)
        write_json(out/'quality_counts.json',audit)
        # Local-only ledger never appears in stdout, figures or sharable reports.
        ledger.to_csv(out/'cohort_ledger_LOCAL.csv.gz',index=False,compression='gzip')
        os.chmod(out/'cohort_ledger_LOCAL.csv.gz',0o600)
        if audit['held_rows']/len(raw)>args.max_held_fraction:
            raise SafeError('Quality holds exceed the explicit guard. Review local reasons; no silently restricted large subset was fitted.')
        n=len(data['a'])
        if n<100 or min(audit['arm_counts'])<args.min_arm:
            raise SafeError('Too few retained patients or a recorded arm below the operational support guard.')
        # Folds are fixed by baseline arm, never by outcome, follow-up duration or success.
        foldvec=np.empty(n,int)
        for f,(_,test) in enumerate(StratifiedKFold(args.folds,shuffle=True,random_state=args.seed).split(np.zeros(n),data['a'])):
            foldvec[test]=f
        save_npz(out/'fold_membership_LOCAL.npz',fold=foldvec,source_row=data['source_rows'])
        write_support(out,data,cfg)
        write_csv(out/'query_catalog.csv',pd.DataFrame([dict(query=name,quantity=kind,**dict(zip(ARMS,c))) for name,kind,c in query_catalog()]))
        log('Analysis patients: {:,}; quality holds: {}; baseline-calendar exclusions: {}'.format(n,suppress(audit['held_rows']),suppress(audit['baseline_calendar_excluded'])))
        estimate,se=run_sample(data,np.arange(n),foldvec,cfg,out/'original')
        write_csv(out/'estimates.csv',result_table(estimate,se,cfg,n))
        diagnostics=[]
        for f in range(args.folds):
            diagnostics.extend(json.loads((out/'original'/('fold_'+str(f))/'diagnostics.json').read_text())['weights'])
        write_csv(out/'overlap.csv',pd.DataFrame(diagnostics))
        if args.bootstrap:
            bplan=dict(B=args.bootstrap,seed=args.seed+900000,
                resampling='patient_pairs_with_original_patient_fold_membership',center='original_point_estimate',
                no_redraw_or_success_only_SE=True)
            planpath=out/'bootstrap_plan.json'
            write_json(planpath,bplan)
            samples=[];records=[]
            for b in range(args.bootstrap):
                dest=out/'bootstrap'/('draw_%04d'%b)
                log('Bootstrap draw %s/%s: complete nuisance and projection refit'%(b+1,args.bootstrap))
                rng=np.random.default_rng(np.random.SeedSequence([args.seed,900000,b]))
                draw=rng.integers(0,n,n)
                try:
                    theta,_=run_sample(data,draw,foldvec,cfg,dest)
                    samples.append(theta)
                    records.append(dict(draw=b,status='complete'))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    message=str(exc) if isinstance(exc,SafeError) else 'Unexpected error; inspect environment locally. No raw exception values exported.'
                    records.append(dict(draw=b,status='failed',error_type=type(exc).__name__,message=message))
                    log('Bootstrap slot failed; retained, not redrawn.')
            bootstrap_summary(out,estimate,cfg,n,args.bootstrap,samples,records)
        print_report(out)
        log('Output: '+str(out))
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except SafeError as e:
        print('STOP: '+str(e),file=sys.stderr);raise SystemExit(2)
    except KeyboardInterrupt:
        print('Interrupted. Run the command again to refit.',file=sys.stderr);raise SystemExit(130)
