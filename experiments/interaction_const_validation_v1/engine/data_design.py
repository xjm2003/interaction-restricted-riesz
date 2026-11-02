"""Read only archived pre-treatment design inputs. No outcome model is imported."""
from __future__ import annotations
import bootstrap
import numpy as np
import pandas as pd
import cvd_adapter, phq_adapter, baseline_experiment as baseline

BITS=np.array([[0,0],[1,0],[0,1],[1,1]],int)

def load_design(dataset,input_path=None,member=None,phq_root=None,smoke=False,patients=240):
    if smoke:
        data=(cvd_adapter.load_mock(patients=patients,seed=912701) if dataset=='cvd'
              else phq_adapter.load_mock(patients=patients,seed=912702))
    elif dataset=='cvd':
        data=cvd_adapter.load_real(input_path,member=member)
    else:
        data=phq_adapter.load_real(phq_root or phq_adapter.DEFAULT_ROOT)
    expected=(['neither','antihtn_only','statin_only','both'] if dataset=='cvd' else
              ['no_active_treatment','antidepressant_only','psychotherapy_only','combined'])
    if list(data['core'].ACTION_ORDER)!=expected:
        raise ValueError('Unknown treatment order: factorial mapping needs review.')
    steps,p,pbar=baseline.population_arrays(data)
    phi,feature_info=build_features(data,dataset)
    return data,steps,p,pbar,phi,feature_info

def build_features(data,dataset):
    """Freeze clinical feature names, never select columns by oracle gains."""
    if dataset=='cvd':
        wanted=['age[linear]','sbp[linear]','eGFR[linear]']
        meta=data['input_metadata']; names=meta['basis_feature_names']
        dropped=meta.get('basis_dropped_feature_names',[])
        cols=[]; status=[]
        for name in wanted:
            if name in names:
                cols.append(np.asarray(data['x_population'][:,names.index(name)],float)); status.append('present')
            elif name in dropped:
                cols.append(np.zeros(len(data['population']))); status.append('constant_in_original_basis')
            else: raise ValueError('Required prespecified modifier absent: '+name)
    else:
        wanted=['baseline_phq9','prior_antidepressant_exposure_count','prior_psychotherapy_exposure_count']
        fills=[15.,0.,0.]; cols=[]; status=[]
        for name,fill in zip(wanted,fills):
            if name not in data['population']: raise ValueError('Required prespecified history variable absent: '+name)
            s=pd.to_numeric(data['population'][name],errors='coerce')
            cols.append(s.fillna(fill).to_numpy(float))
            status.append('prespecified_missing_fill' if s.isna().any() else 'present')
    z=np.column_stack(cols)
    if not np.isfinite(z).all(): raise ValueError('Nonfinite design modifiers.')
    mean=z.mean(0); sd=z.std(0); varying=sd>1e-12
    zz=np.zeros_like(z); zz[:,varying]=(z[:,varying]-mean[varying])/sd[varying]
    phi=np.column_stack([np.ones(len(z)),zz])
    # Do not export centering coefficients or individual features.
    info={'columns':['intercept']+wanted,'availability':status,'varying':varying.tolist(),
          'scope':'full empirical covariate population, no A or outcome use',
          'separate_landmark_coefficients':True,'risk_proportional_model_included':False}
    return phi,info

def cluster_ids(data,dataset):
    if dataset=='cvd': return np.arange(len(data['population']))
    ids=pd.factorize(data['population'][data['core'].ID_COL],sort=False)[0]
    st=data['population']['step'].to_numpy(int)
    for t in range(4):
        if not np.array_equal(np.sort(ids[st==t]),np.arange(ids.max()+1)):
            raise ValueError('PHQ-9 profiles must have one row per landmark.')
    return ids

def make_scale_fn(data,rows):
    """Return query multiplier chunks with exact original normalization."""
    rows=np.asarray(rows,int); steps=data['population']['step'].to_numpy(int)
    qids=np.arange(len(data['query_library']))
    def get(start,stop):
        rr=rows[start:stop]
        return baseline.query_scale(data['queries'],qids,rr,steps[rr],data['core'].P_LANDMARK)
    return get

def classify_contrast(c):
    c=np.asarray(c,float)
    if np.allclose(c,0,atol=1e-12): return 'zero_query'
    if not np.isclose(c.sum(),0,atol=1e-10): return 'policy_or_level'
    inds=np.flatnonzero(np.abs(c)>1e-10)
    if len(inds)==2 and np.isclose(c[inds[0]],-c[inds[1]]):
        left,right=BITS[inds]
        diff=np.abs(left-right)
        if diff.sum()==1:
            axis=int(np.argmax(diff)); other=1-axis
            return f'simple_{"B" if axis==0 else "S"}_at_{"S" if other==1 else "B"}{left[other]}'
        return 'both_vs_neither' if set(inds)=={0,3} else 'single_vs_single'
    mainB=np.array([-.5,.5,-.5,.5]); mainS=np.array([-.5,-.5,.5,.5]); ki=np.array([1.,-1.,-1.,1.])
    for name,v in [('main_B',mainB),('main_S',mainS),('interaction_contrast',ki)]:
        coeff=c@v/(v@v)
        if np.allclose(c,coeff*v,atol=1e-10): return name
    return 'other_contrast'

def catalog(data):
    q=data['queries']; lib=data['query_library']; out=[]
    split={int(j):s for s in ('train','selection','held_out') for j in q['indices'][s]}
    for i in range(len(lib)):
        row={'query_index':i,'query_id':str(lib.iloc[i]['query_id']), 'query_split':split[i],
             'contrast_family':classify_contrast(q['c'][i]),
             'contrast_name':str(lib.iloc[i]['contrast']) if 'contrast' in lib else classify_contrast(q['c'][i]),
             'tilt_index':int(q['inverse_eta'][i]),'is_treatment_contrast':bool(abs(q['c'][i].sum())<1e-10)}
        row.update({f'c_{a}':float(q['c'][i,a]) for a in range(4)})
        row.update({f'h_{t}':float(q['h'][i,t]) for t in range(q['h'].shape[1])})
        out.append(row)
    return out

def draw_actions_only(data,n_rows,seed,dataset):
    """Same action/covariate RNG as own-data v1, but never construct Y."""
    st,p,pbar=baseline.population_arrays(data)
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(n_rows),831]))
    if dataset=='phq9':
        if n_rows%4: raise ValueError('PHQ-9 n_rows must be divisible by 4.')
        pop=data['population']; profiles=[]
        for ii in pop.groupby(data['core'].ID_COL,sort=False).indices.values():
            ii=np.asarray(ii,int); ii=ii[np.argsort(st[ii])]
            if len(ii)!=4 or not np.array_equal(st[ii],np.arange(4)): raise ValueError('Incomplete profile.')
            profiles.append(ii)
        profiles=np.stack(profiles); selected=rng.integers(0,len(profiles),n_rows//4)
        rows=profiles[selected].reshape(-1); clusters=np.repeat(np.arange(n_rows//4),4)
    else:
        rows=rng.integers(0,len(st),n_rows); clusters=np.arange(n_rows)
    a=data['core'].draw_actions(p[rows],rng)
    return dict(rows=rows,x=np.asarray(data['x_population'][rows],float),steps=st[rows],
                actions=a,clusters=clusters,pbar=pbar)

def group_folds(ids,k,seed):
    groups=np.unique(ids)
    if len(groups)<2*k: raise ValueError('Too few independent profiles for folds.')
    groups=np.random.default_rng(seed).permutation(groups)
    return [np.flatnonzero(np.isin(ids,g)) for g in np.array_split(groups,k)]
