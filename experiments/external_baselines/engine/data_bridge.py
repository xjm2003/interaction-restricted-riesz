"""Read archived empirical populations; sample units; define known outcomes.
Patient identifiers remain inside the process. No individual records are saved.
"""
from __future__ import annotations
import numpy as np
import baseline_experiment as baseline


def build_outcome_law(data, effect_scale=1.0):
    """New, explicit standardized continuous simulation; not a clinical endpoint.
    Uses the fixed empirical basis, not treatment probabilities. Frozen before
    any replication and identical for all representer methods.
    """
    x=np.asarray(data['x_population'],float)
    keep=np.flatnonzero(np.std(x,axis=0)>1e-8)[:6]
    if len(keep)<3:
        raise ValueError('At least three varying basis features are required.')
    mean=x[:,keep].mean(0); scale=x[:,keep].std(0)
    z=np.tanh((x[:,keep]-mean)/scale)
    if z.shape[1]<6:
        z=np.pad(z,((0,0),(0,6-z.shape[1])))
    st=data['population']['step'].to_numpy(dtype=int)
    m0=.7*np.sin(1.5*z[:,0])+.5*z[:,1]+.4*z[:,2]*z[:,3]+.2*z[:,4]**2+.15*st
    tau=np.column_stack([np.zeros(len(x)),
        -.35+.20*z[:,0], -.50+.15*z[:,1]-.10*z[:,2],
        -.70+.15*z[:,0]+.15*z[:,1]+.10*z[:,2]*z[:,3]])
    mu=m0[:,None]+float(effect_scale)*tau
    return mu,dict(name='standardized_nonlinear_continuous_v1',effect_scale=float(effect_scale),
                   residual_sd=1.0,clinical_endpoint=False,
                   real_outcomes_used=False,selected_basis_columns=keep.tolist(),
                   scope='real covariates; simulated actions and continuous outcomes')


def finite_population_targets(data,mu,chunk=1024):
    held=np.asarray(data['queries']['indices']['held_out'],int)
    steps=data['population']['step'].to_numpy(dtype=int)
    target=np.zeros(len(held)); N=len(steps)
    for start in range(0,N,chunk):
        rows=np.arange(start,min(start+chunk,N))
        s=baseline.query_scale(data['queries'],held,rows,steps[rows],data['core'].P_LANDMARK)
        target+=np.sum(s*mu[rows,None,:],axis=(0,2))/N
    return target


def draw_sample(data,n_rows,seed,unit):
    steps,p,pbar=baseline.population_arrays(data)
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(n_rows),831]))
    N=len(steps); core=data['core']
    if core.N_STEPS==4 and unit=='patient':
        if n_rows % 4:
            raise ValueError('PHQ-9 patient sampling requires n_rows divisible by four.')
        pop=data['population']
        if core.ID_COL not in pop:
            raise ValueError('Patient-level sampling requires retained patient keys.')
        # Group by the original patient; each empirical profile has exactly 4 steps.
        group=pop.groupby(core.ID_COL,sort=False).indices
        profiles=[]
        for idx in group.values():
            idx=np.asarray(idx,int); ordered=idx[np.argsort(steps[idx])]
            if len(idx)!=4 or not np.array_equal(steps[ordered],np.arange(4)):
                raise ValueError('Incomplete patient landmark profile.')
            profiles.append(ordered)
        profiles=np.stack(profiles)
        sampled=rng.integers(0,len(profiles),size=n_rows//4)
        rows=profiles[sampled].reshape(-1)
        # Copies are independent empirical-population draws with NEW A/Y draws.
        clusters=np.repeat(np.arange(n_rows//4),4)
    else:
        rows=rng.integers(0,N,size=n_rows); clusters=np.arange(n_rows)
    actions=core.draw_actions(p[rows],rng)
    return dict(rows=rows,steps=steps[rows],actions=actions,
                x=np.asarray(data['x_population'][rows],float),clusters=clusters,
                pbar=pbar,p_true=p[rows],sampling_unit='patient_profile' if core.N_STEPS==4 and unit=='patient' else 'iid_row')


def group_folds(clusters,k,seed):
    groups=np.unique(clusters)
    if len(groups)<k*2: raise ValueError('Too few independent units for folds.')
    perm=np.random.default_rng(seed).permutation(groups)
    return [np.flatnonzero(np.isin(clusters,g)) for g in np.array_split(perm,k)]


def observed_artifacts(sample,train_indices,queries,core,pbar,seed):
    groups=sample['clusters'][train_indices]
    inner=group_folds(groups,2,seed)
    slots=[np.asarray(train_indices)[j] for j in inner]+[np.asarray(train_indices)]
    return [core.make_sample_artifact(sample['rows'][j],sample['steps'][j],sample['actions'][j],sample['x'][j],queries,pbar)
            for j in slots]


def empirical_pilot(sample,train,core):
    out=[]
    for t in range(core.N_STEPS):
        take=train[sample['steps'][train]==t]
        if not len(take): raise ValueError('No training observations for a landmark.')
        freq=np.bincount(sample['actions'][take],minlength=4)/len(take)
        out.append(.01+.96*freq)
    return np.asarray(out)
