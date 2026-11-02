"""Exact original-objective logit solver, unchanged from verified mock v1.
CAL = 1/4 of the published Tan--Zhang pairwise multiclass calibration loss.
Only the standalone imports/constants wrapper is adapted for this package.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
KKT_TOLERANCE = 2e-6
MAX_ITERATIONS = 3000

def loss_gradient_logits(f, A, loss):
    """Mean loss and per-observation gradient (gradient already divided by n)."""
    f = np.asarray(f, dtype=float); A = np.asarray(A, dtype=int)
    n = len(f)
    if f.shape != (n,4) or A.shape != (n,) or not n:
        raise ValueError('Four logits and one action per observation required.')
    if loss == 'ce':
        value = float(np.mean(logsumexp(f, axis=1) - f[np.arange(n),A]))
        gradient = softmax(f, axis=1)
        gradient[np.arange(n),A] -= 1.
    elif loss == 'cal':
        d = f - f[np.arange(n),A,None]
        with np.errstate(over='raise', invalid='raise'):
            exponential = np.exp(d)
            terms = d + exponential
            terms[np.arange(n),A] = 0.
            value = float(np.sum(terms) / (8.*n))
            gradient = (1. + exponential) / 8.
            gradient[np.arange(n),A] = 0.
            gradient[np.arange(n),A] = -gradient.sum(axis=1)
    else:
        raise ValueError('Unknown logit loss.')
    gradient /= n
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError('Nonfinite exact logit objective.')
    return value, gradient

def hessian_logits(f, A, loss):
    """Per-observation Hessians of the mean loss, for independent checks."""
    n = len(f); result = np.zeros((n,4,4))
    if loss == 'ce':
        p = softmax(f,axis=1)
        for i in range(n):
            result[i] = (np.diag(p[i])-np.outer(p[i],p[i]))/n
    elif loss == 'cal':
        for i, a in enumerate(A):
            for k in range(4):
                if k != a:
                    v = np.eye(4)[k]-np.eye(4)[a]
                    result[i] += np.exp(f[i,k]-f[i,a])*np.outer(v,v)/(8.*n)
    else:
        raise ValueError(loss)
    return result

def smooth_objective(B, Phi, A, loss, ridge):
    f = np.column_stack([np.zeros(len(Phi)), Phi@B])
    value, gradient = loss_gradient_logits(f,A,loss)
    grad = Phi.T@gradient[:,1:]
    value += .5*ridge*float(np.sum(B[1:]**2))
    grad[1:] += ridge*B[1:]
    return value, grad

def kkt_residual(B, gradient, l1):
    residual = np.maximum(np.abs(gradient[1:])-l1,0.)
    active = B[1:] != 0.
    residual[active] = np.abs(gradient[1:][active]+l1*np.sign(B[1:][active]))
    return max(float(np.max(np.abs(gradient[0]))), float(np.max(residual,initial=0.)))

def fit_coefficients(Phi, A, loss, ridge, l1, initial=None):
    """Exact convex objective; nonfinite line-search trials are rejected."""
    Phi = np.asarray(Phi,dtype=float); A = np.asarray(A,dtype=int)
    if Phi.ndim != 2 or not np.isfinite(Phi).all() or set(np.unique(A)) != set(range(4)):
        raise ValueError('Finite features and all four observed actions are required.')
    if ridge<=0 or l1<0:
        raise ValueError('Positive ridge and nonnegative L1 required.')
    d = Phi.shape[1]; count = (d-1)*3
    if initial is None:
        B = np.zeros((d,3)); counts=np.bincount(A,minlength=4)+1.
        B[0] = np.log(counts[1:]/counts[0])
    else:
        B = np.asarray(initial,dtype=float).copy()
    if B.shape != (d,3):
        raise ValueError('Invalid warm-start dimensions.')
    def pack(b):
        if l1 == 0.:
            return b.ravel()
        slopes=b[1:].ravel()
        return np.r_[b[0],np.maximum(slopes,0.),np.maximum(-slopes,0.)]
    def unpack(v):
        if l1 == 0.:
            return v.reshape(d,3)
        return np.vstack([v[:3],(v[3:3+count]-v[3+count:]).reshape(d-1,3)])
    rejected = 0
    def fun(v):
        nonlocal rejected
        b=unpack(v)
        try:
            value,g=smooth_objective(b,Phi,A,loss,ridge)
            if l1:
                value+=l1*float(v[3:].sum())
                flat=g[1:].ravel(); g=np.r_[g[0],flat+l1,-flat+l1]
            else:
                g=g.ravel()
            if not np.isfinite(value) or not np.isfinite(g).all():
                raise FloatingPointError('Nonfinite trial.')
            return value,g
        except FloatingPointError:
            rejected+=1
            return np.inf,np.zeros_like(v)
    bounds=None if l1==0 else [(None,None)]*3+[(0.,None)]*(2*count)
    history=[]; total_iterations=0
    for attempt in range(2):
        opt=minimize(fun,pack(B),jac=True,method='L-BFGS-B',bounds=bounds,
            options=dict(maxiter=MAX_ITERATIONS,maxls=60,maxcor=30,
                         ftol=1e-14 if attempt==0 else 0.,gtol=5e-8 if attempt==0 else 1e-9))
        B=unpack(opt.x).copy(); total_iterations+=int(opt.nit)
        value,gradient=smooth_objective(B,Phi,A,loss,ridge)
        kkt=kkt_residual(B,gradient,l1)
        history.append(dict(status=int(opt.status),message=str(opt.message),iterations=int(opt.nit),kkt=kkt))
        if np.isfinite(B).all() and kkt<=KKT_TOLERANCE:
            break
    if not np.isfinite(B).all() or kkt>KKT_TOLERANCE:
        raise RuntimeError(f'Uncertified {loss} optimum: KKT={kkt:.6g}, tolerance={KKT_TOLERANCE:.6g}.')
    return B,dict(status='ok',loss=loss,ridge=float(ridge),l1=float(l1),
                  final_kkt_max=kkt,kkt_tolerance=KKT_TOLERANCE,all_final_kkt_passed=True,
                  optimizer_iterations=total_iterations,optimizer_history=history,
                  exact_objective=value+l1*float(np.abs(B[1:]).sum()),
                  rejected_nonfinite_trials=rejected,nonzero_slopes=int(np.count_nonzero(B[1:])),
                  slope_count=int(B[1:].size),n_features=int(d),reference_action=0)
