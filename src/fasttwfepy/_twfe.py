"""Internal FE setup and state-period sufficient statistics for repeated TWFE fits."""
import numpy as np
from ._fe import FEAbsorber,ConvergenceError
from ._inference import InferenceError

def prepare(d):
    G=len(d.states);T=len(d.times);k=len(d.covariates);p=T-1+k;C=G*T
    y=d.data[d.outcome].to_numpy(float);x=d.data[d.covariates].to_numpy(float)
    d._fe_setup=FEAbsorber(d._rowstate)
    within=d._fe_setup.residualize(np.column_stack([y,x]));yw=within[:,0];xw=within[:,1:]
    ns=d.counts.sum(1);shares=d.counts/ns[:,None]
    sy=np.bincount(d._rowcell,weights=yw,minlength=C).reshape(G,T)
    sx=np.empty((G,T,k))
    for j in range(k):sx[:,:,j]=np.bincount(d._rowcell,weights=xw[:,j],minlength=C).reshape(G,T)
    # Small time basis at the state-period level; never an observation x FE-level design.
    ztime=(np.eye(T)[:,1:][None,:,:]-shares[:,None,1:])*d.counts[:,:,None]
    zsum=np.concatenate([ztime,sx],axis=2)
    lhs=np.zeros((G,p,p));rhs=np.zeros((G,p))
    for h in range(G):lhs[h,:T-1,:T-1]=np.diag(d.counts[h,1:])-np.outer(d.counts[h,1:],d.counts[h,1:])/ns[h]
    rhs[:,:T-1]=sy[:,1:]
    if k:
        lhs[:,:T-1,T-1:]=sx[:,1:,:];lhs[:,T-1:,:T-1]=sx[:,1:,:].transpose(0,2,1)
        for j in range(k):
            rhs[:,T-1+j]=np.bincount(d._rowstate,weights=xw[:,j]*yw,minlength=G)
            for l in range(k):lhs[:,T-1+j,T-1+l]=np.bincount(d._rowstate,weights=xw[:,j]*xw[:,l],minlength=G)
    d._twfe_stats=(ns,sy,zsum,lhs,rhs,lhs.sum(0),rhs.sum(0));d._nuisance_cache={}

def absorbed_fit(d,a,omit,procedure,observed):
    keep=np.ones(len(d.data),bool) if omit is None else d._rowstate!=omit
    D=((a[d._rowstate]>0)&(d.times[d._rowtime]>=a[d._rowstate])).astype(float)
    x=d.data[d.covariates].to_numpy(float)[keep]
    if procedure=='add':
        actual=np.asarray(d.assignment if observed is None else observed)
        x=np.column_stack([x,((actual[d._rowstate]>0)&(d.times[d._rowtime]>=actual[d._rowstate]))[keep]])
    x=np.column_stack([x,D[keep]])
    try:
        q,r=np.linalg.qr(x,mode='reduced')
        if np.linalg.matrix_rank(r)<x.shape[1]:raise InferenceError('rank_deficient')
        fe=FEAbsorber(d._rowstate[keep],d._rowtime[keep]);v=fe.residualize(np.column_stack([d.data[d.outcome].to_numpy(float)[keep],q]))
        if np.any(np.linalg.norm(v[:,1:],axis=0)<1e-12):raise InferenceError('rank_deficient')
        beta,_,rank,_=np.linalg.lstsq(v[:,1:],v[:,0],rcond=None)
        if rank<x.shape[1]:raise InferenceError('rank_deficient')
        return float(np.linalg.solve(r,beta)[-1])
    except ConvergenceError as e:raise InferenceError('fe_nonconvergence',str(e)) from e

def fit(d,a,omit,procedure,observed):
    ns,sy,zsum,lhs,rhs,total,totaly=d._twfe_stats;keep=np.ones(len(ns),bool)
    if omit is not None:keep[omit]=False
    D=(a[:,None]>0)&(d.times[None,:]>=a[:,None])
    if not (D&keep[:,None]).any():raise InferenceError('empty_comparison_group')
    if omit not in d._nuisance_cache:
        xx=total if omit is None else total-lhs[omit];xy=totaly if omit is None else totaly-rhs[omit]
        scale=np.sqrt(np.maximum(np.diag(xx),0));inv=None
        if (scale>0).all():
            normalized=xx/scale[:,None]/scale[None,:];eigen=np.linalg.eigvalsh(normalized)
            if eigen[0]>1e-9*eigen[-1]:inv=np.linalg.inv(normalized)/scale[:,None]/scale[None,:]
        d._nuisance_cache[omit]=(inv,xy)
    inv,xy=d._nuisance_cache[omit]
    if inv is None:return absorbed_fit(d,a,omit,procedure,observed)
    assignment=D[:,:,None].astype(float)
    if procedure=='add':
        obs=np.asarray(d.assignment if observed is None else observed)
        actual=(obs[:,None]>0)&(d.times[None,:]>=obs[:,None]);assignment=np.stack([actual,D],axis=2).astype(float)
    assignment=assignment[keep];n=d.counts[keep];zd=np.einsum('gtq,gtp->qp',zsum[keep],assignment)
    raw=np.einsum('gtp,gtq,gt->pq',assignment,assignment,n);sums=np.einsum('gtp,gt->gp',assignment,n)
    dd=raw-(sums.T/ns[keep])@sums;dy=np.einsum('gtp,gt->p',assignment,sy[keep]);schur=dd-zd.T@inv@zd
    if np.linalg.eigvalsh(schur).min()<=1e-10*max(1.,float(np.max(np.diag(raw)))):raise InferenceError('rank_deficient','treatment conditional on nuisance')
    try:return float(np.linalg.solve(schur,dy-zd.T@inv@xy)[-1])
    except np.linalg.LinAlgError as e:raise InferenceError('rank_deficient',str(e)) from e
