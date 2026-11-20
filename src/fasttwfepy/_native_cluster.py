"""One-way state-clustered OLS with the built-in Stata regress CR1 convention."""
import numpy as np
from ._fe import FEAbsorber
from ._probability import student_t
from ._inference import InferenceError

def native_cluster(d,*,null=0.,engine='fast'):
    y=d.data[d.outcome].to_numpy(float);a=d.assignment;G=len(d.states);T=len(d.times);N=len(y)
    treatment=((a[d._rowstate]>0)&(d.times[d._rowtime]>=a[d._rowstate])).astype(float)
    cov=d.data[d.covariates].to_numpy(float);K=G+T-1+cov.shape[1]+1
    if N<=K or G<2:raise InferenceError('insufficient_degrees_of_freedom')
    if engine=='fast':
        fe=FEAbsorber(d._rowstate,d._rowtime)
        # Alternating projections can settle at a platform-dependent floating-point
        # noise floor slightly above 1e-13 on large repeated cross sections.  This
        # remains far tighter than the package's certified numerical tolerances.
        residual=fe.residualize(np.column_stack([y,cov,treatment]),tol=5e-13)
        X=residual[:,1:];beta,_,rank,_=np.linalg.lstsq(X,residual[:,0],rcond=None)
        if rank<X.shape[1]:raise InferenceError('rank_deficient')
        error=residual[:,0]-X@beta
    elif engine=='reference':
        X=np.column_stack([np.eye(G)[d._rowstate],np.eye(T)[d._rowtime,1:],cov,treatment])
        beta,_,rank,_=np.linalg.lstsq(X,y,rcond=None)
        if rank<X.shape[1]:raise InferenceError('rank_deficient')
        error=y-X@beta
    else:raise ValueError('engine must be fast or reference')
    direction=np.linalg.solve(X.T@X,np.eye(X.shape[1])[:,-1])
    scores=np.bincount(d._rowstate,weights=(X@direction)*error,minlength=G)
    correction=G/(G-1)*(N-1)/(N-K)
    se=float(np.linalg.norm(scores)*np.sqrt(correction));theta=float(beta[-1]);stat=(theta-null)/se if se>0 else float('nan')
    return dict(estimate=theta,native_cluster_method='OLS_state_cluster_CR1',native_cluster_se=se,
                native_cluster_stat=stat,native_cluster_df=G-1,native_cluster_p=float(2*student_t.sf(abs(stat),G-1)),
                native_cluster_distribution='student_t',cluster_scores=scores,finite_sample_multiplier=correction,
                status='ok' if se>0 else 'zero_standard_error')
