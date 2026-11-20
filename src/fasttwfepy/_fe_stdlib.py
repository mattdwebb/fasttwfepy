"""Internal additive fixed-effect absorber. Algorithm/spec version: FEABSORB-1.
Experimental standard-library-only implementation; never imports NumPy/pandas.
"""
import math

VERSION='FEABSORB-1'
class FEAbsorber:
    def __init__(self,*groups,weights=None):
        if not groups or not len(groups[0]):raise ValueError('nonempty fixed effects required')
        self.n=len(groups[0]);self.weights=[1.]*self.n if weights is None else list(map(float,weights))
        if len(self.weights)!=self.n or any(not math.isfinite(w) or w<=0 for w in self.weights):raise ValueError('finite positive weights required; zero weights rejected')
        top=max(self.weights);self.weights=[w/top for w in self.weights];self.members=[];self.sums=[];self.singletons=[]
        for group in groups:
            if len(group)!=self.n:raise ValueError('length mismatch')
            members={}
            for i,v in enumerate(group):
                if v is None or isinstance(v,float) and not math.isfinite(v):raise ValueError('missing FE label')
                members.setdefault(v,[]).append(i)
            idx=list(members.values());self.members.append(idx);self.sums.append([math.fsum(self.weights[i] for i in ids) for ids in idx]);self.singletons.append(sum(len(ids)==1 for ids in idx))
    def residualize(self,values,*,tol=1e-12,max_iter=10000,return_diagnostics=False,raise_on_nonconvergence=True):
        vector=not isinstance(values[0],(list,tuple));r=[[float(v)] for v in values] if vector else [list(map(float,row)) for row in values]
        p=len(r[0])
        if len(r)!=self.n or not p or any(len(row)!=p or any(not math.isfinite(v) for v in row) for row in r):raise ValueError('invalid values')
        if tol<=0 or not math.isfinite(tol) or max_iter<1 or int(max_iter)!=max_iter:raise ValueError('invalid convergence settings')
        scale=[max(abs(row[j]) for row in r) or 1. for j in range(p)]
        for iteration in range(1,max_iter+1):
            for groups,sums in zip(self.members,self.sums):
                for ids,sw in zip(groups,sums):
                    means=[math.fsum(self.weights[i]*r[i][j] for i in ids)/sw for j in range(p)]
                    for i in ids:
                        for j in range(p):r[i][j]-=means[j]
            metric=0.
            for groups,sums in zip(self.members,self.sums):
                for ids,sw in zip(groups,sums):
                    for j in range(p):metric=max(metric,abs(math.fsum(self.weights[i]*r[i][j] for i in ids)/sw)/scale[j])
            if metric<=tol:break
        d=dict(iterations=iteration,converged=metric<=tol,metric=metric,tolerance=tol,max_iterations=max_iter,singleton_groups=self.singletons,algorithm=VERSION)
        self.last_diagnostics=d
        if not d['converged'] and raise_on_nonconvergence:raise RuntimeError('fixed-effect nonconvergence: '+str(d))
        result=[row[0] for row in r] if vector else r
        return (result,d) if return_diagnostics else result

def solve(matrix,rhs):
    """Small dense partial-pivoting solve; rejects singular systems explicitly."""
    n=len(matrix);a=[list(map(float,row))+[float(b)] for row,b in zip(matrix,rhs)]
    if len(rhs)!=n or any(len(row)!=n+1 for row in a) or any(not math.isfinite(v) for row in a for v in row):raise ValueError('invalid square system')
    scale=max(abs(v) for row in a for v in row[:-1])
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:abs(a[i][k]))
        if abs(a[pivot][k])<=math.ulp(1.)*max(1,n)*scale:raise ValueError('singular system')
        a[k],a[pivot]=a[pivot],a[k]
        for i in range(k+1,n):
            factor=a[i][k]/a[k][k]
            for j in range(k,n+1):a[i][j]-=factor*a[k][j]
    x=[0.]*n
    for i in range(n-1,-1,-1):x[i]=(a[i][n]-math.fsum(a[i][j]*x[j] for j in range(i+1,n)))/a[i][i]
    return x
