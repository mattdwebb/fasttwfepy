"""Internal additive fixed-effect absorber. Algorithm/spec version: FEABSORB-1.

Weighted cyclic orthogonal projections. No dummy design is constructed.
Weights are positive numerical least-squares weights, not inference corrections.
"""
import numpy as np

VERSION='FEABSORB-1'
class ConvergenceError(RuntimeError):
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        super().__init__('fixed-effect absorption did not converge: '+str(diagnostics))

class FEAbsorber:
    def __init__(self,*groups,weights=None):
        if not groups:raise ValueError('at least one fixed effect is required')
        self.n=len(groups[0]);self.codes=[];self.levels=[];self.maps=[];self.counts=[];self.orders=[];self.starts=[]
        if self.n==0:raise ValueError('empty fixed-effect structure')
        self.weights=np.ones(self.n) if weights is None else np.asarray(weights,float).copy()
        if self.weights.shape!=(self.n,) or not np.isfinite(self.weights).all() or np.any(self.weights<=0):raise ValueError('weights must be finite and strictly positive; zero weights are rejected')
        self.weights/=self.weights.max()
        self.sums=[]
        for group in groups:
            values=list(group)
            if len(values)!=self.n:raise ValueError('fixed-effect lengths differ')
            mapping={};code=[]
            for v in values:
                if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):raise ValueError('missing/nonfinite FE label')
                if v not in mapping:mapping[v]=len(mapping)
                code.append(mapping[v])
            code=np.asarray(code,dtype=np.intp);self.codes.append(code);self.maps.append(mapping);self.levels.append(len(mapping))
            count=np.bincount(code);self.counts.append(count);self.sums.append(np.bincount(code,weights=self.weights))
            order=np.argsort(code,kind='stable');self.orders.append(order);self.starts.append(np.r_[0,np.cumsum(count)[:-1]])
        if any(np.any(w<=0) for w in self.sums):raise ValueError('zero total group weight')
        self.singletons=[int(np.sum(c==1)) for c in self.counts]
        self.components=None;self.component_codes=None
        if len(groups)==2:
            parent=np.arange(sum(self.levels));offset=self.levels[0]
            def find(x):
                while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
                return x
            for a,b in set(zip(self.codes[0],self.codes[1])):
                a=find(a);b=find(offset+b)
                if a!=b:parent[b]=a
            self.component_codes=np.array([find(i) for i in range(len(parent))]);self.components=len(set(self.component_codes))
        self.rank=self.levels[0] if len(groups)==1 else (sum(self.levels)-self.components if len(groups)==2 else None)

    def _means(self,r,k,method='bincount'):
        if method=='reduceat':s=np.add.reduceat((r*self.weights[:,None])[self.orders[k]],self.starts[k],axis=0)
        elif method=='bincount':s=np.column_stack([np.bincount(self.codes[k],weights=self.weights*r[:,j],minlength=self.levels[k]) for j in range(r.shape[1])])
        else:raise ValueError('unknown accumulation backend')
        return s/self.sums[k][:,None]

    def residualize(self,values,*,tol=1e-12,max_iter=10000,return_diagnostics=False,return_effects=False,raise_on_nonconvergence=True,method='bincount'):
        raw=np.asarray(values,float);vector=raw.ndim==1;r=raw[:,None].copy() if vector else raw.copy()
        if r.ndim!=2 or r.shape[0]!=self.n or r.shape[1]==0 or not np.isfinite(r).all():raise ValueError('values must be a finite vector or N x P matrix')
        if not np.isfinite(tol) or tol<=0 or max_iter<1 or int(max_iter)!=max_iter:raise ValueError('invalid convergence settings')
        scale=np.max(np.abs(r),axis=0);scale[scale==0]=1
        effects=[np.zeros((g,r.shape[1])) for g in self.levels] if return_effects else None
        metric=float('inf')
        for iteration in range(1,int(max_iter)+1):
            for k,code in enumerate(self.codes):
                mean=self._means(r,k,method);r-=mean[code]
                if effects is not None:effects[k]+=mean
            metric=max(float(np.max(np.abs(self._means(r,k,method))/scale)) for k in range(len(self.codes)))
            if metric<=tol:break
        diagnostic=dict(iterations=iteration,converged=metric<=tol,metric=metric,tolerance=tol,max_iterations=max_iter,singleton_groups=self.singletons,components=self.components,algorithm=VERSION)
        self.last_diagnostics=diagnostic
        if not diagnostic['converged'] and raise_on_nonconvergence:raise ConvergenceError(diagnostic)
        result=r[:,0] if vector else r
        if return_effects:return result,diagnostic,effects
        return (result,diagnostic) if return_diagnostics else result

    def predict_effects(self,effects,*groups):
        if len(groups)!=len(self.codes):raise ValueError('FE dimensions differ')
        codes=[]
        for group,mapping in zip(groups,self.maps):
            try:codes.append(np.asarray([mapping[v] for v in group]))
            except KeyError as e:raise ValueError('prediction FE level absent from estimation sample') from e
        if self.component_codes is not None and np.any(self.component_codes[codes[0]]!=self.component_codes[self.levels[0]+codes[1]]):raise ValueError('prediction joins disconnected estimation components')
        return sum(e[c] for e,c in zip(effects,codes))
