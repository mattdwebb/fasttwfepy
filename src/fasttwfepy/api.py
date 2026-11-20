"""Assignment-explicit estimator with transparent and cached calculation paths."""
import numpy as np
import pandas as pd
from ._inference import InferenceMixin,InferenceError

class Design(InferenceMixin):
    kind='twfe'
    def native_cluster(self,*,null=0.,engine='fast'):
        from ._native_cluster import native_cluster
        return native_cluster(self,null=null,engine=engine)
    def __init__(self,data,*,outcome,cluster,time,assignment,covariates=()):
        self.outcome=outcome;self.cluster=cluster;self.time=time;self.covariates=list(covariates)
        columns=list(dict.fromkeys([outcome,cluster,time,*self.covariates]))
        self.data=data[columns].dropna().copy().reset_index(drop=True)
        if not np.isfinite(self.data[[outcome,time,*self.covariates]].to_numpy(float)).all():raise InferenceError('nonfinite_estimate','input contains infinity')
        self.states=sorted(self.data[cluster].unique());self.times=np.sort(self.data[time].unique().astype(float))
        self.assignment=self._assignment(assignment);G=len(self.states);T=len(self.times)
        self._sindex={s:i for i,s in enumerate(self.states)}
        self._rowstate=np.array([self._sindex[s] for s in self.data[cluster]])
        self._rowtime=np.searchsorted(self.times,self.data[time].to_numpy(float))
        self._rowcell=self._rowstate*T+self._rowtime
        self.n=np.bincount(self._rowcell,minlength=G*T).astype(float)
        if np.any(self.n==0):raise InferenceError('undefined_estimand','strict inference adapter requires observed cells for every state and period')
        self.ysum=np.bincount(self._rowcell,weights=self.data[outcome],minlength=G*T)
        self.means=(self.ysum/self.n).reshape(G,T);self.counts=self.n.reshape(G,T)
        self._cellstate=np.repeat(np.arange(G),T);self._celltime=np.tile(self.times,G)
        self._nuisance_cache={}
        if self.kind in ('twfe','bjs'):self._compile_regression()

    def _compile_regression(self):
        from ._twfe import prepare
        prepare(self)

    def _subtract(self,value):
        data=self.data.copy();D=(self.assignment[self._rowstate]>0)&(self.data[self.time].to_numpy()>=self.assignment[self._rowstate])
        data[self.outcome]-=value*D
        return type(self)(data,outcome=self.outcome,cluster=self.cluster,time=self.time,assignment=self.assignment,covariates=self.covariates)

    def _estimate(self,a,engine,omit,procedure,observed):
        if procedure=='add' and self.kind!='twfe':raise InferenceError('undefined_estimand','Add is defined only for TWFE')
        if engine=='reference' or (self.covariates and self.kind in ('undid','didint')):
            return self._reference(a,omit,procedure,observed)
        if self.kind in ('twfe','bjs'):return self._regression(a,omit,procedure,observed)
        active=np.ones(len(self.states),dtype=bool)
        if omit is not None:active[omit]=False
        control=active&(a==0)
        if not control.any():raise InferenceError('empty_comparison_group')
        numerator=0.;denominator=0.
        for g in np.unique(a[a>0]):
            treated=active&(a==g)
            if not treated.any():raise InferenceError('undefined_estimand','cohort removed')
            base=np.flatnonzero(self.times<g)[-1]
            for t in np.flatnonzero(self.times>=g):
                delta=self.means[:,t]-self.means[:,base]
                weight=self.counts[:,t]+self.counts[:,base]
                c=np.average(delta[control],weights=weight[control]);target=self.counts[treated,t]
                numerator+=np.sum(target*(delta[treated]-c));denominator+=target.sum()
        if denominator<=0:raise InferenceError('undefined_estimand')
        return numerator/denominator

    def _regression(self,a,omit,procedure,observed):
        from ._twfe import fit
        return fit(self,a,omit,procedure,observed)

    def _reference(self,a,omit,procedure,observed):
        data=self.data if omit is None else self.data.loc[self._rowstate!=omit].copy()
        schedule=dict(zip(self.states,a));data=data.copy();data['__timing']=data[self.cluster].map(schedule)
        if self.kind=='undid':
            from . import undid
            panel=undid.prepare_panel_data(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,gvar_col='__timing',covariates=self.covariates)
            return undid._compute_core(panel,self.cluster,self.time,'__timing','both',False)['att']['estimate']
        if self.kind=='didint':
            from . import fitted_didint
            fit=fitted_didint.prepare_fitted_first_stage(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,gvar_col='__timing',covariates=self.covariates,ccc='int')
            return fitted_didint.run_fitted_core(fit,fit.panel)['att']['estimate']
        if self.kind=='bjs':
            from .estimators import EstimatorCache,estimate_bjs
            cache=EstimatorCache.from_data(data,outcome_col=self.outcome,state_col=self.cluster,time_col=self.time,covariates=self.covariates)
            result=estimate_bjs(cache,schedule)
            if result.status!='ok':raise InferenceError('rank_deficient',str(result.diagnostics))
            return result.att
        z=np.column_stack([pd.get_dummies(data[self.cluster],dtype=float),pd.get_dummies(data[self.time],drop_first=True,dtype=float),data[self.covariates].to_numpy(float)])
        d=((data['__timing']>0)&(data[self.time]>=data['__timing'])).to_numpy(float)
        if procedure=='add':
            obs=dict(zip(self.states,observed if observed is not None else self.assignment));g=data[self.cluster].map(obs)
            z=np.column_stack([z,((g>0)&(data[self.time]>=g)).to_numpy(float)])
        x=np.column_stack([z,d]);beta,_,rank,_=np.linalg.lstsq(x,data[self.outcome].to_numpy(float),rcond=None)
        if rank<x.shape[1]:raise InferenceError('rank_deficient')
        return beta[-1]
