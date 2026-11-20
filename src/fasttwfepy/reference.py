"""Direct observation-level OLS oracle, with no cached algebra."""
import numpy as np
import pandas as pd

def ols_reference(data, outcome, cluster, time, treatment, covariates=()):
    z=np.column_stack([pd.get_dummies(data[cluster],dtype=float).to_numpy(),pd.get_dummies(data[time],drop_first=True,dtype=float).to_numpy(),data[list(covariates)].to_numpy(float)])
    x=np.column_stack([z,np.asarray(treatment)])
    b,_,rank,_=np.linalg.lstsq(x,data[outcome].to_numpy(float),rcond=None)
    if rank<x.shape[1]:raise ValueError("rank_deficient")
    return b[-1]
