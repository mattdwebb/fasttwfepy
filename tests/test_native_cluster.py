import unittest
import numpy as np
import pandas as pd
from fasttwfepy import Design


def fixture(seed=1,G=10,T=8,weighted=False,covariates=False,missing=False,effect=1.,heterogeneous=False,unequal=False):
    rng=np.random.default_rng(seed); rows=[]
    a=np.array([5,5,7,7]+[0]*(G-4),float)
    for s in range(G):
        for t in range(1,T+1):
            if missing and s==G-1 and t==3:continue
            for j in range((1+s%7) if unequal else 3):
                x=rng.normal();w=np.exp(rng.normal()) if weighted else 1.
                tau=effect+(s/3-.5 if heterogeneous else 0)
                y=s/7+t/9+(0.4*x if covariates else 0)+rng.normal(scale=.25)+tau*(a[s]>0 and t>=a[s])
                rows.append((s*3+11,t,y,x,w))
    rows=np.array(rows)
    data=dict(zip(('state','year','y','x','w'),rows.T))
    design=Design(pd.DataFrame(data),outcome='y',cluster='state',time='year',assignment=a,
                  covariates=['x'] if covariates else [])
    return data,a,design



class NativeTests(unittest.TestCase):
    def test_native_against_dummy_scores(self):
        for seed in range(8):
            _,_,d=fixture(seed,covariates=bool(seed%2),unequal=bool(seed%3))
            before=d.point(); f=d.native_cluster();r=d.native_cluster(engine='reference')
            np.testing.assert_allclose(f['cluster_scores'],r['cluster_scores'],atol=1e-10,rtol=0)
            self.assertAlmostEqual(f['native_cluster_se'],r['native_cluster_se'],places=10)
            self.assertAlmostEqual(before,d.point(),places=13)
if __name__=='__main__':unittest.main()
