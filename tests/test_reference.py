import unittest
import numpy as np
import pandas as pd
from fasttwfepy import ols_reference

class ReferenceTest(unittest.TestCase):
    def test_known_additive_model(self):
        d=pd.DataFrame([(h,t,j) for h in range(6) for t in range(5) for j in range(3)],columns=['h','t','j'])
        D=((d.h<3)&(d.t>=2)).astype(float)
        d['x']=np.cos(d.h*d.t+d.j)
        for covs in [[],['x']]:
            d['y']=d.h+2*d.t+1.7*D+(0.3*d.x if covs else 0)
            self.assertAlmostEqual(ols_reference(d,'y','h','t',D,covs),1.7,places=12)
